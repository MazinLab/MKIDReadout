import cPickle as pickle
import os
import subprocess

import numpy as np
cimport numpy as np
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset, strcpy
from mkidreadout.readout.sharedmem import ImageCube, EventBuffer

from mkidcore.corelog import getLogger

READER_CPU = 1
BIN_WRITER_CPU = 2
SHM_IMAGE_WRITER_CPU = 3
CIRC_BUFF_WRITER_CPU = 4

N_WVL_COEFFS = 3

LO_IP = '127.0.0.1'

#WARNING: DO NOT USE IF THERE MAY BE MULTIPLE INSTANCES OF PACKETMASTER;
#         THESE ARE SYSTEM WIDE SEMAPHORES
QUIT_SEM_NAME = 'packetmaster_quitSem'
RINGBUF_RESET_SEM_NAME = 'packetmaseter_ringbuff'

cdef extern from "<stdint.h>":
    ctypedef unsigned int uint32_t
    ctypedef unsigned long long uint64_t
    ctypedef unsigned char uint8_t

cdef extern from "pmthreads.h":
    cdef int STRBUF
    cdef int SHAREDBUF
    cdef int RINGBUF_SIZE
    ctypedef float wvlcoeff_t
    ctypedef struct READER_PARAMS:
        int port;
        int nRoachStreams;
        RINGBUFFER *packBuf;

        char quitSemName[80];
        char ringBufResetSemName[80];

        int cpu; #if cpu=-1 then don't maximize priority
    
    ctypedef struct BIN_WRITER_PARAMS:
        RINGBUFFER *packBuf;

        int writing;
        char writerPath[80];

        char quitSemName[80];
        char ringBufResetSemName[80];

        int cpu; 
    
    ctypedef struct SHM_IMAGE_WRITER_PARAMS:
        RINGBUFFER *packBuf;
        int nRoach;
        int nSharedImages;
        char **sharedImageNames;
        WAVECAL_BUFFER *wavecal; #if NULL don't use wavecal

        char quitSemName[80];
        char ringBufResetSemName[80];

        int cpu; #if cpu=-1 then don't maximize priority
    
    ctypedef struct EVENT_BUFF_WRITER_PARAMS:
        RINGBUFFER *packBuf;
        char bufferName[80];
        WAVECAL_BUFFER *wavecal; #if NULL don't use wavecal

        char quitSemName[80]; 
        char ringBufResetSemName[80];

        int nRows;
        int nCols;

        int cpu; #if cpu=-1 then don't maximize priority

    ctypedef struct WAVECAL_BUFFER:
        char solutionFile[80];
        int writing;
        uint32_t nCols;
        uint32_t nRows;
        # Each pixel has 3 coefficients, with address given by 
        # &a = 3*(nCols*y + x); &b = &a + 1; &c = &a + 2
        wvlcoeff_t *data;

    ctypedef struct READOUT_STREAM:
        uint64_t unread;
        char data[536870912];
    
    ctypedef struct RINGBUFFER:
        uint8_t data[536870912];
        uint64_t writeInd;
        uint64_t nCycles;
    
    ctypedef struct THREAD_PARAMS:
        pass

    cdef int startReaderThread(READER_PARAMS *rparams, THREAD_PARAMS *tparams);
    cdef int startBinWriterThread(BIN_WRITER_PARAMS *rparams, THREAD_PARAMS *tparams);
    cdef int startShmImageWriterThread(SHM_IMAGE_WRITER_PARAMS *rparams, THREAD_PARAMS *tparams);
    cdef int startEventBuffWriterThread(EVENT_BUFF_WRITER_PARAMS *rparams, THREAD_PARAMS *tparams);
    cdef void resetSem(const char *semName);
    cdef void quitAllThreads(const char *quitSemName, int nThreads);

cdef class Packetmaster(object): 
    """
    Receives and parses photon events for the MKID readout. This class is a python frontend for 
    the code in packetmaster/packetmaster.c. 
    """
    cdef BIN_WRITER_PARAMS writerParams
    cdef SHM_IMAGE_WRITER_PARAMS imageParams
    cdef EVENT_BUFF_WRITER_PARAMS eventBuffParams
    cdef READER_PARAMS readerParams
    cdef WAVECAL_BUFFER wavecal
    cdef RINGBUFFER packBuf
    cdef THREAD_PARAMS *threads
    cdef int nRows
    cdef int nCols
    cdef int nThreads
    cdef int nSharedImages
    cdef readonly object sharedImages
    cdef readonly object eventBuffer
    cdef readonly object samplicatorProcess

    #TODO useWriter->savebinfiles, ramdiskPath->ramdisk ?use '' as default?
    def __init__(self, nRoaches, port, nRows=None, nCols=None, useWriter=True, wvlCoeffs=None,
                 beammap=None, sharedImageCfg=None, eventBuffCfg=None, maximizePriority=False, 
                 recreate_images=False, forwarding=None):
        """
        Starts the reader (packet receiving) thread along with the appropriate number of parsing 
        threads according to the specified configuration.

        Parameters
        ----------
            nRoaches: int
                Number of ROACH2 boards currently set up to read out the array and stream photons.
            port: int
                Port to use for receiving photon stream
            nRows: int
                Number of rows on MKID array, required in no beammap, ignored if beammap
            nCols: int
                Number of columns on MKID array, required in no beammap, ignored if beammap
            useWriter: bool
                If true, starts the writer thread for writing .bin files to disk
            ramdiskPath: string
                Path to "ramdisk", where writer looks for START and STOP files from dashboard. Required
                if useWriter is True, otherwise not used.
            wvlCoeffs: dict 
                Contains cal coefficients and corresponding resIDs. Used to fill buffer containing 
                wavecal solution LUT.
            beammap: Beammap object.
                Required if wvlCoeffs is set, used for nRows and nCols if present
            sharedImageCfg: yaml config object.
                Configuration object specifying shared memory objects for acquiring realtime images.
                Typical usage would pass a configdict specified in dashboard.yml. Creates/opens 
                ImageCube objects for each image.
                Object must have keys corresponding to the names of the images, and values must have a get method for
                valid for the attributes n_wave_bins, use_wave, wave_start, wave_stop (i.e. a ConfigThing or a dict)
            eventBuffCfg: yaml config object (or dict)
                Config dict specifying name and size of event buffer. If None no event buffer is created/used.
            recreate_images: bool
                Remove and recreate the shared images if true
            forwarding: dict or yaml config object
                Use if you want to forward photon packets to another machine (i.e. RTC). keys:
                    localport: port that packetmaster listens to on local machine ('port'
                        is the port that samplicator will bind to; localport must be different)
                    destIP: IP address to forward packets to
                    destport: port to use for forwarded IP
        """

        #TODO: modify to include circular buffer
        if recreate_images and sharedImageCfg is not None:
            for k in sharedImageCfg:
                f = '/dev/shm/{}'.format(k)
                if os.path.exists(f):
                    os.remove(f)
                    os.remove(f+'.buf')

        try:
            self.nRows = int(beammap.nrows)
            self.nCols = int(beammap.ncols)
        except AttributeError:
            if nRows is None or nCols is None:
                raise ValueError('nRows and nCols must be set if no beammap specified')
            self.nRows = int(nRows)
            self.nCols = int(nCols)

        npix = self.nRows*self.nCols

        #DEAL W/ CPU PRIORITY
        if maximizePriority:
            self.readerParams.cpu = READER_CPU
            self.writerParams.cpu = BIN_WRITER_CPU
            self.imageParams.cpu = SHM_IMAGE_WRITER_CPU
        else:
            self.readerParams.cpu = -1
            self.writerParams.cpu = -1
            self.imageParams.cpu = -1
        
        #INITIALIZE SHARED MEMORY IMAGES
        self.sharedImages = {}
        if sharedImageCfg is not None:
            self.imageParams.nRoach = nRoaches
            self.imageParams.nSharedImages = len(sharedImageCfg)
            self.imageParams.wavecal = &(self.wavecal)
            self.imageParams.sharedImageNames = <char**>malloc(len(sharedImageCfg)*sizeof(char*))
            for i,image in enumerate(sharedImageCfg):
                self.sharedImages[image] = ImageCube(name=image, nRows=self.nRows, nCols=self.nCols,
                                                     useWvl=sharedImageCfg[image].get('use_wave', False),
                                                     nWvlBins=sharedImageCfg[image].get('n_wave_bins', 1),
                                                     wvlStart=sharedImageCfg[image].get('wave_start', False),
                                                     wvlStop=sharedImageCfg[image].get('wave_stop', False))
                self.imageParams.sharedImageNames[i] = <char*>malloc(STRBUF*sizeof(char*))
                strcpy(self.imageParams.sharedImageNames[i], image.encode('UTF-8'))

        #INITIALIZE EVENT BUFFER
        print 'initializing event buffer...'
        self.eventBuffer = None
        if eventBuffCfg is not None:
            self.eventBuffParams.nRows = self.nRows
            self.eventBuffParams.nCols = self.nCols
            self.eventBuffParams.wavecal = &(self.wavecal)
            self.eventBuffer = EventBuffer(eventBuffCfg['name'], eventBuffCfg['size'])
            strcpy(self.eventBuffParams.bufferName, eventBuffCfg['name'].encode('UTF8'))

        #INITIALIZE WAVECAL
        print 'initializing wavecal...'
        self.wavecal.data = <wvlcoeff_t*>malloc(N_WVL_COEFFS*sizeof(wvlcoeff_t)*npix)
        memset(self.wavecal.data, 0, N_WVL_COEFFS*sizeof(wvlcoeff_t)*npix)
        if wvlCoeffs is not None:
            if beammap is None:
                raise Exception('Must provide a beammap to use a wavecal')
            self.applyWvlSol(wvlCoeffs, beammap)



        #INITIALIZE QUIT SEM
        strcpy(self.imageParams.quitSemName, QUIT_SEM_NAME.encode('UTF-8'))
        strcpy(self.eventBuffParams.quitSemName, QUIT_SEM_NAME.encode('UTF-8'))
        strcpy(self.writerParams.quitSemName, QUIT_SEM_NAME.encode('UTF-8'))
        strcpy(self.readerParams.quitSemName, QUIT_SEM_NAME.encode('UTF-8'))

        #INITIALIZE PACKBUF SEM
        strcpy(self.imageParams.ringBufResetSemName, RINGBUF_RESET_SEM_NAME.encode('UTF-8'))
        strcpy(self.eventBuffParams.ringBufResetSemName, RINGBUF_RESET_SEM_NAME.encode('UTF-8'))
        strcpy(self.writerParams.ringBufResetSemName, RINGBUF_RESET_SEM_NAME.encode('UTF-8'))
        strcpy(self.readerParams.ringBufResetSemName, RINGBUF_RESET_SEM_NAME.encode('UTF-8'))

        #SETUP IP FORWARDING
        self.samplicatorProcess = None
        if forwarding is not None:
            if port == forwarding['localport']:
                raise Exception('forwarding["localport"] and "port" must be different!')
            argstring = ['samplicate', '-p', str(port), LO_IP+'/'+str(forwarding['localport']), 
                            forwarding['destIP']+'/'+str(forwarding['destport'])]
            self.samplicatorProcess = subprocess.Popen(argstring)
            port = forwarding['localport']
            
        #INITIALIZE REMAINING PARAMS
        self.readerParams.port = port
        self.readerParams.packBuf = &self.packBuf
        self.nThreads = 1
        if useWriter:
            self.writerParams.writing = 0
            self.writerParams.packBuf = &self.packBuf
            self.nThreads += 1
        if self.sharedImages:
            self.imageParams.packBuf = &self.packBuf
            self.nThreads += 1
        if self.eventBuffer:
            self.nThreads += 1
            self.eventBuffParams.packBuf = &self.packBuf

        #START THREADS
        self.threads = <THREAD_PARAMS*>malloc((self.nThreads)*sizeof(THREAD_PARAMS))

        resetSem(QUIT_SEM_NAME.encode('UTF-8'))
        resetSem(RINGBUF_RESET_SEM_NAME.encode('UTF-8'))
        startReaderThread(&(self.readerParams), &(self.threads[0]))
        threadNum = 1
        print 'starting shared image thread'
        if self.sharedImages:
            startShmImageWriterThread(&(self.imageParams), &(self.threads[threadNum]))
            threadNum += 1
        print 'starting event buffer thread'
        if self.eventBuffer:
            startEventBuffWriterThread(&(self.eventBuffParams), &(self.threads[threadNum]))
            threadNum += 1
        if useWriter:
            startBinWriterThread(&(self.writerParams), &(self.threads[threadNum]))

    def startWriting(self, binDir=None):
        if binDir is not None:
            if binDir[-1] != '/':
                binDir += '/'
            strcpy(self.writerParams.writerPath, binDir.encode('UTF-8'))
        self.writerParams.writing = 1

    def stopWriting(self):
        self.writerParams.writing = 0

    def applyWvlSol(self, wvlCoeffs, beammap):
        """
        Fills packetmaster's wavecal buffer with coefficients specified in wvlCoeffs.
        (Should be!) safe to use while packetmaster threads are running, though
        data will be invalid while writing.

        Parameters
        ----------
            wvlCoeffs: coefficient dict saved by wavecal Solution
            beamap: beammap object
        """
        self.wavecal.nCols = self.nCols
        self.wavecal.nRows = self.nRows
        strcpy(self.wavecal.solutionFile, str(wvlCoeffs['solution_file_path']).encode('UTF-8'))

        for image in self.sharedImages:
            self.sharedImages[image].invalidate()

        calCoeffs = wvlCoeffs['calibrations']
        calResIDs = wvlCoeffs['res_ids']
        a = np.zeros((self.nRows, self.nCols))
        b = np.zeros((self.nRows, self.nCols))
        c = np.zeros((self.nRows, self.nCols))
        resIDMap = beammap.residmap.T

        for i,j in np.ndindex(self.nRows, self.nCols):
            curCoeffs = calCoeffs[resIDMap[i,j]==calResIDs]
            if curCoeffs.size:
                curCoeffs = curCoeffs[0]
                a[i,j] = curCoeffs[0]
                b[i,j] = curCoeffs[1]
                c[i,j] = curCoeffs[2]

        a = a.flatten()
        b = b.flatten()
        c = c.flatten()

        coeffArray = np.zeros(N_WVL_COEFFS*self.nRows*self.nCols)
        coeffArray[0::3] = a
        coeffArray[1::3] = b
        coeffArray[2::3] = c
        coeffArray = coeffArray.astype(np.single) #convert to float (ASSUMES wvlcoeff_t is float!)

        self.wavecal.writing = 1
        memcpy(self.wavecal.data, <wvlcoeff_t*>np.PyArray_DATA(coeffArray), N_WVL_COEFFS*self.nRows*self.nCols*sizeof(wvlcoeff_t))
        self.wavecal.writing = 0

    def quit(self):
        """ Exit all threads """
        quitAllThreads(QUIT_SEM_NAME.encode('UTF-8'), self.nThreads)
        if self.samplicatorProcess is not None:
            self.samplicatorProcess.terminate()

    def __dealloc__(self):
        for i in range(len(self.sharedImages)):
            free(self.imageParams.sharedImageNames[i])
        free(self.imageParams.sharedImageNames)
        free(self.threads)
        free(self.wavecal.data)
        


