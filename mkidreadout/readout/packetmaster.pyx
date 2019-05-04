import cPickle as pickle
import os

import numpy as np
cimport numpy as np
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset, strcpy
from mkidreadout.readout.sharedmem import ImageCube

import mkidpipeline.calibration.wavecal as wvl
from mkidcore.corelog import getLogger

READER_CPU = 1
BIN_WRITER_CPU = 2
SHM_IMAGE_WRITER_CPU = 3
CIRC_BUFF_WRITER_CPU = 4

N_WVL_COEFFS = 3

#WARNING: DO NOT USE IF THERE MAY BE MULTIPLE INSTANCES OF PACKETMASTER;
#         THESE ARE SYSTEM WIDE SEMAPHORES
STREAM_SEM_BASENAME = 'packetmaster_readoutStreamSem'
QUIT_SEM_NAME = 'packetmaster_quitSem'

cdef extern from "<stdint.h>":
    ctypedef unsigned int uint32_t
    ctypedef unsigned long long uint64_t

cdef extern from "pmthreads.h":
    cdef int STRBUF
    cdef int SHAREDBUF
    ctypedef float wvlcoeff_t
    ctypedef struct READER_PARAMS:
        int port;
        int nRoachStreams;
        READOUT_STREAM *roachStreamList;
        char streamSemBaseName[80]; #append 0, 1, 2, etc for each name

        char quitSemName[80];

        int cpu; #if cpu=-1 then don't maximize priority
    
    ctypedef struct BIN_WRITER_PARAMS:
        READOUT_STREAM *roachStream;

        int writing;
        char writerPath[80];

        char quitSemName[80];
        char streamSemName[80];

        int cpu; 
    
    ctypedef struct SHM_IMAGE_WRITER_PARAMS:
        READOUT_STREAM *roachStream;
        int nRoach;
        int nSharedImages;
        char **sharedImageNames;
        WAVECAL_BUFFER *wavecal; #if NULL don't use wavecal

        char quitSemName[80];
        char streamSemName[80];

        int cpu; #if cpu=-1 then don't maximize priority
    
    ctypedef struct CIRC_BUFF_WRITER_PARAMS:
        READOUT_STREAM *roachStream;
        char bufferName[80];
        WAVECAL_BUFFER *wavecal; #if NULL don't use wavecal

        char quitSemName[80];
        char streamSemName[80];

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
    
    ctypedef struct THREAD_PARAMS:
        pass

    cdef int startReaderThread(READER_PARAMS *rparams, THREAD_PARAMS *tparams);
    cdef int startBinWriterThread(BIN_WRITER_PARAMS *rparams, THREAD_PARAMS *tparams);
    cdef int startShmImageWriterThread(SHM_IMAGE_WRITER_PARAMS *rparams, THREAD_PARAMS *tparams);
    cdef void quitAllThreads(const char *quitSemName, int nThreads);

cdef class Packetmaster(object): 
    """
    Receives and parses photon events for the MKID readout. This class is a python frontend for 
    the code in packetmaster/packetmaster.c. 
    """
    cdef BIN_WRITER_PARAMS writerParams
    cdef SHM_IMAGE_WRITER_PARAMS imageParams
    cdef READER_PARAMS readerParams
    cdef WAVECAL_BUFFER wavecal
    cdef READOUT_STREAM *streams
    cdef THREAD_PARAMS *threads
    cdef int nRows
    cdef int nCols
    cdef int nStreams
    cdef int nThreads
    cdef int nSharedImages
    cdef readonly object sharedImages

    #TODO useWriter->savebinfiles, ramdiskPath->ramdisk ?use '' as default?
    def __init__(self, nRoaches, port, nRows=None, nCols=None, useWriter=True, wvlSol=None,
                 beammap=None, sharedImageCfg=None, maximizePriority=False, recreate_images=False):
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
            wvlSol: Wavecal Solution object. 
                Used to fill buffer containing wavecal solution LUT.
            beammap: Beammap object.
                Required if wvlSol is set, used for nRows and nCols if present
            sharedImageCfg: yaml config object.
                Configuration object specifying shared memory objects for acquiring realtime images.
                Typical usage would pass a configdict specified in dashboard.yml. Creates/opens 
                ImageCube objects for each image.
                Object must have keys corresponding to the names of the images, and values must have a get method for
                valid for the attributes n_wave_bins, use_wave, wave_start, wave_stop (i.e. a ConfigThing or a dict)
            recreate_images: bool
                Remove and recreate the shared images if true
        """

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

        #INITIALIZE WAVECAL
        self.wavecal.data = <wvlcoeff_t*>malloc(N_WVL_COEFFS*sizeof(wvlcoeff_t)*npix)
        memset(self.wavecal.data, 0, N_WVL_COEFFS*sizeof(wvlcoeff_t)*npix)
        if wvlSol is not None:
            if beammap is None:
                raise Exception('Must provide a beammap to use a wavecal')
            self.applyWvlSol(wvlSol, beammap)

        #INITIALIZE READOUT STREAMS 
        self.nStreams = 0
        if self.sharedImages:
            self.nStreams += 1
        if useWriter:
            self.nStreams += 1
        self.streams = <READOUT_STREAM*>malloc(self.nStreams*sizeof(READOUT_STREAM))
        self.readerParams.roachStreamList = self.streams
        self.readerParams.nRoachStreams = self.nStreams

        streamNum = 0
        if self.sharedImages:
            self.imageParams.roachStream = &self.streams[streamNum]
            strcpy(self.imageParams.streamSemName, (STREAM_SEM_BASENAME + str(streamNum)).encode('UTF-8'))
            streamNum += 1

        if useWriter:
            self.writerParams.roachStream = &self.streams[streamNum]
            strcpy(self.writerParams.streamSemName, (STREAM_SEM_BASENAME + str(streamNum)).encode('UTF-8'))
            streamNum += 1

        strcpy(self.readerParams.streamSemBaseName, STREAM_SEM_BASENAME.encode('UTF-8'))

        #INITIALIZE QUIT SEM
        strcpy(self.imageParams.quitSemName, QUIT_SEM_NAME.encode('UTF-8'))
        strcpy(self.imageParams.quitSemName, QUIT_SEM_NAME.encode('UTF-8'))
        strcpy(self.writerParams.quitSemName, QUIT_SEM_NAME.encode('UTF-8'))
        strcpy(self.readerParams.quitSemName, QUIT_SEM_NAME.encode('UTF-8'))

        #INITIALIZE REMAINING PARAMS
        self.readerParams.port = port
        if useWriter:
            self.writerParams.writing = 0

        #START THREADS
        self.nThreads = self.nStreams + 1
        self.threads = <THREAD_PARAMS*>malloc((self.nThreads)*sizeof(THREAD_PARAMS))

        startReaderThread(&(self.readerParams), &(self.threads[0]))
        threadNum = 1
        if self.sharedImages:
            startShmImageWriterThread(&(self.imageParams), &(self.threads[threadNum]))
            threadNum += 1
        if useWriter:
            startBinWriterThread(&(self.writerParams), &(self.threads[threadNum]))

    def startWriting(self, binDir=None):
        if binDir is not None:
            strcpy(self.writerParams.writerPath, binDir.encode('UTF-8'))
        self.writerParams.writing = 1

    def stopWriting(self):
        self.writerParams.writing = 0

    def applyWvlSol(self, wvlSol_file, beammap):
        """
        Fills packetmaster's wavecal buffer with solution specified in wvlSol.
        (Should be!) safe to use while packetmaster threads are running, though
        data will be invalid while writing.

        Parameters
        ----------
            wvlSol: Wavecal Solution object
            beamap: beammap object
        """
        wvlSol = wvl.Solution(wvlSol_file) #make sure the solution isn't just a file name
        self.wavecal.nCols = self.nCols
        self.wavecal.nRows = self.nRows
        strcpy(self.wavecal.solutionFile, wvlSol._file_path.encode('UTF-8'))

        for image in self.sharedImages:
            self.sharedImages[image].invalidate()

        try:

            with open('cache_{}.pickle'.format(os.path.basename(wvlSol_file))) as f:
                a,b,c = pickle.load(f)
            getLogger(__name__).info('Using cache '+'cache_{}.pickle'.format(os.path.basename(wvlSol_file)))
        except IOError:
            calCoeffs, calResIDs = wvlSol.find_calibrations()
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
            with open('cache_{}.pickle'.format(os.path.basename(wvlSol_file)),'w') as f:
                pickle.dump((a,b,c), f, protocol=2)
            getLogger(__name__).info('Wrote cache '+'cache_{}.pickle'.format(os.path.basename(wvlSol_file)))

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

    def __dealloc__(self):
        free(self.streams)
        for i in range(len(self.sharedImages)):
            free(self.imageParams.sharedImageNames[i])
        free(self.imageParams.sharedImageNames)
        free(self.threads)
        free(self.wavecal.data)
        


