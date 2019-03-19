import numpy as np
cimport numpy as np
from mkidreadout.readout.mkidshm.pymkidshm import MKIDShmImage
import mkidpipeline.calibration.wavecal as wvl

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy, strcpy

READER_CPU = 1
BIN_WRITER_CPU = 2
SHM_IMAGE_WRITER_CPU = 3
CIRC_BUFF_WRITER_CPU = 4

N_WVL_COEFFS = 3

#WARNING: DO NOT USE IF THERE MAY BE MULTIPLE INSTANCES OF PACKETMASTER;
#         THESE ARE SYSTEM WIDE SEMAPHORES
STREAM_SEM_BASENAME = 'readoutStreamSem'
QUIT_SEM_NAME = 'quitSem'

cdef extern from "<stdint.h>":
    ctypedef unsigned int uint32_t
    ctypedef unsigned long long uint64_t

cdef extern from "packetmaster.h":
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
        char ramdiskPath[80];

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
    cdef object sharedImages
    
    def __init__(self, nRoaches, nRows, nCols, port, useWriter, ramdiskPath=None, wvlSol=None, 
                beammap=None, sharedImageCfg=None, maximizePriority=False):
        """
        Starts the reader (packet receiving) thread along with the appropriate number of parsing 
        threads according to the specified configuration.

        Parameters
        ----------
            nRoaches: int
                Number of ROACH2 boards currently set up to read out the array and stream photons.
            nRows: int
                Number of rows on MKID array
            nCols: int
                Number of columns on MKID array
            port: int
                Port to use for receiving photon stream
            useWriter: bool
                If true, starts the writer thread for writing .bin files to disk
            ramdiskPath: string
                Path to "ramdisk", where writer looks for START and STOP files from dashboard. Required
                if useWriter is True, otherwise not used.
            wvlSol: Wavecal Solution object. 
                Used to fill buffer containing wavecal solution LUT.
            beammap: Beammap object.
                Required if wvlSol is set, otherwise not used.
            sharedImageCfg: yaml config object.
                Configuration object specifying shared memory objects for acquiring realtime images.
                Typical usage would pass a configdict specified in dashboard.yml. Creates/opens 
                MKIDShmImage objects for each image.
                
        """
        #MISC param initialization
        self.nRows = nRows
        self.nCols = nCols

        #DEAL W/ CPU PRIORITY
        if maximizePriority:
            self.readerParams.cpu = READER_CPU
            self.writerParams.cpu = BIN_WRITER_CPU
            self.imageParams.cpu = SHM_IMAGE_WRITER_CPU
        else:
            self.readerParams.cpu = -1
            self.writerParams.cpu = -1
            self.imageParams.cpu = -1

        #INITIALIZE WAVECAL
        self.wavecal.data = <wvlcoeff_t*>malloc(N_WVL_COEFFS*sizeof(wvlcoeff_t)*nRows*nCols)
        memset(self.wavecal.data, 0, N_WVL_COEFFS*sizeof(wvlcoeff_t)*nRows*nCols)
        if wvlSol is not None:
            if beammap is None:
                raise Exception('Must provide a beammap to use a wavecal')
            self.applyWvlSol(wvlSol, beammap)
        
        #INITIALIZE SHARED MEMORY IMAGES
        self.sharedImages = []
        if sharedImageCfg is not None:
            self.nSharedImages = len(sharedImageCfg) 

            self.imageParams.nRoach = nRoaches
            self.imageParams.nSharedImages = self.nSharedImages
            self.imageParams.wavecal = &(self.wavecal)
            self.imageParams.sharedImageNames = <char**>malloc(self.nSharedImages*sizeof(char*))
            for i,image in enumerate(sharedImageCfg):
                nWvlBins = sharedImageCfg[image].nWvlBins if sharedImageCfg[image].has_key('nWvlBins') else 1
                useWvl = sharedImageCfg[image].useWvl if sharedImageCfg[image].has_key('useWvl') else False
                wvlStart = sharedImageCfg[image].wvlStart if sharedImageCfg[image].has_key('wvlStart') else False
                wvlStop = sharedImageCfg[image].wvlStop if sharedImageCfg[image].has_key('wvlStop') else False

                self.sharedImages.append(MKIDShmImage(name=image, 
                                 nRows=nRows, nCols=nCols, useWvl=useWvl,
                                 nWvlBins=nWvlBins, wvlStart=wvlStart,
                                 wvlStop=wvlStop))
                self.imageParams.sharedImageNames[i] = <char*>malloc(STRBUF*sizeof(char*))
                strcpy(self.imageParams.sharedImageNames[i], image.encode('UTF-8'))
                
        else:
            self.nSharedImages = 0 

        #INITIALIZE READOUT STREAMS 
        self.nStreams = 0
        if self.nSharedImages > 0:
            self.nStreams += 1
        if useWriter:
            self.nStreams += 1
        self.streams = <READOUT_STREAM*>malloc(self.nStreams*sizeof(READOUT_STREAM))
        self.readerParams.roachStreamList = self.streams
        self.readerParams.nRoachStreams = self.nStreams

        streamNum = 0
        if self.nSharedImages>0:
            self.imageParams.roachStream = &self.streams[streamNum]
            streamSemName = STREAM_SEM_BASENAME + str(streamNum)
            strcpy(self.imageParams.streamSemName, streamSemName.encode('UTF-8'))
            streamNum += 1

        if useWriter:
            self.writerParams.roachStream = &self.streams[streamNum]
            streamSemName = STREAM_SEM_BASENAME + str(streamNum)
            strcpy(self.writerParams.streamSemName, streamSemName.encode('UTF-8'))
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
            if ramdiskPath is None:
                raise Exception('Must specify ramdisk path to use binWriter')
            strcpy(self.writerParams.ramdiskPath, ramdiskPath.encode('UTF-8'))

        #START THREADS
        self.nThreads = self.nStreams + 1
        self.threads = <THREAD_PARAMS*>malloc((self.nThreads)*sizeof(THREAD_PARAMS))

        startReaderThread(&(self.readerParams), &(self.threads[0]))
        threadNum = 1
        if self.nSharedImages > 0:
            startShmImageWriterThread(&(self.imageParams), &(self.threads[threadNum]))
            threadNum += 1
        if useWriter:
            startBinWriterThread(&(self.writerParams), &(self.threads[threadNum]))
        
 
    def applyWvlSol(self, wvlSol, beammap):
        """
        Fills packetmaster's wavecal buffer with solution specified in wvlSol.
        (Should be!) safe to use while packetmaster threads are running, though
        data will be invalid while writing.

        Parameters
        ----------
            wvlSol: Wavecal Solution object
            beamap: beammap object
        """
        self.wavecal.nCols = self.nCols
        self.wavecal.nRows = self.nRows
        strcpy(self.wavecal.solutionFile, wvlSol._file_path.encode('UTF-8'))

        calResIDs, calCoeffs = wvlSol.find_calibrations()
        a = np.zeros((self.nRows, self.nCols))
        b = np.zeros((self.nRows, self.nCols))
        c = np.zeros((self.nRows, self.nCols))
        resIDMap = beammap.residmap.T

        for i in range(self.nRows):
            for j in range(self.nCols):
                resID = resIDMap[i,j]
                if np.any(resID==calResIDs):
                    curCoeffs = calCoeffs[resID==calResIDs]
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
        """
        Exit all threads
        """
        quitAllThreads(QUIT_SEM_NAME.encode('UTF-8'), self.nThreads)

    def __dealloc__(self):
        free(self.streams)
        for i in range(self.nSharedImages):
            free(self.imageParams.sharedImageNames[i])
        free(self.imageParams.sharedImageNames)
        free(self.threads)
        free(self.wavecal.data)
        


