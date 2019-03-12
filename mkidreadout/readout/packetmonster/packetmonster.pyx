cimport numpy as np
from pymkidshm import MKIDShmImage

from libc.stdlib cimport malloc, free
from libc.string cimport memset, strcpy

READER_CPU = 1
BIN_WRITER_CPU = 2
SHM_IMAGE_WRITER_CPU = 3
CIRC_BUFF_WRITER_CPU = 4

#WARNING: DO NOT USE IF THERE MAY BE MULTIPLE INSTANCES OF PACKETMASTER;
#         THESE ARE SYSTEM WIDE SEMAPHORES
STREAM_SEM_BASENAME = 'readoutStreamSem'
QUIT_SEM_NAME = 'quitSem'

cdef extern from "<stdint.h>":
    ctypedef unsigned int uint32_t
    ctypedef unsigned long long uint64_t

cdef extern from "packetmonster.h":
    cdef int STRBUF
    cdef int SHAREDBUF
    ctypedef float wvlcoeff_t
    ctypedef struct READER_PARAMS:
        int port;
        int nRoachStreams;
        READOUT_STREAM **roachStreamList;
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
        uint32_t nXPix;
        uint32_t nYPix;
        # Each pixel has 3 coefficients, with address given by 
        # &a = 3*(nXPix*y + x); &b = &a + 1; &c = &a + 2
        wvlcoeff_t *data;

    ctypedef struct READOUT_STREAM:
        uint64_t unread;
        char data[536870912];
    
    ctypedef struct THREAD_PARAMS:
        pass

cdef class PacketMonster(object): 
    cdef BIN_WRITER_PARAMS writerParams;
    cdef SHM_IMAGE_WRITER_PARAMS imageParams;
    cdef READER_PARAMS readerParams;
    cdef WAVECAL_BUFFER wavecal;
    cdef READOUT_STREAM *streams;
    
    def __init__(self, nRoaches, nRows, nCols, port, useWriter, wvlSol=None, 
                beammap=None, sharedImageCfg=None, maximizePriority=False):
        #MISC param initialization
        self.nRows = nRows
        self.nCols = nCols

        #DEAL W/ CPU PRIORITY
        if maximizePriority:
            self.readerParams.cpu = -1
            self.writerParams.cpu = -1
            self.imageParams.cpu = -1
        else:
            self.readerParams.cpu = READER_CPU
            self.writerParams.cpu = BIN_WRITER_CPU
            self.imageParams.cpu = SHM_IMAGE_WRITER_CPU

        #INITIALIZE WAVECAL
        self.wavecal.data = <wvlcoeff_t*>malloc(sizeof(wvlcoeff_t)*nRows*nCols)
        memset(self.wavecal.data, 0, sizeof(wvlcoeff_t)*nRows*nCols)
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
                nWvlBins = image.nWvlBins if image.has_key('nWvlBins') else 1
                useWvl = image.useWvl if image.has_key('useWvl') else False
                wvlStart = image.wvlStart if image.has_key('wvlStart') else False
                wvlStop = image.wvlStop if image.has_key('wvlStop') else False

                self.sharedImages.append(MKIDShmImage(name=image.name, 
                                 nYPix=nRows, nXPix=nCols, useWvl=useWvl,
                                 nWvlBins=nWvlBins, wvlStart=wvlStart,
                                 wvlStop=wvlStop))
                strcpy(self.imageParams.sharedImageNames[i], image.name.encode('UTF-8'))
                
        else:
            self.nSharedImages = 0 

        #INITIALIZE READOUT STREAMS 
        self.nStreams = 0
        if self.nSharedImages > 0:
            self.nStreams += 1
        if useWriter:
            self.nStreams += 1
        self.streams = <READOUT_STREAM*>malloc(self.nStreams*sizeof(READOUT_STREAM))

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

        #INITIALIZE QUIT SEM
        strcpy(self.imageParams.quitSemName, QUIT_SEM_NAME.encode('UTF-8'))
        strcpy(self.imageParams.quitSemName, QUIT_SEM_NAME.encode('UTF-8'))
        strcpy(self.writerParams.quitSemName, QUIT_SEM_NAME.encode('UTF-8'))
        strcpy(self.readerParams.quitSemName, QUIT_SEM_NAME.encode('UTF-8'))
 
    def applyWvlSol(self, wvlSol, beammap):
        self.wavecal.writing = 1
        self.wavecal.nXPix = self.nCols
        self.wavecal.nYPix = self.nRows
        strcpy(wvlSol.solutionFile, wvlSol._file_path.encode('UTF-8'))

    def __dealloc__(self):
        free(self.streams)


