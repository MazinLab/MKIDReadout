# Cython wrapper for libmkidshm
# Compile with: cython sharedmem.pyx -o pymkidshm.c
#               gcc pymkidshm.c -fPIC -shared -I/home/neelay/anaconda2/envs/readout/include/python2.7/ -o pymkidshm.so -lmkidshm -lpthread -lrt

import numpy as np
cimport numpy as np
import datetime
import calendar
from mkidcore.corelog import getLogger
import os
from libc.string cimport strcpy

DEFAULT_EVENT_BUFFER_SIZE = 200000 #total number of events

cdef extern from "<stdint.h>":
    ctypedef unsigned int uint32_t
    ctypedef unsigned long long uint64_t

cdef extern from "<semaphore.h>":
    ctypedef union sem_t:
        pass

cdef extern from "mkidshm.h":
    ctypedef int image_t
    ctypedef float coeff_t

    #PARTIAL DEFINITION, only exposing necessary attributes
    ctypedef struct MKID_IMAGE_METADATA:
        uint32_t nCols
        uint32_t nRows
        uint32_t useWvl
        uint32_t nWvlBins
        uint32_t useEdgeBins
        uint32_t wvlStart
        uint32_t wvlStop
        uint32_t valid
        uint32_t integrationTime
        char name[80]
        char wavecalID[150]

    #PARTIAL DEFINITION, only exposing necessary attributes
    ctypedef struct MKID_IMAGE:
        MKID_IMAGE_METADATA *md

    #PARTIAL DEFINITION, only exposing necessary attributes
    ctypedef struct MKID_EVENT_BUFFER_METADATA:
        uint32_t size
        int endInd
        int writing
        int nCycles
        int useWvl
    
        char name[80]
        char eventBufferName[80]
        char newPhotonSemName[80]
        char wavecalID[150]
    
    #PARTIAL DEFINITION, only exposing necessary attributes
    ctypedef struct MKID_EVENT_BUFFER:
        MKID_EVENT_BUFFER_METADATA *md
    
    cdef int MKIDShmImage_open(MKID_IMAGE *imageStruct, char *imgName)
    cdef int MKIDShmImage_close(MKID_IMAGE *imageStruct)
    cdef int MKIDShmImage_create(MKID_IMAGE_METADATA *imageMetadata, char *imgName, MKID_IMAGE *outputImage)
    cdef int MKIDShmImage_populateMD(MKID_IMAGE_METADATA *imageMetadata, char *name, int nCols, int nRows, int useWvl, int nWvlBins, int useEdgeBins, int wvlStart, int wvlStop)
    cdef int MKIDShmImage_startIntegration(MKID_IMAGE *image, uint64_t startTime, uint64_t integrationTime)
    cdef int MKIDShmImage_wait(MKID_IMAGE *image, int semInd)
    cdef int MKIDShmImage_timedwait(MKID_IMAGE *image, int semInd, int time, int stopImage) nogil
    cdef int MKIDShmImage_checkIfDone(MKID_IMAGE *image, int semInd)
    cdef void MKIDShmImage_copy(MKID_IMAGE *image, image_t *outputBuffer)

    cdef int MKIDShmEventBuffer_open(MKID_EVENT_BUFFER *bufferStruct, const char *bufferName)
    cdef int MKIDShmEventBuffer_create(MKID_EVENT_BUFFER_METADATA *bufferMetadata, const char *bufferName, MKID_EVENT_BUFFER *outputBuffer)
    cdef int MKIDShmEventBuffer_populateMD(MKID_EVENT_BUFFER_METADATA *metadata, const char *name, int size, int useWvl)


cdef class ImageCube(object):
    """
    Python interface to MKID shared memory image defined in mkidshm.h (MKID_IMAGE struct)
    """
    cdef MKID_IMAGE image
    cdef int doneSemInd

    def __init__(self, name, doneSemInd=0, **kwargs):
        """
        Opens or creates a MKID_IMAGE shared memory buffer specified by name (should be located 
        in /dev/shm/name). 
        
        Parameters
        ----------
            name: string
                Name of shared memory buffer. If buffer exists, opens it, else create.
            doneSemInd: int
                Index of semaphore to wait on when receiving an image. Should be 0
                unless multiple processes are using the same image.
            kwargs:
                nRows: int (default: 100)
                nCols: int (default: 100)
                useWvl: bool (default: False)
                nWvlBins: bool (default: 1)
                useEdgeBins: bool (default: False)
                wvlStart: float (default: 0)
                wvlStop: float (default: 0)

        """

        self.doneSemInd = doneSemInd

        if not name.startswith('/'):
            name = '/'+name
        if os.path.isfile(os.path.join('/dev/shm', name[1:])):
            self._open(name)
            paramsMatch = True
            if kwargs.get('nCols') is not None:
                paramsMatch &= (kwargs.get('nCols') == self.image.md.nCols)
            if kwargs.get('nRows') is not None:
                paramsMatch &= (kwargs.get('nRows') == self.image.md.nRows)
            if kwargs.get('nWvlBins') is not None:
                paramsMatch &= (kwargs.get('nWvlBins') == self.image.md.nWvlBins)
            if kwargs.get('useEdgeBins') is not None:
                paramsMatch &= (kwargs.get('useEdgeBins') == bool(self.image.md.useEdgeBins))
            if not paramsMatch:
                raise Exception('Image already exists, and provided parameters do not match.')

            if kwargs.get('useWvl') is not None:
                self.set_useWvl(kwargs.get('useWvl'))
            if kwargs.get('wvlStart') is not None:
                self.set_wvlStart(kwargs.get('wvlStart'))
            if kwargs.get('wvlStop') is not None:
                self.set_wvlStop(kwargs.get('wvlStop'))

        else:
            self._create(name, kwargs.get('nCols', 100), kwargs.get('nRows', 100), kwargs.get('useWvl', False), 
                        kwargs.get('nWvlBins', 1), kwargs.get('useEdgeBins', False), kwargs.get('wvlStart', 0), kwargs.get('wvlStop', 0))

    def _create(self, name, nCols, nRows, useWvl, nWvlBins, useEdgeBins, wvlStart, wvlStop):
        cdef MKID_IMAGE_METADATA imagemd
        MKIDShmImage_populateMD(&imagemd, name.encode('UTF-8'), nCols, nRows, int(useWvl), nWvlBins, int(useEdgeBins), wvlStart, wvlStop)
        rval = MKIDShmImage_create(&imagemd, name.encode('UTF-8'), &(self.image));
        if rval != 0:
            raise Exception('Error opening shared memory file')

    def _open(self, name):
        rval = MKIDShmImage_open(&(self.image), name.encode('UTF-8'))
        if rval != 0:
            raise Exception('Error opening shared memory file')

    def startIntegration(self, startTime=0, integrationTime=1):
        """
        Tells packetmaster to start an integration for this image
        Parameters
        ----------
            startTime: double
                image start time (in seconds UTC)
                If 0, start immediately w/ timestamp that packetmaster is currently parsing.
            integrationTime: double
                integration time in seconds
        """
        if startTime != 0:
            curYr = datetime.datetime.utcnow().year
            yrStart = datetime.date(curYr, 1, 1)
            tsOffs = calendar.timegm(yrStart.timetuple())
            startTime -= tsOffs

        startTime = int(startTime*2000)
        integrationTime = int(integrationTime*2000) #convert to half-ms
        MKIDShmImage_startIntegration(&(self.image), startTime, integrationTime)

    def receiveImage(self):
        """
        Waits for doneImage semaphore to be posted by packetmaster,
        then grabs the image from buffer
        """
        with nogil:
            retval = MKIDShmImage_timedwait(&(self.image), self.doneSemInd, self.image.md.integrationTime, 1)
        flatImage = self._readImageBuffer()
        if not self.valid:
            raise RuntimeError('Wavecal parameters changed during integration!')
        if self.useWvl:
            return np.reshape(flatImage, self._shape).squeeze()
        else:
            return np.reshape(flatImage[:self._shape[1]*self._shape[2]],
                              (self._shape[1], self._shape[2]))

    def _checkIfDone(self):
        """
        Non blocking. Returns True if image is done (doneImageSem is posted),
        False otherwise. Basically a wrapper for sem_trywait
        """
        return (MKIDShmImage_checkIfDone(&(self.image), self.doneSemInd) == 0)


    def _readImageBuffer(self):
        imageSize = self._shape[0] * self._shape[1] * self._shape[2]
        imageBuffer = np.empty(imageSize, dtype=np.intc)
        MKIDShmImage_copy(&(self.image), <image_t*>np.PyArray_DATA(imageBuffer))
        return imageBuffer

    def invalidate(self):
        """
        Use to indicate (permissible) changes in image parameters (wvl ranges,
        wavecal, etc). Invalidates current image if integrating
        """
        self.image.md.valid = 0

    @property
    def wavecalID(self):
        return '' if not self.useWvl else self.image.md.wavecalID.decode(encoding='UTF-8')

    @property
    def name(self):
        return self.image.md.name.decode()

    @property
    def shape(self):
        if self.useWvl:
            return self._shape
        else:
            return self.image.md.nRows, self.image.md.nCols

    @property
    def _shape(self):
        if self.useEdgeBins:
            depth = self.nWvlBins + 2
        else:
            depth = self.nWvlBins
        return depth, self.image.md.nRows, self.image.md.nCols

    @property
    def nWvlBins(self):
        return self.image.md.nWvlBins

    @property
    def wvlBinEdges(self):
        wvlBinEdges = np.linspace(self.wvlStart, self.wvlStop, self.nWvlBins + 1)
        if self.useEdgeBins:
            wvlBinEdges = np.insert(wvlBinEdges, 0, 0)
            wvlBinEdges = np.append(wvlBinEdges, np.inf)
        return wvlBinEdges

    @property
    def useWvl(self):
        return self.image.md.useWvl

    @property 
    def useEdgeBins(self):
        return self.image.md.useEdgeBins

    def set_useWvl(self, use):
        if self.image.md.useWvl == use:
            return
        self.invalidate()
        self.image.md.useWvl = 1 if use else 0
        msg = 'Wavecal application to data in shared image {} {}.'
        getLogger('mkidreadout.readout.sharedmem').debug(msg.format(self.image.md.name.decode(),
                                                                    'enabled' if use else ' disabled'))

    @useWvl.setter
    def useWvl(self, use):
        self.set_useWvl(use)

    @property
    def wvlStart(self):
        return self.image.md.wvlStart

    @wvlStart.setter
    def wvlStart(self, wvl):
        self.invalidate()
        self.image.md.wvlStart = wvl

    @property
    def wvlStop(self):
        return self.image.md.wvlStop

    @wvlStop.setter
    def wvlStop(self, wvl):
        self.invalidate()
        self.image.md.wvlStop = wvl

    def set_wvlStop(self, wvl):
        self.wvlStop = float(wvl)

    def set_wvlStart(self, wvl):
        self.wvlStart = float(wvl)

    @property 
    def valid(self):
        return bool(self.image.md.valid)
    
cdef class EventBuffer:
    cdef MKID_EVENT_BUFFER eventBuffer;

    def __init__(self, name, size=None):
        """
        Opens a photon event buffer given by name (file in /dev/shm).
        Creates it if it doesn't exist.

        Parameters
        ----------
            name: string
                name of event buffer (in /dev/shm)
            size: int
                Number of photon events stored in buffer.
                default: 200000

        """
        
        if not name.startswith('/'):
            name = '/'+name
        if os.path.isfile(os.path.join('/dev/shm', name[1:])):
            self._open(name)
            if size is not None:
                if size != self.eventBuffer.md.size:
                    raise Exception('Buffer already exists, and provided size does not match.')

        else:
            if size is None:
                size = DEFAULT_EVENT_BUFFER_SIZE
            self._create(name, size)

    def _create(self, name, size):
        cdef MKID_EVENT_BUFFER_METADATA md
        MKIDShmEventBuffer_populateMD(&md, name.encode('UTF-8'), size, 0)
        rval = MKIDShmEventBuffer_create(&md, name.encode('UTF-8'), &(self.eventBuffer));
        if rval != 0:
            raise Exception('Error opening shared memory file')

    def _open(self, name):
        rval = MKIDShmEventBuffer_open(&(self.eventBuffer), name.encode('UTF-8'))
        if rval != 0:
            raise Exception('Error opening shared memory file')

    @property
    def size(self):
        return self.eventBuffer.md.size

