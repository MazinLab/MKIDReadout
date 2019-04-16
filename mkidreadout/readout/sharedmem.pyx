# Cython wrapper for libmkidshm
# Compile with: cython sharedmem.pyx -o pymkidshm.c
#               gcc pymkidshm.c -fPIC -shared -I/home/neelay/anaconda2/envs/readout/include/python2.7/ -o pymkidshm.so -lmkidshm -lpthread -lrt

import numpy as np
cimport numpy as np
from mkidcore.corelog import getLogger
import os
from libc.string cimport strcpy

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
        char wavecalID[80]

    #PARTIAL DEFINITION, only exposing necessary attributes
    ctypedef struct MKID_IMAGE:
        MKID_IMAGE_METADATA *md

    ctypedef struct MKID_WAVECAL_METADATA:
        uint32_t nCols
        uint32_t nRows

    ctypedef struct MKID_WAVECAL:
        MKID_WAVECAL_METADATA *md
        coeff_t *data

    cdef int MKIDShmImage_open(MKID_IMAGE *imageStruct, char *imgName)
    cdef int MKIDShmImage_close(MKID_IMAGE *imageStruct)
    cdef int MKIDShmImage_create(MKID_IMAGE_METADATA *imageMetadata, char *imgName, MKID_IMAGE *outputImage)
    cdef int MKIDShmImage_populateMD(MKID_IMAGE_METADATA *imageMetadata, char *name, int nCols, int nRows, int useWvl, int nWvlBins, int useEdgeBins, int wvlStart, int wvlStop)
    cdef int MKIDShmImage_startIntegration(MKID_IMAGE *image, uint64_t startTime, uint64_t integrationTime)
    cdef int MKIDShmImage_wait(MKID_IMAGE *image, int semInd)
    cdef int MKIDShmImage_timedwait(MKID_IMAGE *image, int semInd, int time, int stopImage)
    cdef int MKIDShmImage_checkIfDone(MKID_IMAGE *image, int semInd)
    cdef void MKIDShmImage_copy(MKID_IMAGE *image, image_t *outputBuffer)


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
        if not name[0]=='/':
            name = '/'+name
        if os.path.isfile(os.path.join('/dev/shm', name[1:])):
            self._open(name)
            paramsMatch = True
            if kwargs.get('nCols') is not None:
                paramsMatch &= (kwargs.get('nCols') == self.image.md.nCols)
            if kwargs.get('nRows') is not None:
                paramsMatch &= (kwargs.get('nRows') == self.image.md.nRows)
            if kwargs.get('useWvl') is not None:
                paramsMatch &= (int(kwargs.get('useWvl')) == self.image.md.useWvl)
            if kwargs.get('nWvlBins') is not None:
                paramsMatch &= (kwargs.get('nWvlBins') == self.image.md.nWvlBins)
            if kwargs.get('useEdgeBins') is not None:
                paramsMatch &= (kwargs.get('useEdgeBins') == self.image.md.useEdgeBins)
            if kwargs.get('wvlStart') is not None:
                paramsMatch &= (kwargs.get('wvlStart') == self.image.md.wvlStart)
            if kwargs.get('wvlStop') is not None:
                paramsMatch &= (kwargs.get('wvlStop') == self.image.md.wvlStop)
            if not paramsMatch:
                raise Exception('Image already exists, and provided parameters do not match.')

        else:
            self._create(name, kwargs.get('nCols', 100), kwargs.get('nRows', 100), kwargs.get('useWvl', False), 
                        kwargs.get('nWvlBins', 1), kwargs.get('useEdgeBins', False), kwargs.get('wvlStart', 0), kwargs.get('wvlStop', 0))

    def _create(self, name, nCols, nRows, useWvl, nWvlBins, useEdgeBins, wvlStart, wvlStop):
        cdef MKID_IMAGE_METADATA imagemd
        MKIDShmImage_populateMD(&imagemd, name.encode('UTF-8'), nCols, nRows, int(useWvl), nWvlBins, int(useEdgeBins), wvlStart, wvlStop)
        MKIDShmImage_create(&imagemd, name.encode('UTF-8'), &(self.image));

    def _open(self, name):
        MKIDShmImage_open(&(self.image), name.encode('UTF-8'))

    def startIntegration(self, startTime=0, integrationTime=1):
        """
        Tells packetmaster to start an integration for this image
        Parameters
        ----------
            startTime: double
                image start time (in seconds since 00:00 Jan 1 UTC of current year). 
                If 0, start immediately w/ timestamp that packetmaster is currently parsing.
            integrationTime: double
                integration time in seconds
        """
        startTime = int(startTime*2000)
        integrationTime = int(integrationTime*2000) #convert to half-ms
        MKIDShmImage_startIntegration(&(self.image), startTime, integrationTime)

    def receiveImage(self):
        """
        Waits for doneImage semaphore to be posted by packetmaster,
        then grabs the image from buffer
        """
        retval = MKIDShmImage_timedwait(&(self.image), self.doneSemInd, self.image.md.integrationTime, 1)
        if retval == -1:
            getLogger(__name__).warning("Image timeout")
        flatImage = self._readImageBuffer()
        if not self.valid:
            raise RuntimeError('Wavecal parameters changed during integration!')
        if self.useWvl:
            return np.reshape(flatImage, self._shape)
        else:
            flatImage = flatImage[:self._shape[1]*self._shape[2]]
            return np.reshape(flatImage, (self._shape[1], self._shape[2]))

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
        return '' if not self.useWvl else self.image.md.wavecalID.decode()

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

    @useWvl.setter
    def useWvl(self, use):
        self.invalidate()
        self.image.md.useWvl = 1 if use else 0

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

    def update_wvlStop(self, wvl):
        self.wvlStop = float(wvl)

    def update_wvlStart(self, wvl):
        self.wvlStart = float(wvl)

    @property 
    def valid(self):
        return bool(self.image.md.valid)
    
        

