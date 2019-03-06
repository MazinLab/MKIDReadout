# Cython wrapper for libmkidshm
# Compile with: cython pymkidshm.pyx -o pymkidshm.c
#               gcc pymkidshm.c -fPIC -shared -I/home/neelay/anaconda2/envs/readout/include/python2.7/ -o pymkidshm.so -lmkidshm -lpthread -lrt

import numpy as np
cimport numpy as np
import os

cdef extern from "<stdint.h>":
    ctypedef unsigned int uint32_t
    ctypedef unsigned long long uint64_t

cdef extern from "mkidshm.h":
    ctypedef int image_t
    ctypedef float coeff_t

    #PARTIAL DEFINITION, only exposing necessary attributes
    ctypedef struct MKID_IMAGE_METADATA:
        uint32_t nXPix
        uint32_t nYPix
        uint32_t useWvl
        uint32_t nWvlBins
        uint32_t wvlStart
        uint32_t wvlStop

    #PARTIAL DEFINITION, only exposing necessary attributes
    ctypedef struct MKID_IMAGE:
        MKID_IMAGE_METADATA *md

    ctypedef struct MKID_WAVECAL_METADATA:
        uint32_t nXPix
        uint32_t nYPix

    ctypedef struct MKID_WAVECAL:
        MKID_WAVECAL_METADATA *md
        coeff_t *data

    cdef int MKIDShmImage_open(MKID_IMAGE *imageStruct, char *imgName)
    cdef int MKIDShmImage_close(MKID_IMAGE *imageStruct)
    cdef int MKIDShmImage_create(MKID_IMAGE_METADATA *imageMetadata, char *imgName, MKID_IMAGE *outputImage)
    cdef int MKIDShmImage_populateMD(MKID_IMAGE_METADATA *imageMetadata, char *name, int nXPix, int nYPix, int useWvl, int nWvlBins, int wvlStart, int wvlStop)
    cdef int MKIDShmImage_startIntegration(MKID_IMAGE *image, uint64_t startTime, uint64_t integrationTime)
    cdef int MKIDShmImage_wait(MKID_IMAGE *image, int semInd)
    cdef int MKIDShmImage_checkIfDone(MKID_IMAGE *image, int semInd)
    cdef void MKIDShmImage_copy(MKID_IMAGE *image, image_t *outputBuffer);

    cdef int MKIDShmWavecal_open(MKID_WAVECAL *wavecal, const char *name);
    cdef int MKIDShmWavecal_close(MKID_WAVECAL *wavecal);
    cdef int MKIDShmWavecal_create(MKID_WAVECAL_METADATA *wavecalMetadata, const char *name, MKID_WAVECAL *outputWavecal);
    cdef coeff_t MKIDShmWavecal_getEnergy(MKID_WAVECAL *wavecal, int x, int y, float phase);



cdef class MKIDShmImage(object):
    cdef MKID_IMAGE image
    cdef int doneSemInd

    def __init__(self, name, doneSemInd=0, **kwargs):
        self.doneSemInd = doneSemInd
        if not name[0]=='/':
            name = '/'+name
        if os.path.isfile(os.path.join('/dev/shm', name[1:])):
            self.open(name)
            paramsMatch = True
            if kwargs.get('nXPix') is not None:
                paramsMatch &= (kwargs.get('nXPix') == self.image.md.nXPix)
            if kwargs.get('nYPix') is not None:
                paramsMatch &= (kwargs.get('nYPix') == self.image.md.nYPix)
            if kwargs.get('useWvl') is not None:
                paramsMatch &= (int(kwargs.get('useWvl')) == self.image.md.useWvl)
            if kwargs.get('nWvlBins') is not None:
                paramsMatch &= (kwargs.get('nWvlBins') == self.image.md.nWvlBins)
            if kwargs.get('wvlStart') is not None:
                paramsMatch &= (kwargs.get('wvlStart') == self.image.md.wvlStart)
            if kwargs.get('wvlStop') is not None:
                paramsMatch &= (kwargs.get('wvlStop') == self.image.md.wvlStop)
            if not paramsMatch:
                raise Exception('Image already exists, and provided parameters do not match.')

        else:
            self.create(name, kwargs.get('nXPix', 100), kwargs.get('nYPix', 100), kwargs.get('useWvl', False), 
                        kwargs.get('nWvlBins', 1), kwargs.get('wvlStart', 0), kwargs.get('wvlStop', 0))
            
         
    
    def create(self, name, nXPix, nYPix, useWvl, nWvlBins, wvlStart, wvlStop):
        cdef MKID_IMAGE_METADATA imagemd
        MKIDShmImage_populateMD(&imagemd, name.encode('UTF-8'), nXPix, nYPix, int(useWvl), nWvlBins, wvlStart, wvlStop)
        MKIDShmImage_create(&imagemd, name.encode('UTF-8'), &(self.image));

    def open(self, name):
        MKIDShmImage_open(&(self.image), name.encode('UTF-8'))

    def startIntegration(self, startTime=0, integrationTime=1):
        """
        Tells packetmaster to start an integration for this image
        Parameters
        ----------
            startTime: image start time (in seconds UTC?). If 0, start immediately w/
                timestamp that packetmaster is currently parsing.
            integrationTime: integration time in seconds(?)
        """
        MKIDShmImage_startIntegration(&(self.image), startTime, integrationTime)

    def receiveImage(self):
        """
        Waits for doneImage semaphore to be posted by packetmaster,
        then grabs the image from buffer
        """
        MKIDShmImage_wait(&(self.image), self.doneSemInd)
        return self._readImageBuffer()

    def _checkIfDone(self):
        """
        Non blocking. Returns True if image is done (doneImageSem is posted),
        False otherwise. Basically a wrapper for sem_trywait
        """
        return (MKIDShmImage_checkIfDone(&(self.image), self.doneSemInd) == 0)


    def _readImageBuffer(self):
        imageSize = self.image.md.nXPix * self.image.md.nYPix * self.image.md.nWvlBins
        imageBuffer = np.empty(imageSize, dtype=np.intc)
        MKIDShmImage_copy(&(self.image), <image_t*>np.PyArray_DATA(imageBuffer))
        return imageBuffer

    @property
    def dims(self):
        return [self.image.md.nXPix, self.image.md.nYPix]

    @property
    def nWvlBins(self):
        return self.image.md.nWvlBins

    @property
    def useWvl(self):
        return self.image.md.useWvl

    @property
    def wvlStart(self):
        return self.image.md.wvlStart

    @property
    def wvlStop(self):
        return self.image.md.wvlStop
    
cdef class RealtimeWavecal(object):

    cdef MKID_WAVECAL wavecalshm;
    
    def __init__(self, name, sol=None, beammap=None):
        self.sol = sol
        self.beammap = beammap
        self.name = name
        if not name[0]=='/':
            name = '/'+name
        if os.path.isfile(os.path.join('/dev/shm', name[1:])):
            if sol is not None:
                raise Exception('Solution buffer already exists!')
            else:
                MKIDShmWavecal_open(&(self.wavecalshm), name.encode('UTF-8'))
        else:
            if beammap is None:
                raise Exception('Must provide beammap to create solution buffer!')
            self.create()
            self.applySol(sol, beammap)

    def create(self):
        cdef MKID_WAVECAL_METADATA md
        md.nXPix = self.beammap.ncols
        md.nYPix = self.beammap.nrows
        MKIDShmWavecal_create(&md, self.name.encode('UTF-8'), &(self.wavecalshm))
        
        

    def applySol(self, sol, beammap):
        self.sol = sol
        self.beammap = beammap
        

