# Cython wrapper for libmkidshm
cimport numpy as np

cdef extern from "<stdint.h>":
    ctypedef unsigned int uint32_t
    ctypedef unsigned long long uint64_t

cdef extern from "mkidshm.h":
    ctypedef struct MKID_IMAGE_METADATA:
        pass
    ctypedef struct MKID_IMAGE:
        pass
    cdef int openMKIDShmImage(MKID_IMAGE *imageStruct, char *imgName)
    cdef int closeMKIDShmImage(MKID_IMAGE *imageStruct)
    cdef int createMKIDShmImage(MKID_IMAGE_METADATA *imageMetadata, char *imgName, MKID_IMAGE *outputImage)
    cdef int populateImageMD(MKID_IMAGE_METADATA *imageMetadata, char *name, int nXPix, int nYPix, int useWvl, int nWvlBins, int wvlStart, int wvlStop)
    cdef int startIntegration(uint64_t startTime)
    cdef int waitForImage()
    cdef int checkDoneImage()



cdef class MKIDShmImage:
    cdef MKID_IMAGE image

    def __cinit__(self):
        pass

    def create(self, name, nXPix, nYPix, useWvl=0, nWvlBins=1, wvlStart=0, wvlStop=0):
        cdef MKID_IMAGE_METADATA imagemd
        populateImageMD(&imagemd, name.encode('UTF-8'), nXPix, nYPix, useWvl, nWvlBins, wvlStart, wvlStop)
        createMKIDShmImage(&imagemd, name.encode('UTF-8'), &(self.image));

    def open(self, name):
        openMKIDShmImage(&(self.image), name.encode('UTF-8'))
