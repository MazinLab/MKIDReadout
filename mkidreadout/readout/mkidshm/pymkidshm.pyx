# Cython wrapper for libmkidshm
# Compile with: cython pymkidshm.pyx -o pymkidshm.c
#               gcc pymkidshm.c -fPIC -shared -I/home/neelay/anaconda2/envs/readout/include/python2.7/ -o pymkidshm.so -lmkidshm -lpthread -lrt

cimport numpy as np

cdef extern from "<stdint.h>":
    ctypedef unsigned int uint32_t
    ctypedef unsigned long long uint64_t

cdef extern from "mkidshm.h":
    ctypedef struct MKID_IMAGE_METADATA:
        pass
    ctypedef struct MKID_IMAGE:
        pass
    ctypedef int image_t
    cdef int openMKIDShmImage(MKID_IMAGE *imageStruct, char *imgName)
    cdef int closeMKIDShmImage(MKID_IMAGE *imageStruct)
    cdef int createMKIDShmImage(MKID_IMAGE_METADATA *imageMetadata, char *imgName, MKID_IMAGE *outputImage)
    cdef int populateImageMD(MKID_IMAGE_METADATA *imageMetadata, char *name, int nXPix, int nYPix, int useWvl, int nWvlBins, int wvlStart, int wvlStop)
    cdef int startIntegration(uint64_t startTime)
    cdef int waitForImage(MKID_IMAGE *image)
    cdef int checkDoneImage(MKID_IMAGE *image)



cdef class MKIDShmImage(object):
    cdef MKID_IMAGE image

    def __cinit__(self):
        pass

    def create(self, name, nXPix, nYPix, useWvl=0, nWvlBins=1, wvlStart=0, wvlStop=0):
        cdef MKID_IMAGE_METADATA imagemd
        populateImageMD(&imagemd, name.encode('UTF-8'), nXPix, nYPix, useWvl, nWvlBins, wvlStart, wvlStop)
        createMKIDShmImage(&imagemd, name.encode('UTF-8'), &(self.image));

    def open(self, name):
        openMKIDShmImage(&(self.image), name.encode('UTF-8'))

    def startIntegration(self, startTime=0, integrationTime=1):
        """
        Tells packetmaster to start an integration for this image
        Parameters
        ----------
            startTime: image start time (in seconds UTC?). If 0, start immediately w/
                timestamp that packetmaster is currently parsing.
            integrationTime: integration time in seconds(?)
        """
        pass

    def receiveImage(self):
        """
        Waits for doneImage semaphore to be posted by packetmaster,
        then grabs the image from buffer
        """
        _wait(self)
        return _readImageBuffer(self)

    def _wait(self):
        """
        Blocking. Waits for doneImageSem to be posted by packetmaster
        """
        waitForImage(&(self.image))

    def _checkIfDone(self):
        """
        Non blocking. Returns True if image is done (doneImageSem is posted),
        False otherwise. Basically a wrapper for sem_trywait
        """
        return (checkDoneImage(&(self.image)) == 0)


    def _readImageBuffer(self):
        pass

    
