
import numpy as np
from numpy cimport ndarray
cimport cython
cimport numpy as np

from cython.parallel import prange

import numpy as np
from cython.parallel import prange
from numpy cimport ndarray

DTYPE=np.double
ctypedef np.double_t DTYPE_t
ctypedef double complex complex128_t

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_generateTones(ndarray[DTYPE_t] freqList, int nSamples, double sampleRate, ndarray[DTYPE_t] amplitudeList, ndarray[DTYPE_t] phaseList, ndarray[DTYPE_t] iqRatioList, ndarray[DTYPE_t] iqPhaseOffsList):

    # Quantize the frequencies to their closest digital value
    cdef double freqResolution = sampleRate/nSamples
    cdef ndarray[DTYPE_t] quantizedFreqList = np.round(freqList/freqResolution)*freqResolution
    cdef ndarray[DTYPE_t] iqPhaseOffsRadList = np.pi/180.*iqPhaseOffsList
    
    #Declare some types
    cdef ndarray[DTYPE_t, ndim=2] iValList = np.zeros((quantizedFreqList.shape[0],nSamples),dtype=DTYPE)
    cdef ndarray[DTYPE_t, ndim=2] qValList = np.zeros((quantizedFreqList.shape[0],nSamples),dtype=DTYPE)
    cdef double dt = 1. / sampleRate
    cdef ndarray[DTYPE_t] t = dt*np.arange(nSamples, dtype=DTYPE)
    #cdef ndarray[DTYPE_t] phi = np.zeros(nSamples, dtype=DTYPE)
    cdef ndarray[complex128_t] expValues = np.zeros(nSamples, dtype=np.complex128)
    cdef ndarray[DTYPE_t] iScale=np.sqrt(2)*iqRatioList/np.sqrt(1+iqRatioList**2)
    cdef ndarray[DTYPE_t] qScale=np.sqrt(2)/np.sqrt(1+iqRatioList**2)

    
    # generate each signal
    '''
    cdef int i=0
    for i in range(quantizedFreqList.shape[0]):
        expValues = amplitudeList[i]*np.exp(1.j*(2.*np.pi*quantizedFreqList[i]*t+phaseList[i]))
        #iScale = np.sqrt(2)*iqRatioList[i]/np.sqrt(1+iqRatioList[i]**2)
        #qScale = np.sqrt(2)/np.sqrt(1+iqRatioList[i]**2)
        iValList[i,:] = iScale[i]*(np.cos(iqPhaseOffsRadList[i])*np.real(expValues)+np.sin(iqPhaseOffsRadList[i])*np.imag(expValues))
        qValList[i,:] = qScale[i]*np.imag(expValues)
    '''
    cdef int i=0
    for i in prange(quantizedFreqList.shape[0],nogil=True):
        with gil:
            expValues = amplitudeList[i]*np.exp(1.j*(2.*np.pi*quantizedFreqList[i]*t+phaseList[i]))
            expValues2 = amplitudeList[-i]*np.exp(1.j*(3.*np.pi*quantizedFreqList[-i]*t+phaseList[i]))
            #iScale = np.sqrt(2)*iqRatioList[i]/np.sqrt(1+iqRatioList[i]**2)
            #qScale = np.sqrt(2)/np.sqrt(1+iqRatioList[i]**2)
            iValList[i,:] = iScale[i]*(np.cos(iqPhaseOffsRadList[i])*np.real(expValues)+np.sin(iqPhaseOffsRadList[i])*np.imag(expValues))
            qValList[i,:] = qScale[i]*np.imag(expValues)

    return {'I':iValList,'Q':qValList,'quantizedFreqList':quantizedFreqList,'phaseList':phaseList}
    #return iValList








