
import numpy as np
import pdb
import time
from functools import partial
from multiprocessing import Pool
from numba import jit


def loadImgFiles(fnList, nRows, nCols):
    imageList = []
    for fn in fnList:
        image = np.fromfile(open(fn, mode='rb'),dtype=np.uint16)
        image = np.transpose(np.reshape(image, (nCols, nRows)))
        imageList.append(image)
    return np.asarray(imageList)

def crossCorrelate(template, data):
    '''
    returns np.correlate's cross correlation of the template with the data.
    The max of the correlation is where your timestream matches up
    '''
    corr = []
    for i in range(len(data)):
        corr.append(np.correlate(template, data[i], mode='full'))
    corr = np.array(corr)
    return corr

def xcorr(x):
    """FFT based autocorrelation function, which is faster than numpy.correlate"""
    # x is supposed to be an array of sequences, of shape (totalelements, length)
    fftx = np.fft.fft(x, n=(len(x[0])*2-1), axis=1)
    ret = np.fft.ifft(fftx * np.conjugate(fftx), axis=1)
    ret = np.fft.fftshift(ret, axes=1)
    return ret

def xCrossCorr(x,y):
    fftx = np.fft.rfft(x)
    ffty = np.fft.rfft(y)
    ret = np.fft.irfft(fftx * np.conjugate(ffty),n = len(x))
    ret = np.fft.fftshift(ret)
    return ret

def determineSelfconsistentPixelLocs2(corrMatrix, a):
    """
    This function ranks the pixels by least variance
    """
    corrMatrix2 = corrMatrix - a[:,np.newaxis]
    medDelays = np.median(corrMatrix2,axis=0)
    corrMatrix2 = corrMatrix2 - medDelays[np.newaxis,:]
    #totalVar = np.var(corrMatrix2,axis=1)
    totalVar = np.sum(np.abs(corrMatrix2)<=1,axis=1)
    bestPixels = np.argsort(totalVar)[::-1]
    #pdb.set_trace()
    return bestPixels, np.sort(totalVar)[::-1]

def crossCorrelateImageTimestreams(imageList):
    """
    Cross correlate each pixels timestream with every other pixel's
    
    INPUTS:
        imageList - List of images with pixel counts (3d array)
    OUTPUTS:
        correlationMatrix - cross correlation of each pixels timestream with every other
                            shape: nCols*nRows X nCols*nRows X nTimeSamples
                            row i, col j is the cross correlation of pixel i's timesteam with pixel j.
    """
    shape=np.shape(imageList)   #nTimeSamples X nRows X nCols
    fftImage = np.fft.rfft(imageList,axis=0)   # fft of each pixel's timestream
    fftImage=np.rollaxis(fftImage,0,3)                                  # now nRows X nCols X nFreqs
    fftImage = np.reshape(fftImage, (shape[1]*shape[2],-1))        # reshape into list of timestream ffts
    fftCorrelationMatrix = np.einsum('ik,jk->ijk',fftImage,np.conj(fftImage))   # multiply each timestream with the conjugate of every other timestream elementwise
    correlationMatrix = np.fft.irfft(fftCorrelationMatrix,n=shape[2],axis=2)       # inverse fourier transform to complete correlation calculation
    correlationMatrix = np.fft.fftshift(correlationMatrix,axis=2)
    return correlationMatrix

def crossCorrelateTimestreams(timestreams, minCounts=5, maxCounts=2499):
    """
    This cross correlates every 'good' pixel with every other 'good' pixel.
    
    Inputs: 
        timestreams - List of timestreams
        minCounts - The minimum number of total counts across all time to be good
        maxCounts - The maximum number of counts during a single time frame to be considered good
    
    Outputs:
        correlationList - List of cross correlation products for good pixels
                          Shape: [(len(goodPix)-1)*len(goodPix)/2, time]
                          The first len(goodPix)-1 arrays are for pixel 0 cross correlated with the n-1 other pixels
                          The next len(goodPix)-2 arrays are for pixel 1 cross correlated with pixels 2,3,4....
                          etc.
        goodPix - List of indices 'i' of good pixels. 
    """
    nTime = len(timestreams[0])
    bkgndList = 1.0*np.median(timestreams,axis=1)
    nCountsList = 1.0*np.sum(timestreams,axis=1)
    maxCountsList = 1.0*np.amax(timestreams,axis=1)
    goodPix = np.where((nCountsList>minCounts) * (maxCountsList<maxCounts) * (bkgndList < nCountsList/nTime))[0]
    print "Num good Pix: "+str(len(goodPix))

    #Normalize the timestreams for cross correlation and remove bad pixels
    timestreams=timestreams[goodPix] - bkgndList[goodPix,np.newaxis]            # subtract background
    timestreams = timestreams / (1.0*nCountsList[goodPix,np.newaxis]/nTime)      # divide by avg count rate

    print "taking fft..."
    fftImage = np.fft.rfft(timestreams, axis=1)                       # fft the timestream
    #del timestreams
    print "...Done"

    fftCorrelationList=np.zeros(((len(goodPix)-1)*len(goodPix)/2,nTime))
    def crossCorrelate_i(index):
        corrList = np.multiply(fftImage[index,:], np.conj(fftImage)[index+1:,:])
        startIndex = len(goodPix)*index - index*(index+1)/2
        endIndex = startIndex+len(goodPix) - index - 1
        
        corrList=np.fft.irfft(corrList, n=nTime, axis=1)
        corrList = np.fft.fftshift(corrList,axes=1)
        
        fftCorrelationList[ startIndex : endIndex, :] = corrList
    
    print "Cross correlating..."
    startTime=time.time()
    for i in range(len(goodPix)-1):
        crossCorrelate_i(i)         # could be fast if we use multiprocessing
        #cor = np.multiply(fftImage[i,:],np.conj(fftImage)[i,:])
        #cor = np.fft.irfft(cor,n=shape[2])
        #cor = np.fft.fftshift(cor)
        #print np.argmax(cor)
    print "...cross Correlate: "+str((time.time()-startTime)*1000)+' ms'
    #pdb.set_trace()
    
    #del fftImage
    fftImage=None
    #fftCorrelationList = None
    
    print "Inverse fft..."
    #correlationList=np.fft.irfft(fftCorrelationList, n=shape[2], axis=1)
    #correlationList = np.fft.fftshift(correlationList,axes=1)
    correlationList=fftCorrelationList
    print "...Done"

    #pdb.set_trace()
    return correlationList, goodPix

def crossCorrelateImageTimestreams2(imageList, minCounts=5, maxCounts=2499):
    """
    This cross correlates every 'good' pixel with every other 'good' pixel.
    
    Inputs: 
        imageList - List of 2D images in shape [x, y, time]
        minCounts - The minimum number of total counts across all time to be good
        maxCounts - The maximum number of counts during a single time frame to be considered good
    
    Outputs:
        correlationList - List of cross correlation products for good pixels
                          Shape: [(len(goodPix)-1)*len(goodPix)/2, time]
                          The first len(goodPix)-1 arrays are for pixel 0 cross correlated with the n-1 other pixels
                          The next len(goodPix)-2 arrays are for pixel 1 cross correlated with pixels 2,3,4....
                          etc.
        goodPix - List of indices 'i' of good pixels. 
                  I think x=i/x_Max, y=i%x_Max or something like that. Need to check
    """
    imageList=np.rollaxis(imageList, 0, 3)
    #imageList = imageList[:,:,:-2]
    shape=imageList.shape
    imageList=np.reshape(imageList,(shape[0]*shape[1],shape[2]))
    bkgndList = 1.0*np.median(imageList,axis=1)
    nCountsList = 1.0*np.sum(imageList,axis=1)
    maxCountsList = 1.0*np.amax(imageList,axis=1)
    goodPix = np.where((nCountsList>minCounts) * (maxCountsList<maxCounts) * (bkgndList < nCountsList/shape[2]))[0]
    print "Num good Pix: "+str(len(goodPix))
    goodPix = goodPix[:1000]
    #goodPix = np.asarray([2002,2117,2126])
    
    #Normalize the images for cross correlation and remove bad pixels
    imageList=imageList[goodPix] - bkgndList[goodPix,np.newaxis]            # subtract background
    imageList = imageList / (nCountsList[goodPix,np.newaxis]/shape[2])      # divide by avg count rate
    #imageList=imageList[goodPix]/maxCountsList[goodPix,np.newaxis]                                    # nGoodPix X nTimeSamples array
    #imageList=imageList[goodPix]
    
    print "taking fft..."
    fftImage = np.fft.rfft(imageList, axis=1)                       # fft the timestream
    del imageList
    print "...Done"
    
    #fftCorrelationList=np.zeros(((len(goodPix)-1)*len(goodPix)/2,fftImage.shape[1]),dtype=np.complex128)
    fftCorrelationList=np.zeros(((len(goodPix)-1)*len(goodPix)/2,shape[2]))
    def crossCorrelate_i(index):
        corrList = np.multiply(fftImage[index,:], np.conj(fftImage)[index+1:,:])
        startIndex = len(goodPix)*index - index*(index+1)/2
        endIndex = startIndex+len(goodPix) - index - 1
        
        corrList=np.fft.irfft(corrList, n=shape[2], axis=1)
        corrList = np.fft.fftshift(corrList,axes=1)
        
        fftCorrelationList[ startIndex : endIndex, :] = corrList
    
    print "Cross correlating..."
    startTime=time.time()
    for i in range(len(goodPix)-1):
        crossCorrelate_i(i)         # could be fast if we use multiprocessing
        #cor = np.multiply(fftImage[i,:],np.conj(fftImage)[i,:])
        #cor = np.fft.irfft(cor,n=shape[2])
        #cor = np.fft.fftshift(cor)
        #print np.argmax(cor)
    print "...cross Correlate: "+str((time.time()-startTime)*1000)+' ms'
    #pdb.set_trace()
    
    #del fftImage
    fftImage=None
    #fftCorrelationList = None
    
    print "Inverse fft..."
    #correlationList=np.fft.irfft(fftCorrelationList, n=shape[2], axis=1)
    #correlationList = np.fft.fftshift(correlationList,axes=1)
    correlationList=fftCorrelationList
    print "...Done"

    #pdb.set_trace()
    return correlationList, goodPix
    
@jit    
def minimizePixelLocationVariance(corrMatrix, weights=None):
    """
    This function is a bit tricky to understand.
    
    The corrMatrix describes the relative distance between pixels. For example:
    (row, col)=(i, j)=L_ij is the distance between pixel i and j. This is in 
    units of times frames (usually each frame is 1 second) which correspond 
    to the beammap bar moving across the array.
    
    Now consider 3 pixels. The distance between pixel 1 and 2 should be 
    the same as the distance between [[pixel 3 - pixel 1] - [pixel 3 - pixel 2]].
    
    Call the absolute position of each pixel a_i. Then the above statement is:
    L_ij - L_kj = a_i - a_k for all i,j,k. However, there is measurement error 
    and systematics (hot pixels) so it's just approximately equal.
    
    Mathematically we can turn this into a least squares minimization problem:
    minimize the sum over all i,j,k of (L_ij - L_kj - a_i + a_k)^2. If we restrict
    the sum over j to be only over the best pixels, then it is like weighting
    the best pixels with 1 and the bad pixels with 0. Or you can choose arbitrary weights
    
    Of course, we only know relative distances, so we arbitrarily set the
    absolute location of the first pixel to 0. 
    
    Inputs:
        corrMatrix - (i,j) is the distance between pixel i and pixel j
                      The shape can be [all pixels, best pixels]
        weights - list of weights for each pixel. If not given, then assume equal weighting.
    
    Returns:
        a - vector where a_i is the location of pixel i relative to a_0 = 0. 
    """
    shape=corrMatrix.shape
    n = shape[0]
    if weights is None: weights = np.ones(shape[1])*1.0/n
    assert len(weights) == shape[1]

    Q = np.ones((n-1,n-1))*-1.*np.sum(weights**2)/n**2
    Q[np.diag_indices(n-1)] = np.sum(weights**2)*(1.0/n - 1.0/n**2.)

    b=np.zeros(n)
    for k in range(1,n):
        bk = 0
        for i in range(0,n):
            for j in range(0,shape[1]):
                #pdb.set_trace()
                bk += (weights[j]**2.)*(corrMatrix[k,j] - corrMatrix[i,j])
        b[k] = bk*1.0/n**2.

    a = np.zeros(n)
    a[1:] = np.dot(np.linalg.inv(Q), b[1:])


    #q1 = cal_q(a,corrMatrix)
    #print q1
    #q2 = cal_q(np.array([-6.5,-11.5]), corrMatrix)
    #print q2

    #q3 = cal_q(np.array([-20./3.,-37./3]), corrMatrix)
    #print q3

    return a
    
@jit    
def cal_q(a, corrMatrix, weights=None):
    """
    Calculate the quadratic form with the minimizer a
    
    Inputs:
        a - minimizer returned from minimizePixelLocationVariance()
        corrMatrix - matrix of relative pixel locations
        weights - optional weights on the pixels
        
    Outputs:
        q - Value of quadratic form
        C - array of values of constant term in quadratic form that can't be minimized away
            Sum to get the total C
    """
    n=corrMatrix.shape[0]
    m=corrMatrix.shape[1]
    if weights is None: weights = np.ones(m)*1.0/n
    assert len(weights) == m
    q=0.
    for i in range(n):
        for j in range(m):
            for k in range(n):
                q+= weights[j]**2*((corrMatrix[i,j] - a[i]) - (corrMatrix[k,j] - a[k]))**2.
    
    C=np.zeros(n)
    for i in range(n):
        C_i=0
        for j in range(m):
            for k in range(n):
                C_i+=(weights[j]**2.)*(corrMatrix[i,j] - corrMatrix[k,j])**2.
        C[i]=C_i
    
    return q/(2.*n**2.), C/(2.*n**2.)
    
    
    
    


