
import numpy as np
import pdb
import os
import time
from functools import partial
from multiprocessing import Pool
from numba import jit
import ConfigParser
import scipy.optimize as spo
from mkidreadout.configuration.beammap.flags import beamMapFlags

MEC_FL_WIDTH = 14
MEC_FL_LENGTH = 146
DARKNESS_FL_WIDTH = 25
DARKNESS_FL_LENGTH = 80

N_FL_MEC = 10
N_FL_DARKNESS = 5

def getFLCoordRangeDict(FLmap):
    """
    INPUTS:
        FLmap - map of array with values equal to the FL. see getFLMap()
    OUTPUTS:
        coordFLRangeDict - Dictionary with key = FL# and value = [xMin, xMax, yMin, yMax] for that FL
    """
    FLs = np.unique(FLmap)
    coordFLRangeDict={}
    for FL in FLs:
        inds = np.where(FLmap==FL)
        xMin = np.amin(inds[0])
        xMax = np.amax(inds[0])
        yMin = np.amin(inds[1])
        yMax = np.amax(inds[1])
        coordFLRangeDict[FL] = [xMin, xMax, yMin, yMax]
    return coordFLRangeDict

def getFLCoordRangeMaps(FLmap):
    """
    INPUTS:
        FLmap - map of array with values equal to the FL. see getFLMap()
    OUTPUTS:
        xMinFLmap - map of array with value equal to the minimum allowable x coordinate for that feedline
        xMaxFLmap
        yMinFLmap
        yMaxFLmap
    """
    FLs = np.unique(FLmap)
    xMinFLmap=np.empty(FLmap.shape)
    xMaxFLmap=np.empty(FLmap.shape)
    yMinFLmap=np.empty(FLmap.shape)
    yMaxFLmap=np.empty(FLmap.shape)
    for FL in FLs:
        inds = np.where(FLmap==FL)
        xMinFLmap[inds] = np.amin(inds[0])
        xMaxFLmap[inds] = np.amax(inds[0])
        yMinFLmap[inds] = np.amin(inds[1])
        yMaxFLmap[inds] = np.amax(inds[1])
    return xMinFLmap, xMaxFLmap, yMinFLmap, yMaxFLmap
        
def isInCorrectFL(resIDs, x, y, instrument, slack=0, flip=False):
    if instrument.lower()=='mec':
        nFL = N_FL_MEC
        flWidth = MEC_FL_WIDTH
        flCoords = x
    elif instrument.lower()=='darkness':
        nFL = N_FL_DARKNESS
        flWidth = DARKNESS_FL_WIDTH
        flCoords = y

    correctFL = getFLFromID(resIDs)
    flFromCoordsP = getFLFromCoords(x+slack, y+slack, instrument, flip)
    flFromCoordsN = getFLFromCoords(x-slack, y-slack, instrument, flip)
    
    return (flFromCoordsP == correctFL)|(flFromCoordsN == correctFL)

def getFLFromID(resIDs):
    correctFL = resIDs/10000 
    correctFL = correctFL.astype(np.int)
    return correctFL

def getFLFromCoords(x, y, instrument, flip=False):
    if instrument.lower()=='mec':
        nFL = N_FL_MEC
        flWidth = MEC_FL_WIDTH
        flCoords = x
    elif instrument.lower()=='darkness':
        nFL = N_FL_DARKNESS
        flWidth = DARKNESS_FL_WIDTH
        flCoords = y

    flFromCoords = np.floor(flCoords/flWidth)

    if flip:
        flFromCoords = nFL - flFromCoords
    else:
        flFromCoords += 1

    return flFromCoords

def getDesignFreqMap(designFreqFL, FLmap):
    FLs = np.unique(FLmap)
    designFreqMap = np.empty(FLmap.shape)
    for FL in FLmap:
        designFreqMap[np.where(FLmap==FL)] = designFreqFL
    return designFreqMap

def addBeammapReadoutFlag(initialBeammapFn, outputBeammapFn, templarCfg):
    config = ConfigParser.ConfigParser()
    config.read(templarCfg)
    goodResIDs=np.asarray([])
    for r in config.sections():
        try:
            freqFN = config.get(r,'freqfile')
            print freqFN
            resIDs, _, _ = np.loadtxt(freqFN,unpack=True)
            goodResIDs = np.unique(np.concatenate((goodResIDs,resIDs)))
            #pdb.set_trace()
        except: pass
    allResIDs, flags, x, y = np.loadtxt(initialBeammapFn, unpack=True)
    badPixels = np.where(np.logical_not(np.in1d(allResIDs, goodResIDs)))
    flags[badPixels]=beamMapFlags['noDacTone']
    
    data=np.asarray([allResIDs, flags, x, y]).T
    np.savetxt(outputBeammapFn, data, fmt='%7d %3d %5d %5d')

def restrictRoughBeammapToFLs(roughBeammapFN, fl):
    data = np.loadtxt(roughBeammapFN)
    data[np.where(np.trunc(data[:,0]/10000).astype(np.int)!=fl),1]=beamMapFlags['noDacTone']
    outputFN = roughBeammapFN.split('.')[0]+'_FL'+str(int(fl))+'.bmap'
    np.savetxt(outputFN, data, fmt='%7d %3d %7f %7f')

def convertBeammapToNewFlagFormat(initialBeammapFn, outputBeammapFn, templarCfg):
    allResIDs, flags, x, y = np.loadtxt(initialBeammapFn, unpack=True)
    nonZeroFlagInds = np.where(flags!=0)[0]
    flags[nonZeroFlagInds] += 1 #increment existing flags by 1
    data=np.asarray([allResIDs, flags, x, y]).T
    np.savetxt(outputBeammapFn, data, fmt='%7d %3d %5d %5d')
    addBeammapReadoutFlag(outputBeammapFn, outputBeammapFn, templarCfg)

#@jit
def getFreqMap(initialBeammap, templarCfg):
    resIDs, _, x, y=np.loadtxt(initialBeammap,unpack=True)
    y=y.astype(np.int)
    x=x.astype(np.int)
    nCols = np.amax(x)+1
    nRows = np.amax(y)+1
    freqMap = np.empty((nRows,nCols))
    freqMap[:]=np.nan
    config = ConfigParser.ConfigParser()
    config.read(templarCfg)
    for r in config.sections():
        freqFN = config.get(r,'freqfile')
        if os.path.isfile(freqFN):
            print freqFN
            freqResIDs, freqs, _ = np.loadtxt(freqFN,unpack=True)
            for i,resID in enumerate(freqResIDs):
                ind=np.where(resIDs==resID)[0][0]
                freqMap[y[ind],x[ind]]=freqs[i]
    return freqMap

def getFLMap(initialBeammap):
    resIDMap = getBeammapResIDImage(initialBeammap)
    resIDMap/=10000
    return np.trunc(resIDMap).astype(np.int)

#@jit
def shapeBeammapIntoImages(initialBeammap, roughBeammap):
    resIDs, flag, x, y=np.loadtxt(initialBeammap,unpack=True)
    nCols = int(np.amax(x)+1)
    nRows = int(np.amax(y)+1)
    resIDimage = np.empty((nRows,nCols))
    flagImage = np.empty((nRows,nCols))
    xImage = np.empty((nRows,nCols))
    xImage[:]=np.nan
    yImage = np.empty((nRows,nCols))
    yImage[:]=np.nan
    y=y.astype(np.int)
    x=x.astype(np.int)
        
    try:
        roughResIDs, roughFlags, roughX, roughY=np.loadtxt(roughBeammap,unpack=True)
        for i,resID in enumerate(roughResIDs):
            ind=np.where(resIDs==resID)[0][0]
            resIDimage[y[ind],x[ind]]=int(resID)
            flagImage[y[ind],x[ind]]=int(roughFlags[i])
            xImage[y[ind],x[ind]]=roughX[i]
            yImage[y[ind],x[ind]]=roughY[i]
    except IOError:
        for i in range(len(resIDs)):
            resIDimage[y[i],x[i]]=int(resIDs[i])
            flagImage[y[i],x[i]]=int(flag[i])
    return resIDimage, flagImage, xImage, yImage

@jit
def getBeammapFlagImage(beammap, roughBeammap=None):
    resIDs, flag, x, y=np.loadtxt(beammap,unpack=True)
    nCols = np.amax(x)+1
    nRows = np.amax(y)+1
    image = np.empty((nRows,nCols))
    
    if roughBeammap is not None:
        roughResIDs, roughFlags, _, _=np.loadtxt(roughBeammap,unpack=True)
        for i,resID in enumerate(roughResIDs):
            ind=np.where(resIDs==resID)[0][0]
            #print ind
            image[y[ind],x[ind]]=int(roughFlags[i])
    else:
        for i in range(len(resIDs)):
            image[y[i],x[i]]=int(flag[i])
    return image

@jit
def getBeammapResIDImage(initialBeammap):
    resIDs, flag, x, y=np.loadtxt(initialBeammap,unpack=True)
    nCols = np.amax(x)+1
    nRows = np.amax(y)+1
    image = np.empty((nRows,nCols))
    for i in range(len(resIDs)):
        image[y[i],x[i]]=int(resIDs[i])
    return image


def snapToPeak(data, guess_arg, width=5):
    if not np.isfinite(guess_arg) or guess_arg<0 or guess_arg>=len(data): return np.nan
    guess_arg=int(guess_arg)
    startInd = max(guess_arg-width,0)
    endInd = min(guess_arg+width+1, len(data)-1)
    return np.argmax(data[startInd:endInd])+startInd

@jit
def gaussian(x, center, scale, width, offset):
    return scale*np.exp(-(x - center)**2/width**2) + offset

def fitPeak(timestream, initialGuess=np.nan, fitWindow=20):
    """
    This function fits a gaussian to a timestream with an initial Guess for location

    INPUT:
        timestream - 
        initialGuess - guess for location of peak
        fitWindow - only consider data around this window
    OUTPUT:
        fitParams - center, scale, width of fitted gaussian
    """ 
    providedInitialGuess = initialGuess #initial guess provided by user

    if not(np.isfinite(initialGuess) and initialGuess >=0 and initialGuess<len(timestream)):
        initialGuess=np.argmax(timestream)

    if fitWindow is not None:
        minT = int(max(0, initialGuess-fitWindow))
        maxT = int(min(len(timestream), initialGuess+fitWindow))
    else:
        minT = 0
        maxT = len(timestream)
    initialGuess-=minT #need this if we're only fitting to a small window
    timestream = timestream[minT:maxT]
    
    try:
        width=2.
        offset=np.median(timestream)
        scale = np.amax(timestream) - offset
        #print [initialGuess+minT, scale, width,offset]
        fitParams, _ = spo.curve_fit(gaussian, xdata=range(len(timestream)), ydata=timestream, p0=[initialGuess, scale, width,offset], sigma=np.sqrt(timestream))
        if fitParams[0]<0 or fitParams[0]>len(timestream):
            raise RuntimeError('Fit center is outside the available range')
        fitParams[0]+=minT
        return fitParams
    except RuntimeError:
        return [providedInitialGuess, np.nan, np.nan, np.nan]

def getPeakCoM(timestream, initialGuess=np.nan, fitWindow=15):
    """
    This function determines the center of mass moment of the peak around fitWindow
    """

    if not(np.isfinite(initialGuess) and initialGuess >=0 and initialGuess<len(timestream)):
        initialGuess=np.argmax(timestream)

    if fitWindow is not None:
        minT = int(max(0, initialGuess-fitWindow))
        maxT = int(min(len(timestream), initialGuess+fitWindow))
    else:
        minT = 0
        maxT = len(timestream)

    timestream -= np.median(timestream) #baseline subtract
    timestream = np.correlate(timestream, np.ones(11), mode='same')
    timestreamLabels = np.arange(len(timestream))

    timestream = timestream[minT:maxT]
    #timestream -= np.median(timestream) #baseline subtract
    timestreamLabels = timestreamLabels[minT:maxT]

    return np.sum(timestreamLabels*timestream)/np.sum(timestream)

def loadImgFiles(fnList, nRows, nCols):
    imageList = []
    for fn in fnList:
        image = np.fromfile(open(fn, mode='rb'),dtype=np.uint16)
        image = np.transpose(np.reshape(image, (nCols, nRows)))
        imageList.append(image)
    return np.asarray(imageList)

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


def isResonatorOnCorrectFeedline(resID, xcoordinate, ycoordinate, instrument='', flip=False):
    correctFeedline = np.floor(resID / 10000)
    flFromCoord = getFLFromCoords(xcoordinate, ycoordinate,instrument,flip)
    if correctFeedline == flFromCoord:
        return True
    else:
        return False

def getFLFromCoords(x, y, instrument='', flip=False):
    if instrument.lower() == 'mec':
        numFL = 10
        flWidth = 14
        flCoord = x
    elif instrument.lower() == 'darkness':
        numFL = 5
        flWidth = 25
        flCoord = y

    flFromCoords = np.floor(flCoord/flWidth)
    if flip:
        flFromCoords = numFL - flFromCoords
    else:
        flFromCoords = flFromCoords + 1
    return flFromCoords


def placeResonatorOnFeedline(xCoord,yCoord,instrument):
    if instrument.lower() == 'mec':
        x = int(xCoord % MEC_FL_WIDTH)
        y = int(yCoord % MEC_FL_LENGTH)
    elif instrument.lower() == 'darkness':
        x = int(xCoord % DARKNESS_FL_LENGTH)
        y = int(yCoord % DARKNESS_FL_WIDTH)

    return x,y
