from __future__ import print_function
import numpy as np
import scipy.optimize as opt
from baselineIIR import IirFilter
import makeNoiseSpectrum as mNS
import warnings
from phase_wrap import fix_phase_wrap

def makeTemplate(rawData, numOffsCorrIters=1 , decayTime=50, nSigmaTrig=4.,
                 isVerbose=False, defaultFilter=(), fix_wrap=False):
    '''
    Make a matched filter template using a raw phase timestream
    INPUTS:
    rawData - noisy phase timestream with photon pulses
    numOffsCorrIters - number of pulse offset corrections to perform
    decayTime - approximate decay time of pulses (units: ticks)
    nSigmaTrig - threshold to detect pulse in terms of standard deviation of data
    isVerbose - print information about the template fitting process
    defaultFilter - default filter for testing if noise data has pulse contamination

    OUTPUTS:
    finalTemplate - template of pulse shape
    time - use as x-axis when plotting template
    noiseDict - dictionary containing noise spectrum and corresponding frequencies
    templateList - list of template iterations by correcting offsets
    peakIndices - list of peak indices from rawData used for template
    '''
    # correct for phase wrapping (off by default until it works properly)
    if fix_wrap:
        rawData = fix_phase_wrap(rawData)

    # high pass filter data to remove any baseline
    data = hpFilter(rawData)

    # make filtered data set with default filter
    filteredData=np.convolve(data,defaultFilter,mode='same')

    # trigger on pulses in data
    peakDict = sigmaTrigger(filteredData,nSigmaTrig=nSigmaTrig, decayTime=decayTime,isVerbose=isVerbose)
    
    #if too many triggers raise an error
    if len(peakDict['peakIndices'])>(len(data)/400):
        raise ValueError('makeTemplate.py: triggered on too many pulses')
    if len(peakDict['peakIndices'])>1000:
        peakDict['peakIndices']=peakDict['peakIndices'][:1000]
        peakDict['peakMaxIndices']=peakDict['peakMaxIndices'][:1000]
    
    # remove pulses with additional triggers in the pulse window
    peakIndices = cutPulsePileup(peakDict['peakMaxIndices'], decayTime=decayTime, isVerbose=isVerbose)

    # remove pulses that may be phase wrapping
    peakIndices = cutPhaseWrap(peakIndices, data, isVerbose=isVerbose)
    
    # back to non filtered data
    peakIndices = findNearestMax(data,peakIndices)
        
    # Create rough template
    roughTemplate, time = averagePulses(data, peakIndices, decayTime=decayTime)
    
    # create noise spectrum from pre-pulse data for filter
    noiseDict = mNS.makeNoiseSpectrum(rawData,peakIndices,window=200,filt=defaultFilter,isVerbose=isVerbose)

    # Correct for errors in peak offsets due to noise
    templateList = [roughTemplate]
    for i in range(numOffsCorrIters):
        peakIndices = correctPeakOffs(data, peakIndices, noiseDict, roughTemplate)
        #calculate a new corrected template
        roughTemplate, time = averagePulses(data, peakIndices, decayTime=decayTime) 
        templateList.append(roughTemplate)
    
    # fit template to function
    bnds=([.1, 2, 5],[4, 60, 60])
    warnings.filterwarnings("ignore")
    taufit=opt.curve_fit(pulseFitFun , np.arange(0,200), roughTemplate , [2.,10.,20.], bounds=bnds)  
    warnings.filterwarnings("default")
 
    finalTemplate = pulseFitFun(np.arange(0,200),taufit[0][0],taufit[0][1],taufit[0][2])

    return finalTemplate, time, noiseDict, templateList, peakIndices

def hpFilter(rawData, criticalFreq=20, sampleRate = 1e6):
    '''
    High pass filters the raw phase timestream
    INPUTS:
    rawData - data to be filtered
    criticalFreq - cutoff frequency of filter (in Hz)
    sampleRate - sample rate of rawData

    OUTPUTS:
    data - filtered data
    '''
    f=2*np.sin(np.pi*criticalFreq/sampleRate)
    Q=.7
    q=1./Q
    hpSvf = IirFilter(sampleFreqHz=sampleRate,numCoeffs=np.array([1,-2,1]),denomCoeffs=np.array([1+f**2, f*q-2,1-f*q]))
    data = hpSvf.filterData(rawData)
    return data

def sigmaTrigger(data,nSigmaTrig=5.,deadTime=200,decayTime=30,isVerbose=False):
    '''
    Detects pulses in raw phase timestream
    INPUTS:
    data - phase timestream
    nSigmaTrig - threshold to detect pulse in terms of standard deviation of data
    deadTime - minimum amount of time between any two pulses (units: ticks (1 us assuming 1 MHz sample rate))
    decayTime - expected pulse decay time (units: ticks)
    isVerbose - print information about the template fitting process
    
    OUTPUTS:
    peakDict - dictionary of trigger indicies 
               peakIndices: initial trigger index
               peakMaxIndices: index of the max near the initial trigger
    '''
    data = np.array(data)
    med = np.median(data)
    #print 'sdev',np.std(data),'med',med,'max',np.max(data)
    #trigMask = np.logical_or( data > (med + np.std(data)*nSigmaTrig) , data < (med - np.std(data)*nSigmaTrig) )
    trigMask=data < (med - np.std(data)*nSigmaTrig)
    if np.sum(trigMask) > 0:
        peakIndices = np.where(trigMask)[0]
        i = 0
        p = peakIndices[i]
        peakMaxIndices = []
        while p < peakIndices[-1]:
            peakIndices = peakIndices[np.logical_or(peakIndices-p > deadTime , peakIndices-p <= 0)]#apply deadTime
            if p+decayTime<len(data):            
                peakData = data[p:p+decayTime]
            else:
                peakData = data[p:]
            peakMaxIndices = np.append(peakMaxIndices, np.argmin(peakData)+int(p))
                            
            i+=1
            if i < len(peakIndices):
                p = peakIndices[i]
            else:
                p = peakIndices[-1]
         
            
    else:
        raise ValueError('sigmaTrigger: No triggers found in dataset')
    
    if isVerbose:
        print('triggered on', len(peakIndices), 'pulses')
    
    peakDict={'peakIndices':np.array(peakIndices), 'peakMaxIndices':np.array(peakMaxIndices).astype(int)}
    return peakDict

def findNearestMax(data,peakIndices):
    newIndices=[]
    for iPeak, peakIndex in enumerate(peakIndices):
        peakIndex=int(peakIndex)
        if peakIndex>=25 and peakIndex<=len(data)-25:
            arg=np.argmax(np.abs(data[peakIndex-25:peakIndex]))
            arg=peakIndex+arg-25
        elif peakIndex<25:
            arg=np.argmax(np.abs(data[:peakIndex]))
        else:
            arg=np.argmax(np.abs(data[peakIndex-25:peakIndex]))
            arg=peakIndex+arg-25
        newIndices=np.append(newIndices,arg)

    return newIndices

def cutPulsePileup(peakIndices, nPointsBefore= 5, nPointsAfter = 195 , decayTime=50, isVerbose=False):
    '''
    Removes any pulses that have another pulse within 'window' (in ticks) This is
    to ensure that template data is not contaminated by extraneous pulses.
    
    INPUTS:
    peakIndices - list of pulse positions
    nPointsBefore - number of points before peakIndex included in template
    nPointsAfter - number of points after peakIndex included in template
    decayTime - expected pulse decay time (units: ticks)    
    isVerbose - print information about the template fitting process    

    OUTPUTS:
    newPeakIndices - list of pulse positions, with unwanted pulses deleted
    '''
    #set window for pulse rejection
    window=nPointsBefore+nPointsAfter
    if window<10*decayTime:
        window=10*decayTime
        
    peakIndices=np.array(peakIndices)
    newPeakIndices=np.array([])
    #check that no peaks are near current peak and then add to new indices variable
    for iPeak, peakIndex in enumerate(peakIndices):
        if np.min(np.abs(peakIndices[np.arange(len(peakIndices))!=iPeak]-peakIndex))>window:
            newPeakIndices=np.append(newPeakIndices,int(peakIndex))

    if len(newPeakIndices)==0:
        raise ValueError('cutPulsePileup: no pulses passed the pileup cut')       
    
    if isVerbose:
        print(len(peakIndices)-len(newPeakIndices), 'indices cut due to pileup')
    
    return newPeakIndices


def cutPhaseWrap(peakIndices, data, nPointsBefore=5, nPointsAfter=195, isVerbose=False):
    """
    Removes any pulses that have a phase difference of over pi radians in their trace
    INPUTS:
    peakIndices - list of pulse positions
    data - data with the pulses
    nPointsBefore - number of points before peakIndex included in template
    nPointsAfter - number of points after peakIndex included in template
    decayTime - expected pulse decay time (units: ticks)
    isVerbose - print information about the template fitting process

    OUTPUTS:
    newPeakIndices - list of pulse positions, with unwanted pulses deleted
    """
    newPeakIndices = np.array([])
    # loop through indices and remove those that might be wrapping
    for iPeak, peakIndex in enumerate(peakIndices):
        delta = np.diff(data[int(peakIndex - nPointsBefore):int(peakIndex + nPointsAfter)])
        if not (np.abs(delta) > np.pi).any():
            newPeakIndices = np.append(newPeakIndices, int(peakIndex))

    if isVerbose:
        print(len(peakIndices)-len(newPeakIndices), 'indices cut due to phase wrapping')

    return newPeakIndices


def averagePulses(data, peakIndices, nPointsBefore=5, nPointsAfter=195, decayTime=30, sampleRate=1e6):
    '''
    Average together pulse data to make a template
    
    INPUTS:
    data - raw phase timestream
    peakIndices - list of pulse positions
    nPointsBefore - number of points before peakIndex to include in template
    nPointsAfter - number of points after peakIndex to include in template
    decayTime - expected pulse decay time (in ticks (us))
    sampleRate - sample rate of 'data'
    
    OUTPUTS:
    template - caluculated pulse template
    time - time markers indexing data points in 'template'
           (use as x-axis when plotting)
    '''
    numPeaks = 0
    template=np.zeros(nPointsBefore+nPointsAfter)
    for iPeak,peakIndex in enumerate(peakIndices):
        peakIndex=int(peakIndex)
        if peakIndex >= max(nPointsBefore,decayTime) and peakIndex < min(len(data)-nPointsAfter,len(data)-decayTime):
            peakRecord = data[int(peakIndex-nPointsBefore):int(peakIndex+nPointsAfter)]
            peakData = data[int(peakIndex-decayTime):int(peakIndex+decayTime)]

            peakHeight = np.max(np.abs(peakData))
            peakRecord /= peakHeight

            template += peakRecord
            numPeaks += 1
    if numPeaks==0:
        raise ValueError('averagePulses: No valid peaks found')
    
    template=template/np.max(np.abs(template))
    time = np.arange(0,nPointsBefore+nPointsAfter)/sampleRate
    return template, time
    
def correctPeakOffs(data, peakIndices, noiseDict, template, offsets=np.arange(-20,21), nPointsBefore=5, nPointsAfter=195):
    '''
    Correct the list of peak indices to improve the alignment of photon pulses.  

    INPUTS:
    data - raw phase timestream
    peakIndices - list of photon pulse indices
    noiseDict - dictionary containing noise spectrum and corresponding frequencies
    template - template of pulse to use for filter
    offsets - list of peak index offsets to check
    nPointsBefore - number of points before peakIndex to include in pulse
    nPointsAfter - number of points after peakIndex to include in pulse

    OUTPUTS:
    newPeakIndices - list of corrected peak indices
    '''
        
    nOffsets = len(offsets)
    nPointsTotal = nPointsBefore + nPointsAfter
    filterSet = np.zeros((nOffsets,len(np.fft.rfftfreq(nPointsTotal)) ),dtype=np.complex64)
    newPeakIndices = []
    
    # Create a set of filters from different template offsets
    for i,offset in enumerate(offsets):
        templateOffs = np.roll(template, offset)
        filterSet[i] = makeWienerFilter(noiseDict, templateOffs)

    #find which peak index offset is the best for each pulse:
    #   apply each offset to the pulse, then determine which offset 
    #   maximizes the pulse amplitude after application of the filter
    for iPeak,peakIndex in enumerate(peakIndices):
        if peakIndex > nPointsBefore-np.min(offsets) and peakIndex < len(data)-(nPointsAfter+np.max(offsets)):
            peakRecord = data[int(peakIndex-nPointsBefore):int(peakIndex+nPointsAfter)]
            peakRecord = peakRecord /peakRecord[nPointsBefore]
            #check which time shifted filter results in the biggest signal
            peakRecordFft = np.fft.rfft(peakRecord)/nPointsTotal
            convSums = np.abs(np.sum(filterSet*peakRecordFft,axis=1))
            bestOffsetIndex = np.argmax(convSums)
            bestConvSum = convSums[bestOffsetIndex]
            bestOffset = offsets[bestOffsetIndex]
            newPeakIndices=np.append(newPeakIndices, int(peakIndex+bestOffset))

    return newPeakIndices
    
def makeWienerFilter(noiseDict, template):
    '''
    Calculate acausal Wiener Filter coefficients in the frequency domain
    
    INPUTS:
    noiseDict - Dictionary containing noise spectrum and list of corresponding frequencies
    template - template of pulse shape
    
    OUTPUTS:
    wienerFilter - list of Wiener Filter coefficients
    '''
    template /= np.max(np.abs(template)) #should be redundant
    noiseSpectrum = noiseDict['spectrum']
    templateFft = np.fft.rfft(template)/len(template)
    wienerFilter = np.conj(templateFft)/noiseSpectrum
    filterNorm = np.sum(np.abs(templateFft)**2/noiseSpectrum)
    wienerFilter /= filterNorm
    return wienerFilter
        
def pulseFitFun(x,t0,t1,t2):
    '''
    double exponential pulse function normalized to one
    INPUTS:
    t0 - pulse rise time
    t1 - pulse fall time 1
    t2 - pulse fall time 2
    
    OUTPUTS:
    y - double exponential pulse array
    '''
    
    x=np.array(x)
    x1=np.arange(0,50,.1)
    t0=float(t0)
    t1=float(t1)
    t2=float(t2)
        
    y = -(1-np.exp(-x1/t0))*(np.exp(-x1/t1)+np.exp(-x1/t2))
    argy=np.argmin(y)
    tmin=x1[argy]
    y = -(1-np.exp(-(x+tmin-5.0)/t0))*(np.exp(-(x+tmin-5.0)/t1)+np.exp(-(x+tmin-5.0)/t2))
    y = -y/np.min(y)
    y[y>0]=0.0
    
    return y
