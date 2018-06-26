from matplotlib import rcParams, rc
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from baselineIIR import IirFilter
import makeNoiseSpectrum as mNS
import makeArtificialData as mAD
import warnings


def makeTemplate(rawData, numOffsCorrIters=1 , decayTime=50, nSigmaTrig=4., isVerbose=False, isPlot=False,sigPass=1,defaultFilter=[]):
    '''
    Make a matched filter template using a raw phase timestream
    INPUTS:
    rawData - noisy phase timestream with photon pulses
    numOffsCorrIters - number of pulse offset corrections to perform
    decayTime - approximate decay time of pulses (units: ticks)
    nSigmaTrig - threshold to detect pulse in terms of standard deviation of data
    isVerbose - print information about the template fitting process
    isPlot - plot information about Chi2 cut
    sigPass - std of data left after Chi2 cut
    defaultFilter - default filter for testing if noise data has pulse contamination

    OUTPUTS:
    finalTemplate - template of pulse shape
    time - use as x-axis when plotting template
    noiseSpectDict - dictionary containing noise spectrum and corresponding frequencies
    templateList - list of template itterations by correcting offsets
    peakIndices - list of peak indicies from rawData used for template
    '''

    #hipass filter data to remove any baseline
    data = hpFilter(rawData)

    #make filtered data set with default filter
    filteredData=np.convolve(data,defaultFilter,mode='same')
    filteredData=data

    #trigger on pulses in data 
    peakDict = sigmaTrigger(filteredData,nSigmaTrig=nSigmaTrig, decayTime=decayTime,isVerbose=isVerbose)
    
    #if too many triggers raise an error
    if len(peakDict['peakIndices'])>(len(data)/400):
        raise ValueError('makeTemplate.py: triggered on too many pulses')
    
    #remove pulses with additional triggers in the pulse window
    peakIndices = cutPulsePileup(peakDict['peakMaxIndices'], decayTime=decayTime, isVerbose=isVerbose)
    
    #back to non filtered data
    #peakIndices = findNearestMax(data,peakIndices)

    #remove pulses with a large chi squared value
    #peakIndices = cutChiSquared(data,peakIndices,sigPass=sigPass, decayTime=decayTime, isVerbose=isVerbose, isPlot=isPlot)
        
    #Create rough template
    roughTemplate, time = averagePulses(data, peakIndices, decayTime=decayTime)
    
    #create noise spectrum from pre-pulse data for filter
    noiseSpectDict = mNS.makeWienerNoiseSpectrum(rawData,peakIndices,numBefore=60,numAfter=0,template=defaultFilter,isVerbose=isVerbose)

    #Correct for errors in peak offsets due to noise
    templateList = [roughTemplate]
    for i in range(numOffsCorrIters):
        peakIndices = correctPeakOffs(data, peakIndices, noiseSpectDict, roughTemplate, 'wiener')
        #calculate a new corrected template
        roughTemplate, time = averagePulses(data, peakIndices,isoffset=True, decayTime=decayTime) 
        templateList.append(roughTemplate)
    
    #fit = lambda x, tau: -np.exp(-x/tau)
    fit = lambda x ,tau1, tau2: -(np.exp(-x/tau1)+np.exp(-x/tau2))/2.    
    bounds=(1,[100,100])
    warnings.filterwarnings("ignore")
    taufit=opt.curve_fit(fit , np.arange(0,60), roughTemplate , [20.,20.] ) 
    warnings.filterwarnings("default")
 
    finalTemplate = fit(np.arange(0,60),taufit[0][0],taufit[0][1])

    return finalTemplate, time, noiseSpectDict, templateList, peakIndices

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
    trigMask = np.logical_or( data > (med + np.std(data)*nSigmaTrig) , data < (med - np.std(data)*nSigmaTrig) )
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
            peakMaxIndices = np.append(peakMaxIndices, np.argmax(np.abs(peakData))+int(p))
                            
            i+=1
            if i < len(peakIndices):
                p = peakIndices[i]
            else:
                p = peakIndices[-1]
         
            
    else:
        raise ValueError('sigmaTrigger: No triggers found in dataset')
    
    if isVerbose:
        print 'triggered on', len(peakIndices), 'pulses'    
    
    peakDict={'peakIndices':peakIndices, 'peakMaxIndices':peakMaxIndices.astype(int)}
    return peakDict

def findNearestMax(data,peakIndices):
    newIndices=[]
    for iPeak, peakIndex in enumerate(peakIndices):
        peakIndex=int(peakIndex)
        if peakIndex>=15 and peakIndex<=len(data)-15:
            arg=np.argmax(np.abs(data[peakIndex:peakIndex+15]))
            arg=peakIndex+arg
        elif peakIndex<15:
            arg=np.argmax(np.abs(data[peakIndex:peakIndex+15]))
        else:
            arg=np.argmax(np.abs(data[peakIndex-15:]))
            arg=peakIndex+arg
        newIndices=np.append(newIndices,arg)

    return newIndices

def cutPulsePileup(peakIndices, nPointsBefore= 0, nPointsAfter = 60 , decayTime=50, isVerbose=False):
    '''
    Removes any pulses that have another pulse within 'window' (in ticks) This is
    to ensure that template data is not contaminated by extraneous pulses.
    
    INPUTS:
    peakIndices - list of pulse positions
    nPointsBefore - number of points before peakIndex included in template
    nPointsAfter - number of points after peakIndex included in template
    decayTime - expected pulse decay time (units: ticks)    
    isVerbose - print information about the template fitting process    

    OUTPUS:
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
        print len(peakIndices)-len(newPeakIndices), 'indices cut due to pileup'
    
    return newPeakIndices

def cutChiSquared(data,peakIndices,sigPass=1, decayTime=50, nPointsBefore= 0, nPointsAfter=60, isVerbose=False, isPlot=False):   
    '''
    Removes a fraction of pulses with the worst Chi Squared fit to 
    the exponential tail. This should remove any triggers that don't look like 
    pulses. Currently not optimized to fit in the frequency domain.
    
    INPUTS:
    data - raw phase timestream
    peakIndices - list of pulse positions
    sigPass - fraction of pulses selected to pass the cut
    decayTime - expected pulse decay time (units: ticks)
    nPointsBefore - number of points before peakIndex included in template
    nPointsAfter - number of points after peakIndex included in template
    isVerbose - print information about the template fitting process    
    
    OUTPUTS:
    newPeakIndices - list of pulse positions, with unwanted pulses deleted
    '''
    
    chiSquared=np.zeros(len(peakIndices))
    for iPeak, peakIndex in enumerate(peakIndices):
        time=np.arange(0,len(data[peakIndex+int(decayTime/2):peakIndex+nPointsAfter]))
        currentData=data[peakIndex+int(decayTime/2):peakIndex+nPointsAfter]
        ampGuess=currentData[np.argmax(np.abs(currentData))]
        try:
            warnings.filterwarnings("ignore")
            expCoef, _ = opt.curve_fit(lambda t, a, tau: a*np.exp(-t/tau) , time, currentData , [ampGuess, decayTime] )
            warnings.filterwarnings("default")
            ampFit=expCoef[0]
            decayFit=expCoef[1]
            chiSquared[iPeak]=np.sum((currentData-ampFit*np.exp(-time/decayFit))**2)
        except RuntimeError:
            chiSquared[iPeak]=np.nan
     
    chi2Median=np.median(chiSquared)
    chi2Sig=np.std(chiSquared[np.invert(np.isnan(chiSquared))])
     
    newPeakIndices=np.array([])
    newChiSquared=np.array([])
    for iPeak, peakIndex in enumerate(peakIndices):
        if np.abs(chiSquared[iPeak]-chi2Median)<sigPass*chi2Sig:
            newPeakIndices=np.append(newPeakIndices,int(peakIndex))
            newChiSquared=np.append(newChiSquared,chiSquared[iPeak])
            
    
    if isVerbose:
        print len(peakIndices)-len(newPeakIndices), 'indices cut with worst Chi Squared value'
        
    if len(newPeakIndices)==0:
        raise ValueError('cutChiSquared: no pulses passed the Chi Squared cut') 
        
    if isPlot:
        worstDataIndex=np.argmax(np.abs(newChiSquared-chi2Median))
        fig = plt.figure()
        plt.plot(data[newPeakIndices[worstDataIndex]-nPointsBefore:newPeakIndices[worstDataIndex]+nPointsAfter])
        plt.title('Worst pulse not cut by $\chi^2$ cut')
        plt.show()    
        
        worstDataIndex=np.argmax(np.abs(chiSquared-chi2Median))
        fig=plt.figure()
        plt.plot(data[peakIndices[worstDataIndex]-nPointsBefore:peakIndices[worstDataIndex]+nPointsAfter])
        plt.title('Worst pulse cut by $\chi^2$ cut')
        plt.show()
        
        #fig =plt.figure()
        #plt.hist(chiSquared)
        #ax = plt.gca()
        #ax.set_xlabel('$\chi^2$')
        #splt.show()
                
    return newPeakIndices
    
    
def averagePulses(data, peakIndices, isoffset=False, nPointsBefore=0, nPointsAfter=60, decayTime=30, sampleRate=1e6):
    '''
    Average together pulse data to make a template
    
    INPUTS:
    data - raw phase timestream
    peakIndices - list of pulse positions
    isoffset - true if peakIndices are the locations of peak maxima
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
        if peakIndex >= nPointsBefore and peakIndex < len(data)-nPointsAfter:
            peakRecord = data[int(peakIndex-nPointsBefore):int(peakIndex+nPointsAfter)]
            peakData = data[int(peakIndex-decayTime):int(peakIndex+decayTime)]
            
            if isoffset:
                peakRecord/=np.abs(data[int(peakIndex)])
            else:
                peakHeight = np.max(np.abs(peakData))
                peakRecord /= peakHeight
            template += peakRecord
            numPeaks += 1
    if numPeaks==0:
        raise ValueError('averagePulses: No valid peaks found')
    
    template=template/np.max(np.abs(template))
    time = np.arange(0,nPointsBefore+nPointsAfter)/sampleRate
    return template, time
    
def correctPeakOffs(data, peakIndices, noiseSpectDict, template, filterType, offsets=np.arange(-20,21), nPointsBefore=0, nPointsAfter=60):
    '''
    Correct the list of peak indices to improve the alignment of photon pulses.  

    INPUTS:
    data - raw phase timestream
    peakIndices - list of photon pulse indices
    noiseSpectDict - dictionary containing noise spectrum and corresponding frequencies
    template - template of pulse to use for filter
    filterType - string specifying the type of filter to use
    offsets - list of peak index offsets to check
    nPointsBefore - number of points before peakIndex to include in pulse
    nPointsAfter - number of points after peakIndex to include in pulse

    OUTPUTS:
    newPeakIndices - list of corrected peak indices
    '''
    
    if filterType=='wiener':
        makeFilter = makeWienerFilter
    elif filterType=='matched':
        #does not work yet 08/18/2016
        makeFilter = makeMatchedFilter
    else:
        raise ValueError('makeFilterSet: Filter not defined')
    
    nOffsets = len(offsets)
    nPointsTotal = nPointsBefore + nPointsAfter
    filterSet = np.zeros((nOffsets,nPointsTotal),dtype=np.complex64)
    newPeakIndices = []
    
    #Create a set of filters from different template offsets
    for i,offset in enumerate(offsets):
        templateOffs = np.roll(template, offset)
        filterSet[i] = makeFilter(noiseSpectDict, templateOffs)

    #find which peak index offset is the best for each pulse:
    #   apply each offset to the pulse, then determine which offset 
    #   maximizes the pulse amplitude after application of the filter
    for iPeak,peakIndex in enumerate(peakIndices):
        if peakIndex > nPointsBefore-np.min(offsets) and peakIndex < len(data)-(nPointsAfter+np.max(offsets)):
            peakRecord = data[int(peakIndex-nPointsBefore):int(peakIndex+nPointsAfter)]
            peakRecord = peakRecord / np.max(np.abs(peakRecord))
            #check which time shifted filter results in the biggest signal
            peakRecordFft = np.fft.fft(peakRecord)/nPointsTotal
            convSums = np.abs(np.sum(filterSet*peakRecordFft,axis=1))
            bestOffsetIndex = np.argmax(convSums)
            bestConvSum = convSums[bestOffsetIndex]
            bestOffset = offsets[bestOffsetIndex]
            newPeakIndices=np.append(newPeakIndices, int(peakIndex+bestOffset))

    return newPeakIndices
    
def makeWienerFilter(noiseSpectDict, template):
    '''
    Calculate acausal Wiener Filter coefficients in the frequency domain
    
    INPUTS:
    noiseSpectDict - Dictionary containing noise spectrum and list of corresponding frequencies
    template - template of pulse shape
    
    OUTPUTS:
    wienerFilter - list of Wiener Filter coefficients
    '''
    template /= np.max(np.abs(template)) #should be redundant
    noiseSpectrum = noiseSpectDict['noiseSpectrum']
    templateFft = np.fft.fft(template)/len(template)
    wienerFilter = np.conj(templateFft)/noiseSpectrum
    filterNorm = np.sum(np.abs(templateFft)**2/noiseSpectrum)
    wienerFilter /= filterNorm
    return wienerFilter

def makeMatchedFilter(noiseSpectDict, template):
    '''
    Calculate Matched Filter coefficients
    Does not work yet 08/18/2016
    
    INPUTS:
    noiseSpectDict - Dictionary containing noise spectrum and list of corresponding frequencies
    template - template of pulse shape
    
    OUTPUTS:
    matchedFilter - list of Matched Filter coefficients
    '''
    noiseSpectrum = noiseSpectDict['noiseSpectrum']
    noiseCovInv = mNS.covFromPsd(noiseSpectrum)['covMatrixInv']    
    filterNorm = np.sqrt(np.dot(template, np.dot(noiseCovInv, template))) 
    matchedFilt = np.dot(noiseCovInv, template)/filterNorm
    return matchedFilt
        
def makeFittedTemplate(template,time,riseGuess=3.e-6, fallGuess=55.e-6, peakGuess=100*1e-6):
    '''
    Fit template to double exponential pulse
    INPUTS:
    template - somewhat noisy template to be fitted
    time - time variable for template
    riseGuess - guess for pulse rise time in same units as 'time' variable
    fallGuess - guess for pulse fall time in same units as 'time' variable
    peakGuess - guess for what time in your template the fitted peak will be 
                in same units as 'time' variable
  
    OUTPUTS:
    fittedTemplate - fitted template with double exponential pulse
    startFit - fitted value of peakGuess
    riseFit - fitted value of riseGuess
    fallFit - fitted value of fallGuess
    '''
    
    if template[np.argmax(np.abs(template[1:len(template)-1]))]>0:
        pos_neg=1
    else:
        pos_neg=-1
        
    startGuess=peakGuess+riseGuess*np.log(riseGuess/(riseGuess+fallGuess))
    coef, coefCov =opt.curve_fit(pulseFitFun , time,pos_neg*template,[startGuess,riseGuess,fallGuess, 1., 0.])
    
    startFit=coef[0]
    riseFit=coef[1]
    fallFit=coef[2]
    aFit=coef[3]
    bFit=coef[4]
    fittedTemplate=pos_neg*pulseFitFun(time,startFit,riseFit,fallFit,aFit,bFit)
    
    #renormalize template to 1 while keeping any small baseline offset imposed by hi-pass filter
    fittedTemplate=(fittedTemplate-bFit)/(np.max(np.abs(fittedTemplate))-bFit)*(1+np.abs(bFit))+bFit
    
    return fittedTemplate, startFit, riseFit, fallFit 

def pulseFitFun(x,t0,t1,t2,a,b):
    '''
    double exponential pulse function normalized to one
    INPUTS:
    x - time array
    t0 - pulse start time
    t1 - pulse rise time
    t2 - pulse fall time
    a - pulse amplitude
    b - baseline offset
    
    OUTPUTS:
    y - double exponential pulse array
    '''
    
    x=np.array(x)
    t0=float(t0)
    t1=float(t1)
    t2=float(t2)
    
    heaviside=np.zeros(len(x))
    heaviside[x>t0]=1;

    if t1<0 or t2<0:
        norm=1
    else: 
        norm=t2/(t1+t2)*(t1/(t1+t2))**(t1/t2)
    
    y = a*(1-b/a)*(1-np.exp(-(x-t0)/t1))*np.exp(-(x-t0)/t2)/norm*heaviside + b*np.ones(len(x))    
    return y
