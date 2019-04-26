import numpy as np

import triggerPhotons as tP


def makeNoiseSpectrum(data, peakIndices=(), window=800, noiseOffsetFromPeak=200, sampleRate=1e6, filt=(),isVerbose=False,baselineSubtract=True):
    '''
    makes one sided noise power spectrum in units of V^2/Hz by averaging together Fourier transforms of noise
    traces. The code screens out any potential pulse contamination with the provided peak indices and filter

    INPUTS:
    data - raw data to calculate noise from
    peakIndices - list of known peak indices. Will choose randomly if not specified
    window - data size to take individual transforms of for averaging
    noiseOffsetFromPeak - takes window this many points to the left of peak 
    sampleRate - sample rate of data
    filt - filter for detecting unspecified pulses in data
    isVerbose - print extra info to terminal
    baselineSubtract - subtract baseline

    OUTPUTS:
    dictionary containing frequencies, noise spectrum and indicies used to make the noise spectrum
    '''
    peakIndices=np.array(peakIndices).astype(int)
    
    #If no peaks, choose random indices to make spectrum 
    if len(peakIndices)==0:
        peakIndices=np.array([0])
        rate = len(data)/float(window)/1000.
        while peakIndices[-1]<(len(data)-1):
            prob=np.random.rand()
            currentIndex=peakIndices[-1]
            peakIndices=np.append(peakIndices,currentIndex+np.ceil(-np.log(prob)/rate*sampleRate).astype(int))
        peakIndices=peakIndices[:-2]      
    if len(peakIndices)==0:
        raise ValueError('makeNoiseSpectrum: input data set is too short for the number of FFT points specified')
    #Baseline subtract noise data
    if(baselineSubtract):
        noiseStream = np.array([])
        for iPeak,peakIndex in enumerate(peakIndices):
            if peakIndex > window+noiseOffsetFromPeak and peakIndex < len(data)+noiseOffsetFromPeak:
                noiseStream = np.append(noiseStream, data[peakIndex-window-noiseOffsetFromPeak:peakIndex-noiseOffsetFromPeak])
        data = data - np.mean(noiseStream)
    
    #Calculate noise spectra for the defined area before each pulse
    if len(peakIndices)>2000:
        peakIndices = peakIndices[:2000]
        noiseSpectra = np.zeros((len(peakIndices), len(np.fft.rfftfreq(window)) ))
    else:
        noiseSpectra = np.zeros((len(peakIndices), len(np.fft.rfftfreq(window)) ))
    rejectInd=np.array([])
    goodInd=np.array([])
    counter=0
    for iPeak,peakIndex in enumerate(peakIndices):
        if peakIndex > window+noiseOffsetFromPeak and peakIndex < len(data)+noiseOffsetFromPeak:
            noiseData = data[peakIndex-window-noiseOffsetFromPeak:peakIndex-noiseOffsetFromPeak]
            noiseSpectra[counter] =4*window/sampleRate*np.abs(np.fft.rfft(data[peakIndex-window-noiseOffsetFromPeak:peakIndex-noiseOffsetFromPeak]))**2
            if len(filt)!=0:
                filteredData=np.convolve(noiseData,filt,mode='same')
                peakDict=tP.detectPulses(filteredData, nSigmaThreshold = 2., negDerivLenience = 1, bNegativePulses=True)
                if len(peakDict['peakIndices'])!=0:
                    rejectInd=np.append(rejectInd,int(counter-1)) 
                else:
                    goodInd=np.append(goodInd,int(peakIndex))
                    counter += 1
        if counter==500:
            break   
    noiseSpectra=noiseSpectra[0:counter]
    #Remove indicies with pulses by convolving with a filt if provided
    if len(filt)!=0: 
        noiseSpectra = np.delete(noiseSpectra, rejectInd.astype(int), axis=0) 
    noiseFreqs = np.fft.rfftfreq(window,1./sampleRate)
    if len(np.shape(noiseSpectra))==0:
        raise ValueError('makeWienerNoiseSpectrum: not enough spectra to average')
    if np.shape(noiseSpectra)[0]<5:
        raise ValueError('makeWienerNoiseSpectrum: not enough spectra to average') 
           
    noiseSpectrum = np.median(noiseSpectra,axis=0)
    noiseSpectrum[0] = noiseSpectrum[1]
    if not np.all(noiseSpectrum>0):
        raise ValueError('makeWienerNoiseSpectrum: not all noise data >0')
    if isVerbose:
        print len(noiseSpectra[:,0]),'traces used to make noise spectrum', len(rejectInd), 'cut for pulse contamination'

    return {'spectrum':noiseSpectrum, 'freqs':noiseFreqs, 'indices':goodInd}
    
def covFromData(data,size=800,nTrials=None):
    '''
    make a covariance matrix from data
    '''
    nSamples = len(data)
    if nTrials is None:
        nTrials = nSamples//size
    data = data[0:nTrials*size]
    data = data.reshape((nTrials,size))
    data = data.T
    
    covMatrix = np.cov(data)
    covMatrixInv = np.linalg.inv(covMatrix)
    return {'covMatrix':covMatrix,'covMatrixInv':covMatrixInv}

def covFromPsd(powerSpectrum,size=None):
    '''
    make a covariance matrix from a power spectral density
    '''
    autocovariance = np.real(np.fft.irfft(powerSpectrum))
    if size is None:
        size = len(autocovariance)
    sampledAutocovariance = autocovariance[0:size]

    shiftingRow = np.concatenate((sampledAutocovariance[:0:-1],sampledAutocovariance))
    covMatrix = []

    for iRow in range(size):
        covMatrix.append(shiftingRow[size-iRow-1:size-iRow-1+size])

    covMatrix = np.array(covMatrix)

    covMatrixInv = np.linalg.inv(covMatrix)
    return {'covMatrix':covMatrix,'covMatrixInv':covMatrixInv,'autocovariance':sampledAutocovariance}
