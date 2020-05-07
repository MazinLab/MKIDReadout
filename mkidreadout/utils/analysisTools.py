import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import scipy.signal as signal
import mkidreadout.configuration.optimalfilters.make_filters as filt

def getPhaseNoiseSpectrum(data, convertToDB=True, fftlen=65536, dt=256./250e6):
    """
    parameters
    ----------
        data - phase in radians
        fftlen - n frequencies
        dt - sampling period

    returns
    -------
        noise data in dBc/Hz

    """

    nFftAvg = int(np.floor(len(data) / fftlen))
    data = np.reshape(data[:nFftAvg * fftlen], (nFftAvg, fftlen))
    noiseData = np.fft.rfft(data)
    noiseFreqs = np.fft.rfftfreq(fftlen, d=dt)
    noiseData = np.abs(noiseData) ** 2  # power spectrum
    noiseData = 2 * dt * np.average(noiseData, axis=0) / fftlen  # normalize and average
    if convertToDB:
        noiseData = 10. * np.log10(noiseData)  # convert to dBc/Hz
    return noiseFreqs, noiseData
    #can also use:
    # noiseFreqs, noiseData = scipy.signal.welch(data, fs=1/dt, return_onesided=True, nperseg=fftlen)
    #has hanning window (boxcar gives almost the same thing, looks a bit smoother

def getTemplateSpectrum(optFiltSol, resNum, fftLen, convertToDB=True, meanSubtract=False, dt=256/250.e6, key='template'):
    assert key=='filter' or key=='template'
    template = optFiltSol.calculators[resNum].result[key]
    if len(template) > fftLen:
        template = template[:fftLen]
    elif len(template) < fftLen:
        if key=='template':
            template = np.pad(template, (0, fftLen-len(template)), 'edge')
        else: 
            template = np.pad(template, (0, fftLen-len(template)), 'constant', constant_values=0)
    if meanSubtract:
        template -= np.mean(template)
    templateFFT = np.fft.rfft(template)
    freqs = np.fft.rfftfreq(fftLen, d=dt)
    templatePSD = np.abs(templateFFT)**2
    if convertToDB:
        templatePSD = 10*np.log10(templatePSD)
    return freqs, templatePSD

def computeOptimalFilter(template, noisePSDFreqs, noisePSD, optFiltLen=50, fftLen=65536, cutoffFreq=100.e3):
    """
    noisePSD is single sided, NOT in dB
    fftLen is length of full (positive and negative freq) FFT
    cutoffFreq - cut off filter at 100 kHz b/c noise rolls off aggressively here
    """
    if len(template) > fftLen:
        template = template[:fftLen]
    elif len(template) < fftLen:
        template = np.pad(template, (0, fftLen-len(template)), 'constant', constant_values=0)
    templateFFT = np.fft.rfft(template)
    optFiltFFT = np.conj(templateFFT)/noisePSD
    cutoffInd = np.argmin(np.abs(noisePSDFreqs - cutoffFreq))
    optFiltFFT[cutoffInd:] = 0
    optFilt = np.fft.irfft(optFiltFFT)
    optFilt = np.roll(optFilt, -1) #put t=0 in the right place
    optFilt[:-optFiltLen] = 0 #make filter only optFiltLen taps
    optFiltFFT = np.fft.rfft(optFilt)
    return optFilt, optFiltFFT

def computePulseVariance(template, noisePSDFreqs, noisePSD, removeNoiseSpurs=False, optFiltLen=50, fftLen=65536):
    """
    noisePSD is single sided, NOT in dB
    fftLen is length of full (positive and negative freq) FFT
    """
    noisePSD[0] = noisePSD[1] #get rid of 0 freq bin
    df = noisePSDFreqs[1] - noisePSDFreqs[0]
    if removeNoiseSpurs:
        noisePSD = removeSpurs(noisePSDFreqs, noisePSD, 100, 300.e3, -90)
    optFilt, optFiltFFT = computeOptimalFilter(template, noisePSDFreqs, noisePSD, optFiltLen, fftLen)
    if len(template) > fftLen:
        template = template[:fftLen]
    elif len(template) < fftLen:
        template = np.pad(template, (0, fftLen-len(template)), 'edge')
    templateFFT = np.fft.rfft(template)
    return 1/np.sum(template*optFilt[::-1]), np.abs(1/np.sum(df*optFiltFFT*templateFFT))
    


def fitPhaseNoiseSpectrum(data, fftlen=65536, dt=256./250e6):
    freqs, spect = getPhaseNoiseSpectrum(data, False, fftlen, dt)

    def flatoverf(freq, a, b):
        return a + b/freq
    
    #popt, pcov = np.polyfit(np.log10(freqs), np.log10(spect), deg=1)
    fitcutoff = np.argmin(np.abs(freqs - 30.e3)) #remove rolloff
    popt, pcov = spo.curve_fit(flatoverf, freqs[1:fitcutoff], spect[1:fitcutoff], sigma=spect[1:fitcutoff])
    return freqs, spect, popt, pcov

def plotFittedSpectrum(data, fftlen=65536, dt=256./250e6):
    f, s, po, _ = fitPhaseNoiseSpectrum(data, fftlen, dt)
    plt.semilogx(f, 10*np.log10(s)); plt.semilogx(f,10*np.log10(po[0] + po[1]/f)); plt.show()

def batchFitPhaseNoiseDir(dir, removePhaseWraps=True):
    fileList = os.listdir(dir)
    freqs, spect = getPhaseNoiseSpectrum(np.load(os.path.join(dir, fileList[0]))['arr_0'])
    fitList = np.zeros((len(fileList), 2))
    spectList = np.zeros((len(fileList), len(spect)))
    fitCovList = np.zeros((len(fileList), 2, 2))
    for i,f in enumerate(fileList):
        data = np.load(os.path.join(dir, f))['arr_0']
        if removePhaseWraps:
            if np.any(np.abs(np.diff(data))>=2*np.pi):
                spectList[i] = np.nan
                fitList[i] = np.nan
                fitCovList[i] = np.nan
                continue
        _, spect, popt, pcov = fitPhaseNoiseSpectrum(data)
        spectList[i] = spect
        fitList[i] = popt
        fitCovList[i] = pcov
    

    return spectList, fitList, fitCovList

def plotNoiseFloors(popt, freqs=None):
    if freqs is None:
        plt.plot(popt[:,0])
    else:
        plt.plot(freqs, popt[:,0])
        plt.xlabel('Frequency (Hz)')

    plt.ylabel('Phase Noise Floor (dBc/Hz)')

def getSpurCorrectionFactor(freqs, spectList, noiseFloorList, spurHalfWin=0, spurThresh=-88):
    """
    freqs - freqs within spectra (NOT list of tones)
    spectList - spectra NOT in dB
    noiseFloorList - also NOT in dB
    """
    correctedNoiseFloorList = np.copy(noiseFloorList)
    for i, spect in enumerate(spectList):
        spurPower, totalPower, _ = getSpurNoisePower(freqs, spect, 100, 100.e3, spurHalfWin, spurThresh)
        correctedNoiseFloorList[i] *= (totalPower/(totalPower - spurPower))

    return correctedNoiseFloorList

def getSpurCorrectionFactorOptFilt(freqs, spectList, noiseFloorList, optFiltCalc, lf=100, hf=300.e3, spurThresh=-82, fftLen=65536):
    """
    freqs - freqs within spectra (NOT list of tones)
    spectList - spectra NOT in dB
    noiseFloorList - also NOT in dB
    """
    optFiltCalc.cfg.update('noise.nwindow', fftLen)
    template = optFiltCalc.result['template'][:50]
    correctedNoiseFloorList = np.copy(noiseFloorList)
    for i, spect in enumerate(spectList):
        optFiltCalc.clear_filter()
        optFiltCalc.result['psd'] = spect
        optFiltCalc.make_filter()
        optFilt = optFiltCalc.result['filter']

        optFiltCalc.clear_filter()
        spectNoSpurs = removeSpurs(freqs, spect, lf, hf, spurThresh)
        optFiltCalc.result['psd'] = spectNoSpurs
        optFiltCalc.make_filter()
        optFiltNoSpurs = optFiltCalc.result['filter']

        var = 1/np.sum(optFilt*template[::-1])
        varNS = 1/np.sum(optFiltNoSpurs*template[::-1])
        correctedNoiseFloorList[i] *= var/varNS

    return correctedNoiseFloorList





def getSpurMask(spectDB, spurHalfWin, spurThresh):
    peakInds, _ = signal.find_peaks(spectDB, height=spurThresh)
    peakMask = np.zeros(len(spectDB), dtype=np.bool)
    peakMask[peakInds] = True
    for i in range(spurHalfWin):
        peakMask |= np.roll(peakMask, i+1)
        peakMask |= np.roll(peakMask, -i-1)

    return peakInds, peakMask

def removeSpurs(freqs, spect, lf, hf, spurThresh=-88):
    lfInd = np.argmin(np.abs(freqs-lf))
    hfInd = np.argmin(np.abs(freqs-hf))
    spectInBand = spect[lfInd:hfInd]
    spectDBInBand = 10*np.log10(spectInBand)
    freqsInBand = freqs[lfInd:hfInd]
    spect = np.array(spect)

    peakInds, _ = getSpurMask(spectDBInBand, 1, spurThresh)
    peakInds += lfInd
    peakValInds = peakInds + 20
    spect[peakInds] = spect[peakValInds]
    for i in range(20):
        spect[peakInds + i] = spect[peakValInds]
        spect[peakInds - i] = spect[peakValInds]
    return spect
    

def getSpurNoisePower(freqs, spect, lf, hf, spurHalfWin=2, spurThresh=-88, spectWeights=None):
    """
    spect is NOT in dB
    lf and hf are freq bounds in Hz
    """
    lfInd = np.argmin(np.abs(freqs-lf))
    hfInd = np.argmin(np.abs(freqs-hf))
    spectInBand = spect[lfInd:hfInd]
    spectDBInBand = 10*np.log10(spectInBand)
    freqsInBand = freqs[lfInd:hfInd]
    df = freqsInBand[1] - freqsInBand[0]

    
    peakInds, peakMask = getSpurMask(spectDBInBand, spurHalfWin, spurThresh)
    if spectWeights is not None:
        spectWeightsInBand = spectWeights[lfInd:hfInd]
        totalPower = np.sum(spectInBand*spectWeightsInBand)*df
        spurPower = np.sum(spectInBand[peakMask]*spectWeightsInBand[peakMask])*df

    else: 
        spurPower = np.sum(spectInBand[peakMask])*df
        totalPower = np.sum(spectInBand)*df

    return spurPower, totalPower, peakInds + lfInd #return peakInds wrt full spectrum


    

def calcRoomTempNoise(inputPower, adcAtten0, adcAtten1, temp=290., rtAmpNT=438.4): 
    """
    parameters
    ----------
        inputPower - in dBm
        adcAtten0 - dB
        adcAtten1 - dB
    """
    rtAmpGainDB = 15. #dB
    rtAmpGain = 10**(rtAmpGainDB/10.)
    bw = 2.e9 #Hz
    splitterLoss = 4 #dB
    rtNoise = 4*1.38065E-023*temp*bw*1000 #milliwatts
    rtAmpNoise = 4*1.38065E-023*(rtAmpNT)*bw*1000

    tonePowerDB = inputPower - splitterLoss
    
    #first amplifier
    noise = rtAmpGain*(rtAmpNoise + rtNoise)
    tonePowerDB += rtAmpGainDB
    print 'firstAmp:', 10*np.log10(noise)

    #first attenuator
    noise = rtNoise + noise/(10**(adcAtten0/10.))
    tonePowerDB -= adcAtten0 
    print 'firstAmp:', 10*np.log10(noise)

    #second amplifier
    #NOTE: I think I might be adding rtNoise twice; since I add it above as well as a term in rtAmpNoise;
    # might explain 1-2 dB discrepancy in plots...
    noise += rtAmpNoise 
    noise *= rtAmpGain
    tonePowerDB += rtAmpGainDB
    print 'firstAmp:', 10*np.log10(noise)

    #second attenuator
    noise = rtNoise + noise/(10**(adcAtten1/10.))
    tonePowerDB -= adcAtten1
    print 'firstAmp:', 10*np.log10(noise)

    #third amplifier
    noise += rtAmpNoise 
    noise *= rtAmpGain
    tonePowerDB += rtAmpGainDB
    print 'firstAmp:', 10*np.log10(noise)

    #3dB atten
    noise = rtNoise + noise/(10**(0.3))
    tonePowerDB -= 3
    print 'firstAmp:', 10*np.log10(noise)

    #fourth amplifier
    noise += rtAmpNoise 
    noise *= rtAmpGain
    tonePowerDB += rtAmpGainDB
    print 'firstAmp:', 10*np.log10(noise)

    relNoise = noise/(2*bw*10**(tonePowerDB/10.))
    return 10*np.log10(relNoise)
    

def calcHEMTNoise(inputPower, hemtGain=40, hemtNT=2.4): 
    """
    parameters
    ----------
        inputPower - in dBm
        adcAtten0 - dB
        adcAtten1 - dB
    """
    bw = 2.e9 #Hz
    noise = 4*10**(hemtGain/10)*1.38065E-023*hemtNT*bw*1000 #milliwatts

    relNoise = noise/(2*bw*10**(inputPower/10.))
    return 10*np.log10(relNoise)

