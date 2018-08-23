import numpy as np
import os, sys
import matplotlib.pyplot as plt

def streamSpectrum(iVals,qVals):
    #TODO break out into library module
    sampleRate = 2.e9 # 2GHz
    MHz = 1.e6
    adcFullScale = 2.**11

    signal = iVals+1.j*qVals
    signal = signal / adcFullScale

    nSamples = len(signal)
    spectrum = np.fft.fft(signal)
    spectrum = 1.*spectrum / nSamples

    freqsMHz = np.fft.fftfreq(nSamples)*sampleRate/MHz

    freqsMHz = np.fft.fftshift(freqsMHz)
    spectrum = np.fft.fftshift(spectrum)

    spectrumDb = 20*np.log10(np.abs(spectrum))

    peakFreq = freqsMHz[np.argmax(spectrumDb)]
    peakFreqPower = spectrumDb[np.argmax(spectrumDb)]
    times = np.arange(nSamples)/sampleRate * MHz
    #print 'peak at',peakFreq,'MHz',peakFreqPower,'dB'
    return {'spectrumDb':spectrumDb,'freqsMHz':freqsMHz,'spectrum':spectrum,'peakFreq':peakFreq,'times':times,'signal':signal,'nSamples':nSamples}

def checkSpectrumForSpikes(specDict):
    # TODO Merge with roach2controls as helper
    sortedSpectrum=np.sort(specDict['spectrumDb'])
    spectrumFlag=False
    #checks if there are spikes above the forest. If there are less than 5 tones at least 10dB above the forest are cosidered spikes
    for i in range(-5,-1):
        if (sortedSpectrum[-1]-sortedSpectrum[i])>10:
            spectrumFlag=True
            break
    return spectrumFlag
