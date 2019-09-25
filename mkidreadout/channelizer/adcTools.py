import numpy as np

def streamSpectrum(iVals,qVals,nBins=None):
    #TODO break out into library module
    sampleRate = 2.e9 # 2GHz
    MHz = 1.e6
    adcFullScale = 2.**11

    signal = iVals+1.j*qVals
    signal = signal / adcFullScale
    nSamples = len(signal)

    if not (nBins is None):
        nAvgs = nSamples // nBins
        nSamplesToUse = nBins*nAvgs
        nSamplesPerFft = nBins
        foldedSignal = np.reshape(signal[:nSamplesToUse],(nAvgs,nBins))
    else:
        nAvgs = 1
        nSamplesToUse = nSamples
        nBins = nSamples
        nSamplesPerFft = nSamples
        foldedSignal = signal

    spectrum = np.fft.fft(foldedSignal)
    spectrum = 1.*spectrum / nSamplesPerFft

    freqsMHz = np.fft.fftfreq(nSamplesPerFft)*sampleRate/MHz

    freqsMHz = np.fft.fftshift(freqsMHz)
    spectrum = np.fft.fftshift(spectrum)
    powerSpectrum = np.abs(spectrum)**2
    if nAvgs > 1:
        powerSpectrum = np.average(powerSpectrum, axis=0)

    spectrumDb = 10*np.log10(powerSpectrum)

    peakFreq = freqsMHz[np.argmax(spectrumDb)]
    peakFreqPower = spectrumDb[np.argmax(spectrumDb)]
    times = np.arange(nSamples)/sampleRate * MHz
    #print 'peak at',peakFreq,'MHz',peakFreqPower,'dB'
    return {'spectrumDb':spectrumDb,'freqsMHz':freqsMHz,'spectrum':spectrum,'peakFreq':peakFreq,'times':times,'signal':signal,'nSamples':nSamplesPerFft}

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
