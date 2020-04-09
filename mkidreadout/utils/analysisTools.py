import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import os

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
    noise = 4*1.38065E-023*temp*bw*1000 #milliwatts

    relNoise = noise/(2*bw*10**(inputPower/10.))
    return 10*np.log10(relNoise)

