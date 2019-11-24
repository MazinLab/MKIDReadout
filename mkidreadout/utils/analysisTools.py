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

def batchFitPhaseNoiseDir(dir):
    fileList = os.listdir(dir)
    freqs, spect = getPhaseNoiseSpectrum(np.load(os.path.join(dir, fileList[0]))['arr_0'])
    fitList = np.zeros((len(fileList), 2))
    spectList = np.zeros((len(fileList), len(spect)))
    fitCovList = np.zeros((len(fileList), 2, 2))
    for i,f in enumerate(fileList):
        data = np.load(os.path.join(dir, f))['arr_0']
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



