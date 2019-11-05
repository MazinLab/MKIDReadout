import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

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
    popt, pcov = spo.curve_fit(flatoverf, freqs[1:2000], spect[1:2000], sigma=spect[1:2000])
    return freqs, spect, popt, pcov
