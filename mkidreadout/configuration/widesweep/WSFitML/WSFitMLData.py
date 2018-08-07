import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os, sys

class WSFitMLData:
    def __init__(self, filenameList):
        self.freqs = np.empty(0)
        self.iVals = np.empty(0)
        self.qVals = np.empty(0)
        self.iqVels = np.empty(0)
        self.boundaryInds = np.empty(0)
        self.filenameList = np.asarray(filenameList)

        for fn in self.filenameList:
            freqs, iVals, qVals = np.loadtxt(fn, unpack=True, skiprows=3)
            iqVels = np.sqrt(np.diff(iVals)**2+np.diff(qVals)**2)
            self.boundaryInds = np.append(self.boundaryInds, len(self.freqs))
            self.freqs = np.append(self.freqs, freqs)
            self.iVals = np.append(self.iVals, iVals)
            self.qVals = np.append(self.qVals, qVals)
            self.iqVels = np.append(self.iqVels, iqVels)

        self.boundaryInds = self.boundaryInds[1:]
        self.mags = np.sqrt(self.iVals**2+self.qVals**2)
        self.magsdb = 20*np.log10(self.mags)
    
    def loadPeaks(self, flag='good'):
        if flag=='good':
            self.peakLocs = np.empty(0)
        else:
            self.allPeakLocs = np.empty(0)
        for i,fn in enumerate(self.filenameList):
            fn = fn.split('.')[0]
            fn += '-freqs-' +  flag + '.txt'
            _, peakLocs, _ = np.loadtxt(fn, unpack=True)
            peakLocs += boundaryInds[i]
            if flag=='good':
                self.peakLocs = np.append(self.peakLocs, peakLocs)
            else:
                self.allPeakLocs = np.append(self.allPeakLocs, peakLocs)
    
    def filterMags(self, mags, order=4, rs=40, wn=0.005):
        b, a = signal.cheby2(order, rs, wn, btype='high', analog=False)
        return signal.filtfilt(b, a, mags)

        
