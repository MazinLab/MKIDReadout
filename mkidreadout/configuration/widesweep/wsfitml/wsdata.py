import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os, sys


class WSFitMLData(object):
    def __init__(self, filenameList, freqStep=None):
        self.freqs = np.empty(0)
        self.iVals = np.empty(0)
        self.qVals = np.empty(0)
        self.iqVels = np.empty(0)
        self.freqStep = freqStep
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

    def stitchDigitalData(self):
        deltas = np.diff(self.freqs)
        boundaryInds = np.where(deltas<0)[0]
        boundaryDeltas = -deltas[boundaryInds]
        nOverlapPoints = (boundaryDeltas/self.freqStep).astype(int) + 1
        #startInds = np.append([0], boundaryInds+1)
        #endInds = np.append(boundaryInds, len(self.freqs))
        boundaryInds = boundaryInds + 1

        for i in range(len(boundaryInds)):
            lfMags = self.mags[boundaryInds[i] - nOverlapPoints[i] : boundaryInds[i]]
            hfMags = self.mags[boundaryInds[i] : boundaryInds[i] + nOverlapPoints[i]]
            hfWeights = np.linspace(0, 1, num=nOverlapPoints[i], endpoint=False)
            lfWeights = 1 - hfWeights
            self.mags[boundaryInds[i] : boundaryInds[i] + nOverlapPoints[i]] = lfWeights*lfMags + hfWeights*hfMags #set mags to average the overlap regions
            self.mags[boundaryInds[i] - nOverlapPoints[i] : boundaryInds[i]] = np.nan #set one side of overlap to 0
            self.freqs[boundaryInds[i] - nOverlapPoints[i] : boundaryInds[i]] = np.nan
            
        
        # stitching I/Q data not yet implemented so get rid of it for now
        self.iVals = None
        self.qVals = None

        self.mags = self.mags[~np.isnan(self.mags)]
        self.magsdb = 20*np.log10(self.mags)
        self.freqs = self.freqs[~np.isnan(self.freqs)]
    
    def saveData(self, fn):
        iVals = self.iVals
        qVals = self.qVals

        if iVals is None or qVals is None:
            iVals = self.mags
            qVals = np.zeros(len(iVals))

        np.savetxt(fn, np.transpose([self.freqs, iVals, qVals]), fmt='%0.9f %0.5f %0.5f')

