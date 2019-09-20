import numpy as np
import logging
import time
import matplotlib.pyplot as plt

from mkidreadout.configuration.sweepdata import SweepMetadata

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
sh.setFormatter(fmt)
log.addHandler(sh)


class Correlator(object):
    def __init__(self, oldPath='', newPath='', beammapPath=''):
        start = time.time()
        log.info("Initializing correlation")
        self.old = SweepMetadata(file=oldPath)
        self.new = SweepMetadata(file=newPath)
        # self.beammap = Beammap(file=beammapPath)
        self.cleanData()

        self.bestShift = 0
        self.shifts = [np.linspace(-5e6, 5e6, 2001), np.linspace(-1e3, 1e3, 2001)]
        self.resIDMatches = None

        self.correlateLists()
        end = time.time()
        log.debug("ID correlation took {} seconds".format(end - start))

    def correlateLists(self):
        log.info("Finding best frequency shifts")
        self.findBestFreqShift()
        log.info("Applying a shift of {} MHz".format(self.bestShift/1.e6))
        self.applyBestShift()
        log.info("Matching resonator IDs to shifted data")
        self.matchResIDs()
        log.info("Resolving resonator IDs assigned multiple times in matched ID lists")
        self.cleanResIDMatches()

    def cleanData(self):
        oldmask = (~np.isnan(self.old.atten)) & (self.old.flag == 1)
        self.oldFreq = self.old.freq[oldmask]
        self.oldRes = self.old.resIDs[oldmask]
        newmask = (~np.isnan(self.new.atten)) & (self.new.flag == 1)
        self.newFreq = self.new.freq[newmask]
        self.newRes = self.new.resIDs[newmask]

        shortmaskO = (self.oldFreq >= 4.e9) & (self.oldFreq <= 4.25e9)
        shortmaskN = (self.newFreq >= 4.e9) & (self.newFreq <= 4.25e9)
        self.oldFreq = self.oldFreq[shortmaskO]
        self.oldRes = self.oldRes[shortmaskO]
        self.newFreq = self.newFreq[shortmaskN]
        self.newRes = self.newRes[shortmaskN]

    def findBestFreqShift(self):
        self.avgResidual = []
        self.newAvgResidual = []
        for i in self.shifts[0]:
            match = self.matchFrequencies(self.newFreq, self.oldFreq, i)
            residuals = abs(match[0]-match[1])
            self.avgResidual.append(np.average(residuals))
        mask = self.avgResidual == np.min(self.avgResidual)
        self.bestShift = self.shifts[0][mask][0]
        self.shifts[1] = self.shifts[1] + self.bestShift
        log.info("The best shift after pass one is {} MHz".format(self.bestShift/1.e6))
        for i in self.shifts[1]:
            match = self.matchFrequencies(self.newFreq, self.oldFreq, i)
            residuals = abs(match[0]-match[1])
            self.newAvgResidual.append(np.average(residuals))
        newmask = self.newAvgResidual == np.min(self.newAvgResidual)
        self.bestShift = self.shifts[1][newmask][0]
        log.info("The best shift is {} MHz".format(self.bestShift/1.e6))

    def matchFrequencies(self, list1, list2, shift=0):
        if len(list1) > len(list2):
            longer = list1 + shift
            shorter = list2
            matches = np.zeros((2, len(longer)))
            matches[0] = longer
            for i in range(len(longer)):
                residual = abs(shorter - longer[i])
                mask = residual == min(residual)
                matches[1][i] = shorter[mask][0]
        else:
            longer = list2
            shorter = list1 + shift
            matches = np.zeros((2, len(longer)))
            matches[1] = longer
            for i in range(len(longer)):
                residual = abs(shorter - longer[i])
                mask = residual == min(residual)
                matches[0][i] = shorter[mask][0]

        return matches

    def applyBestShift(self):
        log.info("A shift of {} MHz was applied to the new frequency list".format(self.bestShift/1.e6))
        self.shiftedFreq = self.newFreq + self.bestShift

    def matchResIDs(self):
        self.bestMatches = self.matchFrequencies(list1=self.shiftedFreq, list2=self.oldFreq)
        self.resIDMatches = np.full_like(self.bestMatches, np.nan, dtype=float)
        self.resIDMatches[0] = self._matchIDtoFreq(self.bestMatches[0], self.shiftedFreq, self.newRes)
        self.resIDMatches[1] = self._matchIDtoFreq(self.bestMatches[1], self.oldFreq, self.oldRes)

    def _matchIDtoFreq(self, freqList, freqListFromData, resIDList):
        matchedList = np.full_like(freqList, np.nan)
        for i in range(len(matchedList)):
            index = np.where(freqListFromData == freqList[i])[0]
            if len(index) > 1:
                index = index[0]
            matchedList[i] = resIDList[index]
        return matchedList

    def cleanResIDMatches(self):
        doubleList = []
        for i in range(len(self.resIDMatches)):
            for j in self.resIDMatches[i]:
                mask = self.resIDMatches[i] == j
                r = self.resIDMatches[i][mask]
                if len(r) > 1:
                    doubleList.append(r[0])
            doubleList = list(set(doubleList))
            for k in doubleList:
                self._resolveDoubleIDs(k, i)


    def _resolveDoubleIDs(self, ID, arraySide):
        idx = np.where(self.resIDMatches[arraySide] == ID)[0]
        toResolveIDs = self.resIDMatches[:, idx[0]:idx[1]+1]
        toResolveFreqs = self.bestMatches[:, idx[0]:idx[1]+1]
        freqResids = abs(toResolveFreqs[0] - toResolveFreqs[1])
        bestFreqMatch = freqResids == min(freqResids)
        for i in range(len(bestFreqMatch)):
            if not bestFreqMatch[i]:
                toResolveIDs[arraySide][i] = np.nan
        self.resIDMatches[:, idx[0]:idx[1]+1] = toResolveIDs

    def plotResults(self):
        plt.plot(self.newFreq, np.zeros(len(self.newFreq)), 'r.', label="New Data")
        plt.plot(self.oldFreq, np.zeros(len(self.oldFreq)), 'b.', label="Old Data")
        plt.plot(self.shiftedFreq, np.zeros(len(self.shiftedFreq))+1, 'm.', label="Shifted Data")
        plt.plot(self.oldFreq, np.zeros(len(self.oldFreq))+1, 'c.', label="Old Data")
        plt.show()
