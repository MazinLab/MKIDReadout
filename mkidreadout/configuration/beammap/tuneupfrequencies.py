"""
Class to take in old sweep metadata and new sweep metadata and correlate the frequency lists between the two to match
beammapped resonators from the original sweep to resonators in the new sweep. This is to reduce the need for
beammapping.
Author: Noah Swimmer - 20 September 2019.

TODO: Implement more diagnostic plotting
"""

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
    def __init__(self, oldPath='', newPath='', boardNum='000'):
        """
        Input paths to the old data, new data, and beammap. Will correlate the resIDs between the old and new frequency
        list using frequency data. Matched/cleaned resID lists are found in self.resIDMatches. Correlator class works on
        a 1/2 feedline basis (purely based on how the metadata files are stored. A metadata file with a full array
        would still work).
        self.resIDMatches[0] = New ResIDs
        self.resIDMatches[1] = Corresponding Old ResIDs
        """
        start = time.time()
        log.info("Initializing correlation")
        self.old = SweepMetadata(file=oldPath)
        self.new = SweepMetadata(file=newPath)
        self.board = boardNum
        self.cleanData()

        self.bestShift = 0
        self.shifts = np.linspace(-1e6, 1e6, 2001)
        self.resIDMatches = None

        self.correlateLists()
        end = time.time()
        log.debug("ID correlation took {} seconds".format(end - start))

    def correlateLists(self):
        """
        Runs the actual algorithm that matches frequencies by shifting the new to try to match to the old, associates
        the old to new resIDs, and cleans up new/old IDs that were assigned twice to remove ambiguity.
        """
        log.info("Finding best frequency shifts")
        self.findBestFreqShift()
        log.info("Applying a shift of {} MHz".format(self.bestShift/1.e6))
        self.applyBestShift()
        log.info("Matching resonator IDs to shifted data")
        self.matchResIDs()
        log.info("Resolving resonator IDs assigned multiple times in matched ID lists")
        self.cleanResIDMatches()
        log.info("Badly mismatched resonators being removed.")
        self.ensureGoodFinalFreqMatches()

    def cleanData(self):
        """
        Strips the input metadata of all bad or unassigned resonators. Could be updated to clean it in a different way
        depending on what resonator information is desired.
        """
        oldmask = (~np.isnan(self.old.atten)) & (self.old.flag == 1)
        self.oldFreq = self.old.freq[oldmask]
        self.oldRes = self.old.resIDs[oldmask]
        newmask = (~np.isnan(self.new.atten)) & (self.new.flag == 1)
        self.newFreq = self.new.freq[newmask]
        self.newRes = self.new.resIDs[newmask]

        # shortmaskO = (self.oldFreq <= 4.5e9) & (self.oldFreq >= 4.e9)
        # shortmaskN = (self.newFreq <= 4.5e9) & (self.newFreq >= 4.e9)
        # self.oldFreq = self.oldFreq[shortmaskO]
        # self.oldRes = self.oldRes[shortmaskO]
        # self.newFreq = self.newFreq[shortmaskN]
        # self.newRes = self.newRes[shortmaskN]

    def findBestFreqShift(self):
        """
        Performs two frequency 'sweeps' to determine the frequency shift which minimizes the error between old and new
        frequency lists. Sweep 1 is coarse and sweeps +/- 5 MHz in 5 kHz steps. Once a frequency shift is found in this
        space, a finer search of +/-1 kHz is performed in 1 Hz steps.
        Sets self.bestShift, the ideal frequency shift to match the new data to the old.
        """
        self.avgResidual = []
        self.newAvgResidual = []
        for i in self.shifts:
            match = self.matchFrequencies(self.newFreq, self.oldFreq, i)
            residuals = abs(match[0]-match[1])
            self.avgResidual.append(np.average(residuals))
        mask = self.avgResidual == np.min(self.avgResidual)
        self.bestShift = self.shifts[mask][0]
        log.info("The best shift after pass one is {} MHz".format(self.bestShift/1.e6))

        tempMatch = self.matchFrequencies(self.newFreq, self.oldFreq, self.bestShift)
        tempResiduals = tempMatch[0]-tempMatch[1]
        averageResidual = np.average(tempResiduals)
        log.info("The average residual after matching frequency lists is {} MHz".format(averageResidual/1.e6))
        self.bestShift = self.bestShift - averageResidual
        log.info("The best shift is {} MHz".format(self.bestShift/1.e6))

    def matchFrequencies(self, list1, list2, shift=0):
        """
        Takes two frequency lists (by convention list1 is the new data, list2 is the old data) and a shift in Hz to
        apply to the new data.
        Returns a (2-by-length of longer frequency list) where each new frequency is matched to the closest frequency in
        the old frequency list
        """
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
        """
        After figuring out the best frequency shift to apply to the new frequency list, applies the shift.
        Stores the shifted frequencies in self.shiftedFreq
        """
        log.info("A shift of {} MHz was applied to the new frequency list".format(self.bestShift/1.e6))
        self.shiftedFreq = self.newFreq + self.bestShift

    def matchResIDs(self):
        """
        Creates a (2-by-length of longer frequency list) array with resIDs from the new data and old data that
        best correspond to each other. resIDs in the same column correspond to each other. This function does not take
        into account doubles where one resonator in one list is matched to multiple resonators in another.
        """
        tempFreq = self.matchFrequencies(list1=self.shiftedFreq, list2=self.oldFreq)
        tempID = np.full_like(tempFreq, np.nan, dtype=float)
        tempID[0] = self._matchIDtoFreq(tempFreq[0], self.shiftedFreq, self.newRes)
        tempID[1] = self._matchIDtoFreq(tempFreq[1], self.oldFreq, self.oldRes)
        tempFreq = np.transpose(tempFreq)
        tempID = np.transpose(tempID)

        self.resIDMatches = np.full((len(tempID), len(tempID[0])+1), np.nan)
        self.resIDMatches[:, 0] = tempID[:, 1]
        self.resIDMatches[:, 1] = tempID[:, 0]
        self.freqMatches = np.full_like(tempFreq, np.nan)
        self.freqMatches[:, 0] = tempFreq[:, 1]
        self.freqMatches[:, 1] = tempFreq[:, 0]

    def _matchIDtoFreq(self, freqList, freqListFromData, resIDList):
        """
        Takes a shifted frequency list and corresponding unshifted requency list and resID list and associates resIDs
        with frequencies.
        Returns a list of resIDs corresponding to the shifted (or not) frequency list.
        """
        matchedList = np.full_like(freqList, np.nan)
        for i in range(len(matchedList)):
            index = np.where(freqListFromData == freqList[i])[0]
            if len(index) > 1:
                index = index[0]
            matchedList[i] = resIDList[index]
        return matchedList

    def cleanResIDMatches(self):
        """
        Function to resolve doubles where a resonator in one list was assigned to multiple resonators in the other. This
        will resolve doubles in the new AND old list.
        """
        for j in range(len(self.freqMatches[0])):
            doubles = []
            for i in self.resIDMatches[:, j]:
                mask = self.resIDMatches[:, j] == i
                r = self.resIDMatches[:, j][mask]
                if len(r) > 1:
                    doubles.append(r[0])
            doubles = list(set(doubles))
            if len(doubles) != 0:
                for k in doubles:
                    self._resolveDoubleIDs(k, j)

    def _resolveDoubleIDs(self, ID, side):
        """
        Helper function to modify the self.resIDMatches object when resolving doubles.
        """
        idx = np.where(self.resIDMatches[:, side] == ID)[0]
        toResolveIDs = self.resIDMatches[idx]
        toResolveFreqs = self.freqMatches[idx]
        freqResidual = abs(toResolveFreqs[:, 0] - toResolveFreqs[:, 1])
        bestFreqMatch = freqResidual == min(freqResidual)
        for i in range(len(toResolveIDs)):
            if not bestFreqMatch[i]:
                toResolveIDs[i][side] = np.nan
        self.resIDMatches[idx] = toResolveIDs

    def ensureGoodFinalFreqMatches(self):
        """
        Ensures that all of the resID matches are also well matched in frequency space.
        Flags: 0 = Both resIDs matched and frequencies match
               1 = ResID in old powerSweep has no analog in new powerSweep (no DAC tone)
               2 = ResID in new powerSweep has no analog in old powerSweep (beammap failed)
               3 = Closest frequency match was too far to reasonably be the same resonator (beammap failed)
        """
        self.residuals = abs(self.freqMatches[:, 0] - self.freqMatches[:, 1])
        # for i in range(len(self.residuals)):
        #     if self.residuals[i] >= 100e3:
        #         self.resIDMatches[i][2] = 3
        for i in self.resIDMatches:
            if np.isnan(i[2]):
                if not np.isnan(i[0]) and not np.isnan(i[1]):
                    m = (self.resIDMatches[:, 0] == i[0]) & (self.resIDMatches[:, 1] == i[1])
                    if self.residuals[m] <= 100e3:
                        i[2] = 0
                    else:
                        i[2] = 3
                elif not np.isnan(i[0]) and np.isnan(i[1]):
                    i[2] = 1
                elif np.isnan(i[0]) and not np.isnan(i[1]):
                    i[2] = 2

    def plotResults(self):
        """
        TODO: Add functionality
        """
        plt.plot(self.newFreq, np.zeros(len(self.newFreq)), 'r.', label="New Data")
        plt.plot(self.oldFreq, np.zeros(len(self.oldFreq)), 'b.', label="Old Data")
        plt.plot(self.shiftedFreq, np.zeros(len(self.shiftedFreq))+1, 'm.', label="Shifted Data")
        plt.plot(self.oldFreq, np.zeros(len(self.oldFreq))+1, 'c.', label="Old Data")
        plt.show()

    def saveResult(self):
        np.savetxt(str(self.board)+"_correlated_IDs.txt", self.resIDMatches, header="Old New", fmt='%.1f')
