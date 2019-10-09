"""
Class to take in old sweep metadata and new sweep metadata and correlate the frequency lists between the two to match
beammapped resonators from the original sweep to resonators in the new sweep. This is to reduce the need for
beammapping.
Author: Noah Swimmer - 20 September 2019.

TODO: Implement more diagnostic plotting
TODO: Add function to take in beammap and determine number of resonators mapped in old list and not found in new
TODO: Make properties/setters, document new code
"""

import numpy as np
import logging
import time
import matplotlib.pyplot as plt
import mkidcore.objects as mko
from mkidcore.pixelflags import beamMapFlags

from mkidreadout.configuration.sweepdata import SweepMetadata

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
sh.setFormatter(fmt)
log.addHandler(sh)


class Correlator(object):
    def __init__(self, oldPath=None, newPath=None, boardNum=None, frequencyCutoff=500e3):
        """
        Input paths to the old data, new data, and beammap. Will correlate the resIDs between the old and new frequency
        list using frequency data. Matched/cleaned resID lists are found in self.resIDMatches. Correlator class works on
        a 1/2 feedline basis (purely based on how the metadata files are stored. A metadata file with a full array
        would still work).
        """
        self.board = boardNum
        if oldPath and newPath:
            self.old = SweepMetadata(file=oldPath)
            self.new = SweepMetadata(file=newPath)
            self.cleanData()

        self.bestShift = 0
        self.shifts = np.linspace(-1e6, 1e6, 2001)
        self._freqCutoff = frequencyCutoff

        self.resIDMatches = None
        self.freqMatches = None
        self.residuals = None

    def correlateLists(self):
        """
        Runs the actual algorithm that matches frequencies by shifting the new to try to match to the old, associates
        the old to new resIDs, and cleans up new/old IDs that were assigned twice to remove ambiguity.
        """
        start = time.time()
        log.info("Correlating board {}".format(self.board))
        log.info("Finding best frequency shifts")
        self.findBestFreqShift()
        log.info("Applying a shift of {} MHz".format(self.bestShift/1.e6))
        self.applyBestShift()
        log.info("Flagging Resonators")
        self.handleFlags()
        end = time.time()
        log.debug("ID correlation took {} seconds".format(end - start))

    def cleanData(self):
        """
        Strips the input metadata of all bad or unassigned resonators. Could be updated to clean it in a different way
        depending on what resonator information is desired.
        """
        oldmask = (np.isfinite(self.old.atten)) & (self.old.flag == 1)
        self.oldFreq = self.old.freq[oldmask]
        self.oldRes = self.old.resIDs[oldmask]
        oldIndices = np.argsort(self.oldFreq)
        self.oldFreq = self.oldFreq[oldIndices]
        self.oldRes = self.oldRes[oldIndices]
        newmask = (np.isfinite(self.new.atten)) & (self.new.flag == 1)
        self.newFreq = self.new.freq[newmask]
        self.newRes = self.new.resIDs[newmask]
        newIndices = np.argsort(self.newFreq)
        self.newFreq = self.newFreq[newIndices]
        self.newRes = self.newRes[newIndices]

        # shortmaskO = (self.oldFreq <= 5.87e9) #& (self.oldFreq >= 4.e9)
        # shortmaskN = (self.newFreq <= 5.87e9) #& (self.newFreq >= 4.e9)
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
        averageResidual = np.median(tempResiduals)
        log.info("The median residual after matching frequency lists is {} MHz".format(averageResidual/1.e6))
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

    def _createResidualGrid(self):
        self.oldFreq = np.reshape(self.oldFreq, (len(self.oldFreq), 1))
        self._residualGrid = abs(self.oldFreq - self.shiftedFreq)

    @property
    def residualGrid(self):
        if self._residualGrid is not None:
            return self._residualGrid
        self._createResidualGrid()

    @property
    def flagGrid(self):
        if self._flagGrid is not None:
            return self._flagGrid
        self._flagGrid = self._createFlagGrid()
        return self._flagGrid

    def _createFlagGrid(self):
        flagGrid = np.zeros(self._residualGrid.shape)
        minResidsNew = np.argmin(self._residualGrid, axis=0)
        minResidsOld = np.argmin(self._residualGrid, axis=1)
        coordsNew = set([(j, i) for i, j in enumerate(minResidsNew)])
        coordsOld = set([(i, j) for i, j in enumerate(minResidsOld)])
        goodCoords = np.array(list(coordsNew & coordsOld))
        noNewMatch = np.array(list(coordsNew - (coordsNew & coordsOld)))
        noOldMatch = np.array(list(coordsOld - (coordsNew & coordsOld)))

        flagGrid[goodCoords[:, 0], goodCoords[:, 1]] = 1
        flagGrid[noNewMatch[:, 0], noNewMatch[:, 1]] = 2
        flagGrid[noOldMatch[:, 0], noOldMatch[:, 1]] = 3

        m = (flagGrid == 1) & (self._residualGrid >= self._freqCutoff)
        flagGrid[m] = 4
        self._flagGrid = flagGrid
        # self.fixMismatches()

    @property
    def freqCutoff(self):
        return self._freqCutoff

    @freqCutoff.setter
    def freqCutoff(self, x):
        self._freqCutoff = x
        self._createFlagGrid()

    def handleFlags(self):
        self._createResidualGrid()
        self._createFlagGrid()
        self.oldFlag = np.zeros(len(self.oldRes))
        self.newFlag = np.zeros(len(self.newRes))

        self.resIDMatches = []
        self.freqMatches = []
        m1 = self._flagGrid == 1
        coords1 = np.where(m1)
        m2 = self._flagGrid == 2
        coords2 = np.where(m2)
        m3 = self._flagGrid == 3
        coords3 = np.where(m3)
        m4 = self._flagGrid == 4
        coords4 = np.where(m4)

        for i, j in zip(coords1[0], coords1[1]):
            self.resIDMatches.append(np.array([self.oldRes[i], self.newRes[j], 1]))
            self.freqMatches.append(np.array([self.oldFreq[i], self.newFreq[j]]))
            self.oldFlag[i] = 1
            self.newFlag[j] = 1
        for i, j in zip(coords2[0], coords2[1]):
            self.resIDMatches.append(np.array([np.nan, self.newRes[j], 2]))
            self.freqMatches.append(np.array([np.nan, self.newFreq[j]]))
            self.newFlag[j] = 2
        for i, j in zip(coords3[0], coords3[1]):
            self.resIDMatches.append(np.array([self.oldRes[i], np.nan, 3]))
            self.freqMatches.append(np.array([self.oldFreq[i], np.nan]))
            self.oldFlag[i] = 3
        for i, j in zip(coords4[0], coords4[1]):
            self.resIDMatches.append(np.array([self.oldRes[i], self.newRes[j], 4]))
            self.freqMatches.append(np.array([self.oldFreq[i], self.newFreq[j]]))
            self.oldFlag[i] = 4
            self.newFlag[j] = 4

        self.resIDMatches = np.array(self.resIDMatches)
        self.freqMatches = np.array(self.freqMatches)
        self.residuals = self.freqMatches[:, 0] - self.freqMatches[:, 1]

        assert(not np.any(self.oldFlag == 0))
        assert(not np.any(self.newFlag == 0))

    def load(self, fName):
        oldIDs, newIDs, correlatorFlags, oldFreqs, newFreqs = np.loadtxt(fName, unpack=True)
        self.resIDMatches = np.transpose(np.array([oldIDs, newIDs, correlatorFlags]))
        self.freqMatches = np.transpose(np.array([oldFreqs, newFreqs]))
        self.residuals = self.freqMatches[:, 0] - self.freqMatches[:, 1]

    def saveResult(self):
        np.savetxt(str(self.board)+"_correlated_IDs.txt", np.concatenate((self.resIDMatches, self.freqMatches), axis=1),
                   header="Old New Flag OldFreq NewFreq", fmt='%.1f %.1f %i %9.7f %9.7f')

    def generateBMResMask(self, beammap):
        self.bmResMask = np.zeros(len(self.oldRes)).astype(bool)
        for i, res in enumerate(self.oldRes):
            self.bmResMask[i] = (beammap.getResonatorFlag(res) == beamMapFlags['good'])

    def fixMismatches(self):
        centers = np.transpose(np.where(self._flagGrid == 2))
        for i in centers:
            square = self._flagGrid[i[0]-1:i[0]+2, i[1]-1:i[1]+2]
            residualSq = self.residualGrid[i[0]-1:i[0]+2, i[1]-1:i[1]+2]
            # if (3 in square) and (2 in square):
            print(i)
            if 3 in square:
                newSq = self.resolveSquare(square, residualSq)
                self._flagGrid[i[0]-1:i[0]+2, i[1]-1:i[1]+2] = newSq
            else:
                pass

    def resolveSquare(self, square, residuals):
        newSq = np.copy(square)
        coords3 = np.transpose(np.where(newSq == 3))
        coords1 = np.transpose(np.where(newSq == 1))
        coord2 = np.transpose(np.where(newSq == 2))
        if len(newSq[newSq == 3]):
            dist = []
            for i in coords3:
                d = np.sqrt((i[0]-coord2[0][0])**2 + (i[1]-coord2[0][1])**2)
                dist.append(d)
            m = dist != min(dist)
            farCoord = coords3[m]
            newSq[farCoord[:, 0], farCoord[:, 1]] = 6

        row = np.where(newSq == 3)[0]
        col = np.where(newSq == 2)[1]
        mask = (newSq == 2) | (newSq == 3)
        newSq[mask] = 0
        newSq[row, col] = 5
        coord5 = np.array([row, col])

        vectorsTo1 = []
        for i in coords1:
            vector = np.array([coord5[0] - i[0], coord5[1] - i[1]])
            vectorsTo1.append(vector)

        toDelete=[]
        for i, j in enumerate(vectorsTo1):
            if (j[0][0] == 1) and (j[1][0] == -1):
                pass
            elif (j[0][0] == -1) and (j[1][0] == 1):
                pass
            elif (j[0][0] == 1) and (j[1][0] == 1):
                toDelete.append(i)
            elif (j[0][0] == -1) and (j[1][0] == -1):
                toDelete.append(i)
            else:
                toDelete.append(i)

        coords = np.transpose(np.where((newSq == 1) | (newSq == 5)))
        vectorsTo1 = np.delete(vectorsTo1, toDelete, axis=0)
        coords = np.delete(coords, toDelete, axis=0)
        newCoords = np.copy(coords)
        for i in vectorsTo1:
            if ((i[0] == 1) and (i[1] == -1)) or ((i[0] == 1) and (i[1] == -1)):
                totalResidualsOriginal = sum(residuals[coords[:, 0], coords[:, 1]])
                colVals = newCoords[:, 1]
                colVals = colVals[::-1]
                newCoords[:, 1] = colVals

                totalResidualsNew = sum(residuals[newCoords[:, 0], newCoords[:, 1]])
                if totalResidualsOriginal <= totalResidualsNew:
                    return square
                else:
                    newSq[coords[:, 0], coords[:, 1]] = 0
                    newSq[newCoords[:, 0], newCoords[:, 1]] = 1
                    gMask = newSq == 5
                    bMask = newSq == 6
                    newSq[gMask] = 1
                    newSq[bMask] = 3

        gMask = newSq == 5
        bMask = newSq == 6
        newSq[gMask] = 1
        newSq[bMask] = 3
        return newSq