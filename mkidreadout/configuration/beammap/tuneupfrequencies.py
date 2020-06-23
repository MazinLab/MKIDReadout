"""
Class to take in old sweep metadata and new sweep metadata and correlate the frequency lists between the two to match
beammapped resonators from the original sweep to resonators in the new sweep. This is to reduce the need for
beammapping.
Author: Noah Swimmer - 20 September 2019.

TODO: Turn this into a command-line-runnable program
"""

import numpy as np
import logging
import time
import argparse
import glob
import matplotlib.pyplot as plt
import mkidcore.objects as mko
from mkidcore.pixelflags import beammap

from mkidreadout.configuration.sweepdata import SweepMetadata

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
sh.setFormatter(fmt)
log.addHandler(sh)


class Correlator(object):
    def __init__(self, oldPath=None, newPath=None, boardNum=None, frequencyCutoff=500e3, fitOrder=1):
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
        self.bestStretch = 1
        self.shifts = np.arange(-5e6, 5e6+1e4, 1e4)
        self.stretches = np.arange(0.998, 1.002, .00001)
        self._freqCutoff = frequencyCutoff
        self.shift_fit_coeffs = {}
        self.fit_data = {}
        self.correlation_freqs = []
        self.correlation_strengths = []
        self.fit_order = []
        self.fit_order.append(fitOrder)

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
        # self.findBestFreqShift1()
        # self.findBestFreqShift2()
        cfreqs = self.generate_freqs_to_correlate()
        cstrengths = self.correlate_freqs(cfreqs)
        fit_coefficients, fit_data = self.fit_freq_to_best_shift(cstrengths, cfreqs, self.fit_order[0])
        self.apply_best_shift(self.fit_order[0])
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

        # shortmaskO = (self.oldFreq <= 7.5e9) & (self.oldFreq >= 6e9)
        # shortmaskN = (self.newFreq <= 7.5e9) & (self.newFreq >= 6e9)
        # self.oldFreq = self.oldFreq[shortmaskO]
        # self.oldRes = self.oldRes[shortmaskO]
        # self.newFreq = self.newFreq[shortmaskN]
        # self.newRes = self.newRes[shortmaskN]

    def generate_freqs_to_correlate(self):
        bins1 = np.arange(0, len(self.newFreq)+30, 30)
        bins1[-1] = len(self.newFreq) - 1
        bins2 = np.arange(15, len(self.newFreq), 30)
        f1 = [np.array(self.newFreq[bins1[i]:bins1[i+1]]) for i in range(len(bins1)-1)]
        f2 = [np.array(self.newFreq[bins2[i]:bins2[i+1]]) for i in range(len(bins2)-1)]
        freqs = [j for i in zip(f1, f2) for j in i]
        self.correlation_freqs = freqs
        return freqs

    def stretch_shift_correlate(self):
        old_sweep = np.zeros(int(1e7))
        old_freqs_khz = np.array(self.oldFreq / 1e3, dtype=int)
        old_freqs_extended = np.array([np.arange(i-25, i+26, 1) for i in old_freqs_khz])
        old_freqs_mask = np.unique(old_freqs_extended.flatten())
        old_sweep[old_freqs_mask] = 1

        strengths = []
        for s in self.shifts:
            strengths_sub = []
            for j in self.stretches:
                new_sweep = np.zeros(int(1e7))
                new_freqs_khz = np.array(((self.newFreq * j) + s) / 1e3, dtype=int)
                new_freqs_extended = np.array([np.arange(i-25, i+26, 1) for i in new_freqs_khz])
                new_freqs_mask = np.unique(new_freqs_extended.flatten())
                new_sweep[new_freqs_mask] = 1
                strength = np.correlate(old_sweep, new_sweep)[0]
                print("Testing shift {}, stretch {}, stregth of correlation is {}".format(s, j, strength))
                strengths_sub.append(strength)
            strengths.append(strengths_sub)
        self.correlation_strengths = strengths
        return np.array(strengths)

    def correlate_freqs(self, freqs):
        old_sweep = np.zeros(int(1e7))
        old_freqs_khz = np.array(self.oldFreq / 1e3, dtype=int)
        old_freqs_extended = np.array([np.arange(i-25, i+26, 1) for i in old_freqs_khz])
        old_freqs_mask = np.unique(old_freqs_extended.flatten())
        old_sweep[old_freqs_mask] = 1

        strengths = []
        for num,j in enumerate(freqs):
            strengths_sub = []
            prev = time.time()
            for i in self.shifts:
                new_sweep = np.zeros(int(1e7))
                new_freqs_khz = np.array((j + i) / 1e3, dtype=int)
                new_freqs_extended = np.array([np.arange(freq-25, freq+26, 1) for freq in new_freqs_khz])
                new_freqs_mask = np.unique(new_freqs_extended.flatten())
                new_sweep[new_freqs_mask] = 1
                strength = np.correlate(old_sweep, new_sweep)[0]
                strengths_sub.append(strength)
            print("Peak strength in bin {} is {} at {} kHz. Correlation for this bin took {} "
                  "seconds".format(num, np.max(strengths_sub), self.shifts[np.where(strengths_sub==np.max(strengths_sub))], time.time() - prev))
            strengths.append(np.array(strengths_sub))
        self.correlation_strengths = strengths
        return np.array(strengths)

    def fit_freq_to_best_shift(self, correlation_strengths, freqs, fit_order):
        freq_vals = []
        shift_vals = []
        for i, j in enumerate(correlation_strengths):
            shift_idx = np.where(j == np.max(j))
            shift_vals.append(self.shifts[shift_idx])
            bin_freq = np.mean(freqs[i])
            freq_vals.append(bin_freq)
        data = zip(freq_vals, shift_vals)
        fit_data = np.array([(i[0], j) for i in data for j in i[1]])

        fit_coeffs = np.polyfit(fit_data[:, 0], fit_data[:, 1], fit_order)
        self.shift_fit_coeffs[fit_order] = fit_coeffs
        self.fit_data[fit_order] = fit_data
        return fit_coeffs, fit_data

    def apply_best_shift(self, fit_order):
        fit_function = np.poly1d(self.shift_fit_coeffs[fit_order])
        self.shiftedFreq = fit_function(self.newFreq) + self.newFreq

    # def findBestFreqShift1(self):
    #     """
    #     """
    #     log.debug("Pass 1, board {}".format(self.board))
    #     self.avgResidual = []
    #     for i,shift in enumerate(self.shifts):
    #         if i % 50 == 0:
    #             log.info("{}% done with pass 1".format(float(i)*100./len(self.shifts)))
    #         stretch_residuals = []
    #         for j in self.stretches:
    #             match = self.matchFrequencies(self.newFreq, self.oldFreq, shift, j)
    #             residuals = abs(match[0]-match[1])
    #             stretch_residuals.append(np.average(residuals))
    #         self.avgResidual.append(stretch_residuals)
    #     idx = np.where(self.avgResidual == np.min(self.avgResidual))
    #     # mask = self.avgResidual == np.min(self.avgResidual)
    #     self.bestShift = self.shifts[idx[0][0]]
    #     self.bestStretch = self.stretches[idx[1][0]]
    #     log.info("The best shift after pass one is {} MHz, the best stretch is {}".format(self.bestShift / 1.e6, self.bestStretch))
    #
    # def findBestFreqShift2(self):
    #     """
    #     """
    #     log.debug("Pass 2, board {}".format(self.board))
    #     self.fineAvgResidual = []
    #
    #     shift_step = self.shifts[1]-self.shifts[0]
    #     stretch_step = self.stretches[1]-self.stretches[0]
    #     newShifts = np.arange(self.bestShift-shift_step, self.bestShift+shift_step+1.e3, step=0.5e3)
    #     newStretches = np.arange(self.bestStretch-stretch_step, self.bestStretch+stretch_step+.000002, step=.000002)
    #     for i, shift in enumerate(newShifts):
    #         if i % 10 == 0:
    #             log.info("{}% done with pass 2".format(float(i)*100./len(newShifts)))
    #         stretch_residuals = []
    #         for j in newStretches:
    #             match = self.matchFrequencies(self.newFreq, self.oldFreq, shift, j)
    #             residuals = abs(match[0] - match[1])
    #             stretch_residuals.append(np.average(residuals))
    #         self.fineAvgResidual.append(stretch_residuals)
    #     idx = np.where(self.fineAvgResidual == np.min(self.fineAvgResidual))
    #     self.fineBestShift = newShifts[idx[0][0]]
    #     self.fineBestStretch = newStretches[idx[1][0]]
    #     log.info("The best shift after pass two is {} MHz, the best stretch is {}".format(self.fineBestShift / 1.e6, self.fineBestStretch))

    def matchFrequencies(self, list1, list2, shift=0, stretch=1):
        """
        Takes two frequency lists (by convention list1 is the new data, list2 is the old data) and a shift in Hz to
        apply to the new data.
        Returns a (2-by-length of longer frequency list) where each new frequency is matched to the closest frequency in
        the old frequency list
        """

        if len(list1) > len(list2):
            longer = (list1 * stretch) + shift
            shorter = list2
            matches = np.zeros((2, len(longer)))
            matches[0] = longer
            for i in range(len(longer)):
                residual = abs(shorter - longer[i])
                mask = residual == min(residual)
                matches[1][i] = shorter[mask][0]
        else:
            longer = list2
            shorter = (list1 * stretch) + shift
            matches = np.zeros((2, len(longer)))
            matches[1] = longer
            for i in range(len(longer)):
                residual = abs(shorter - longer[i])
                mask = residual == min(residual)
                matches[0][i] = shorter[mask][0]

        return matches

    # def applyBestShift(self):
    #     """
    #     After figuring out the best frequency shift to apply to the new frequency list, applies the shift.
    #     Stores the shifted frequencies in self.shiftedFreq
    #     """
    #     log.info("A shift of {} MHz was applied to the new frequency list".format(self.bestShift/1.e6))
    #     self.shiftedFreq = (self.newFreq * self.fineBestStretch) + self.fineBestShift

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

    def saveResult(self, path='.'):
        np.savetxt(path+'/'+str(self.board)+"_correlated_IDs.txt", np.concatenate((self.resIDMatches, self.freqMatches), axis=1),
                   header="Old New Flag OldFreq NewFreq", fmt='%.1f %.1f %i %9.7f %9.7f')

    def generateBMResMask(self, beammap):
        self.bmResMask = np.zeros(len(self.oldRes)).astype(bool)
        for i, res in enumerate(self.oldRes):
            self.bmResMask[i] = (beammap.getResonatorFlag(res) == beammap['good'])

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Correlate frequency lists from different powersweeps.")
    parser.add_argument('oldpath', nargs=1, type=str, help="Path to the old frequency list directory")
    parser.add_argument('newpath', nargs=1, type=str, help="Path to the new frequency list directory")
    parser.add_argument('oldpattern', nargs=1, type=str, help="Pattern that the feedline number letter combo follows "
                                                              "(e.g. '5_b' pattern would be '_', '5b' would be '')")
    parser.add_argument('newpattern', nargs=1, type=str, help="Pattern that the feedline number letter combo follows "
                                                              "(e.g. '5_b' pattern would be '_', '5b' would be '')")
    parser.add_argument('boardpattern', nargs=1, type=str, help="Pattern that prefaces board numbers in the new file" \
                                                               "names. Must be at least 2 characters. (e.g. 'psfreq_r222_fl7_a'" \
                                                               "would be '_r')")
    parser.add_argument('--fls', nargs='+', default=np.arange(1, 11, 1), type=int)
    parser.add_argument('--savepath','-s', nargs=1, default='.', type=str, help="Path to directory to save correlator lists to.")
    args = parser.parse_args()

    oldpath = args.oldpath[0]
    newpath = args.newpath[0]
    oldfiles = glob.glob(oldpath+'*')
    newfiles = glob.glob(newpath+'*')

    oldpattern = ["{}{}{}".format(i, args.oldpattern[0], j) for i in args.fls for j in ['a', 'b']]
    newpattern = ["{}{}{}".format(i, args.newpattern[0], j) for i in args.fls for j in ['a', 'b']]
    fls_to_search = np.array(newpattern) if (oldpattern == newpattern) else np.array(zip(oldpattern, newpattern))

    files_to_correlate = []
    if fls_to_search.shape[-1] == 2:
        for i, j in fls_to_search:
            temp=[]
            temp.append(j)
            for k in oldfiles:
                if i in k:
                    temp.append(k)
            for l in newfiles:
                if j in l:
                    temp.append(l)
            files_to_correlate.append(temp)
    else:
        for i in fls_to_search:
            temp = []
            temp.append(i)
            for j in oldfiles:
                if i in j:
                    temp.append(j)
            for k in newfiles:
                if i in k:
                    temp.append(k)
            files_to_correlate.append(temp)

    for i in files_to_correlate:
        print(i)
        if args.boardpattern[0] in i[2]:
            i.append(int(i[2].split(args.boardpattern[0])[1][:3]))

    correlator_objects = [Correlator(i[1], i[2], i[3], frequencyCutoff=500e3, fitOrder=1) for i in files_to_correlate]

    for i in range(len(files_to_correlate)):
        print("Starting correlating feedline {}, board {}".format(files_to_correlate[i][0], files_to_correlate[i][-1]))
        correlator_objects[i].correlateLists()
        correlator_objects[i].saveResult(path=args.savepath[0])
