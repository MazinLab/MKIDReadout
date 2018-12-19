""" Implements a template filter to identify WS peaks """
import numpy as np
import scipy.signal as signal
import mkidreadout.instruments as instrument
from mkidcore.corelog import getLogger
from mkidreadout.configuration.widesweep.wsdata import WSFitMLData
import mkidreadout.configuration.sweepdata as sweepdata
import os
import argparse
import re


class Finder(object):
    def __init__(self, inferenceFileAB):
        self.inferenceData = WSFitMLData(inferenceFileAB)
        self.spacing = self.inferenceData.freqStep/1.e3 #convert to kHz

    def inferPeaks(self,  sigThresh=0.5):
        bpFilt = signal.firwin(1001, (0.7*0.005*12.5/7.6*self.spacing/7.63, 0.175*self.spacing/7.63), pass_zero=False, window=('chebwin', 100))
        firFiltMagsDB = np.convolve(self.inferenceData.magsdb, bpFilt, mode='same')

        thresh = sigThresh*np.std(firFiltMagsDB)
        print 'threshold', thresh
        peaks, _ = signal.find_peaks(-firFiltMagsDB, prominence=thresh, width=2, wlen=int(267/self.spacing))
        morePeaks, _ = signal.find_peaks(-firFiltMagsDB, thresh, prominence=0.4*thresh, width=2, wlen=int(229/self.spacing))
        peaks = np.append(peaks, morePeaks)
        peaks = np.unique(peaks)
        peaks = np.sort(peaks)
        print 'sp signal found', len(peaks), 'peaks'

        self.peakIndices = peaks

    def markCollisions(self, resBWkHz=500):
        if self.peakIndices is None:
            raise Exception('Infer peak locations first!')
        minResSpacing = resBWkHz/self.spacing #resonators must be separated by this number of points
        peakSpacing = np.diff(self.peakIndices)
        collisionMask = peakSpacing<minResSpacing
        collisionInds = np.where(collisionMask)[0] #locations in peakIndices where there are collisions
        goodPeakInds = np.where(np.logical_not(collisionMask))[0]
        self.badPeakIndices = self.peakIndices[collisionInds]
        self.goodPeakIndices = self.peakIndices[goodPeakInds]

    @property
    def num_good(self):
        return self.goodPeakIndices.size

    def findLocalMinima(self):
        if self.peakIndices is None:
            raise Exception('Infer peak locations first!')
        foundMinima = np.zeros(len(self.peakIndices))
        # print (len(foundMinima))
        peakVals = self.inferenceData.magsdb
        while np.any(foundMinima==0):
            peakValsRight = np.roll(peakVals, -1)
            peakValsLeft = np.roll(peakVals, 1)
            peakValsRightLess = np.less_equal(peakVals[self.peakIndices], peakValsRight[self.peakIndices])
            peakValsLeftLess = np.less_equal(peakVals[self.peakIndices], peakValsLeft[self.peakIndices])
            foundMinima = np.logical_and(peakValsLeftLess, peakValsRightLess)
            
            peakValsRightGreater = np.logical_not(peakValsRightLess)
            peakValsLeftGreater = np.logical_and(peakValsRightLess, np.logical_not(foundMinima)) #not greater, but not a minimum
            peakValsRightGreaterInd = np.where(peakValsRightGreater)[0]
            peakValsLeftGreaterInd = np.where(peakValsLeftGreater)[0]

            self.peakIndices[peakValsRightGreaterInd] += 1
            self.peakIndices[peakValsLeftGreaterInd] -= 1
            # print sum(foundMinima)

        self.peakIndices = np.unique(self.peakIndices)

    def getSweepMetadata(self):

        try:
            flNum = int(re.search('fl\d', self.inferenceData.filenameList[0], re.IGNORECASE).group()[-1])
        except AttributeError:
            try:

                ip = int(os.path.splitext(self.inferenceData.filenameList[0])[0][-3:])
                flNum = int(instrument.MEC_NUM_FL_MAP[ip][0])
            except (KeyError, ValueError, IndexError):
                getLogger(__name__).warning('Could not guess feedline from filename {}.')
                flNum = 0

        metadatafile = os.path.splitext(self.inferenceData.filenameList[0])[0] + '_metadata.txt'

        ws_good_inds = self.goodPeakIndices
        ws_bad_inds = self.badPeakIndices
        freqs = np.append(self.inferenceData.freqs[ws_good_inds], self.inferenceData.freqs[ws_bad_inds])
        sort_inds = np.argsort(freqs)
        resIds = np.arange(freqs.size) + flNum * 10000

        flag = np.full(freqs.size, sweepdata.ISBAD)
        flag[:ws_good_inds.size] = sweepdata.ISGOOD
        smd = sweepdata.SweepMetadata(resid=resIds, flag=flag[sort_inds], wsfreq=freqs[sort_inds], file=metadatafile,
                                      wsatten=self.inferenceData.effective_atten)
        return smd

