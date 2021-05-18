""" Implements a template filter to identify WS peaks """
import numpy as np
import scipy.signal as signal

import mkidcore.sweepdata as sweepdata
from mkidcore.corelog import getLogger
from mkidreadout.configuration.widesweep.wsdata import WSFitMLData


class Finder(object):
    def __init__(self, inferenceFileAB):
        self.inferenceData = WSFitMLData(inferenceFileAB)
        self.spacing = self.inferenceData.freqStep/1.e3 #convert to kHz

    def inferPeaks(self,  sigThresh=0.5):
        bpFilt = signal.firwin(1001, (0.7*0.005*12.5/7.6*self.spacing/7.63, 0.175*self.spacing/7.63), pass_zero=False, window=('chebwin', 100))
        firFiltMagsDB = np.convolve(self.inferenceData.magsdb, bpFilt, mode='same')

        thresh = sigThresh*np.std(firFiltMagsDB)
        peaks, _ = signal.find_peaks(-firFiltMagsDB, prominence=thresh, width=2, wlen=int(267/self.spacing))
        morePeaks, _ = signal.find_peaks(-firFiltMagsDB, thresh, prominence=0.4*thresh, width=2, wlen=int(229/self.spacing))
        peaks = np.append(peaks, morePeaks)
        peaks = np.unique(peaks)
        peaks = np.sort(peaks)
        getLogger(__name__).info('Found {} peaks'.format(len(peaks)))
        self.peakIndices = peaks

    def markCollisions(self, resBWkHz=500):
        if self.peakIndices is None:
            raise RuntimeError('Infer peak locations first!')
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
            raise RuntimeError('Infer peak locations first!')
        foundMinima = np.zeros(len(self.peakIndices))
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

        self.peakIndices = np.unique(self.peakIndices)

    def getSweepMetadata(self, flnum):

        ws_good_inds = self.goodPeakIndices
        ws_bad_inds = self.badPeakIndices
        freqs = np.append(self.inferenceData.freqs[ws_good_inds], self.inferenceData.freqs[ws_bad_inds])
        sort_inds = np.argsort(freqs)

        resIds = sweepdata.genResIDsForFreqs(freqs, flnum)

        flag = np.full(freqs.size, sweepdata.ISBAD)
        flag[:ws_good_inds.size] = sweepdata.ISGOOD
        smd = sweepdata.SweepMetadata(resid=resIds, flag=flag[sort_inds], wsfreq=freqs[sort_inds],
                                      wsatten=self.inferenceData.effective_atten)
        return smd

