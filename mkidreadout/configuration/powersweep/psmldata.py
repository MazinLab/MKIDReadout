import matplotlib.pyplot as plt
import numpy as np

from mkidcore.corelog import getLogger
from mkidcore.sweepdata import FreqSweep, ISGOOD, SweepMetadata


class MLData(object):
    def __init__(self, fsweep=None, mdata=None):

        self.freqSweep = fsweep if isinstance(fsweep, FreqSweep) else FreqSweep(fsweep)
        self.metadata = mdata if isinstance(mdata, SweepMetadata) else SweepMetadata(file=mdata)
        
        #to define: self.freqs, self.Is, self.Qs, self.attens, self.resIDs
        #   self.opt_attens, self.opt_freqs, self.scores

        freqSpan = np.array([self.freqSweep.freqs[0,0], self.freqSweep.freqs[-1,-1]])
        self.mdResMask = ((self.metadata.flag & ISGOOD) &
                          (self.metadata.wsfreq > freqSpan[0]) &
                          (self.metadata.wsfreq < freqSpan[1])).astype(bool)
        self.nRes = self.mdResMask.sum()
        if not self.nRes:
            raise RuntimeError('No rood resonators with freqs within sweep bounds')
        self.freqs = np.zeros((self.nRes, self.freqSweep.nlostep))
        self.Is = np.zeros((self.nRes, self.freqSweep.natten, self.freqSweep.nlostep))
        self.Qs = np.zeros((self.nRes, self.freqSweep.natten, self.freqSweep.nlostep))
        self.iq_vels = np.zeros((self.nRes, self.freqSweep.natten, self.freqSweep.nlostep-1))
        self.attens = self.freqSweep.atten

        self.resIDs = self.metadata.resIDs[self.mdResMask]
        self.initfreqs = self.metadata.wsfreq[self.mdResMask]
        self.opt_attens = self.metadata.mlatten[self.mdResMask] # maybe change to freq and atten?
        self.opt_freqs = self.metadata.mlfreq[self.mdResMask]
        self.scores = np.zeros(self.nRes)
        self.bad_scores = np.zeros(self.nRes)
        if not np.all(np.isnan(self.opt_attens)):
            attenblock = np.tile(self.freqSweep.atten, (len(self.opt_attens),1))
            self.opt_iAttens = np.argmin(np.abs(attenblock.T - self.opt_attens), axis=0)

        self.generateMLWindows(self.initfreqs)

    def generateMLWindows(self, resFreqs):
        assert len(resFreqs) == self.nRes, 'Mismatch between provided freq list and number of res in metadata file'
        winCenters = self.freqSweep.freqs[:, self.freqSweep.nlostep/2]
        for i in range(self.nRes):
            freqWinInd = np.argmin(np.abs(resFreqs[i] - winCenters))
            self.freqs[i] = self.freqSweep.freqs[freqWinInd, :]
            self.Is[i] = self.freqSweep.i[:, freqWinInd, :]
            self.Qs[i] = self.freqSweep.q[:, freqWinInd, :]
            self.iq_vels[i] = np.sqrt(np.diff(self.Is[i])**2 + np.diff(self.Qs[i])**2)

    def updatemetadata(self):
        self.metadata.mlatten[self.mdResMask] = self.opt_attens
        self.metadata.mlfreq[self.mdResMask] = self.opt_freqs
        self.metadata.ml_isgood_score[self.mdResMask] = self.scores
        self.metadata.ml_isbad_score[self.mdResMask] = self.bad_scores

    def prioritize_and_cut(self, assume_bad_cut=-np.inf, assume_good_cut=np.inf, plot=False):
        #return self.mdResMask.sum()

        if plot:
            self.metadata.plot_scores()
            plt.axvline(max(assume_bad_cut, -1), color='k', linewidth=.5)
            plt.axvline(max(assume_good_cut, 1), color='k', linewidth=.5)

        netscore = self.metadata.netscore
        badcutmask = netscore < assume_bad_cut
        goodcutmask = netscore > assume_good_cut
        self.metadata.mlatten[badcutmask] = np.inf

        getLogger(__name__).info('Sorting {} resonators'.format(self.mdResMask.sum()))
        msg = 'Bad score cut of {:.2f} kills {} resonators'
        getLogger(__name__).info(msg.format(assume_bad_cut, (badcutmask & self.mdResMask).sum()))

        msg = 'Good score cut of {:.2f} accepts {} resonators'
        getLogger(__name__).info(msg.format(assume_good_cut, (goodcutmask & self.mdResMask).sum()))

        reviewmask = ~badcutmask & ~goodcutmask

        selectmask = reviewmask & self.mdResMask

        stop_ndx = selectmask.sum()

        getLogger(__name__).info('There are {} uncut, sorted resonators.'.format(stop_ndx))

        order = np.argsort(netscore[selectmask])

        self.initfreqs = self.initfreqs[reviewmask[self.mdResMask]][order]
        self.opt_attens = self.opt_attens[reviewmask[self.mdResMask]][order]
        self.opt_freqs = self.opt_freqs[reviewmask[self.mdResMask]][order]
        self.resIDs = self.resIDs[reviewmask[self.mdResMask]][order]
        self.freqs = self.freqs[reviewmask[self.mdResMask]][order]
        self.Is = self.Is[reviewmask[self.mdResMask]][order]
        self.Qs = self.Qs[reviewmask[self.mdResMask]][order]

        return stop_ndx





