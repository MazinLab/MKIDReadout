import numpy as np
import re
import os
import matplotlib.pyplot as plt
from mkidcore.corelog import getLogger
import mkidreadout.instruments as instruments

ISGOOD = 1
ISBAD = 0

MAX_ML_SCORE = 1
MAX_ATTEN = 100


LOCUT=1e9

def resID2fl(resID):
    return int(resID/10000)


def bandfor(lo):
    return 'a' if lo < 6.e9 else 'b'


class FreqSweep(object):
    def __init__(self, file):
        self.file = file
        data = np.load(self.file)
        self.atten = data['atten']  # 1d [nAttens] dB

        flip = np.diff(self.atten)[0] < 0

        self.freqs = data['freqs']  # 2d [nTones, nLOsteps] Hz

        self.i = data['I']  # 3d [nAttens, nTones, nLOsteps] ADC units
        self.q = data['Q']  # 3d [nAttens, nTones, nLOsteps] ADC units

        if flip:
            self.atten = self.atten[::-1]
            self.i = self.i[::-1, :, :]
            self.q = self.q[::-1, :, :]

        self.natten, self.ntone, self.nlostep = data['I'].shape
        self.freqStep = self.freqs[0,1] - self.freqs[0,0]
        
        sortedAttenInds = np.argsort(self.atten)
        self.atten = self.atten[sortedAttenInds]
        self.i = self.i[sortedAttenInds, :, :]
        self.q = self.q[sortedAttenInds, :, :]

    def oldwsformat_effective_atten(self, atten, amax=None):
        atten = np.abs(self.atten-atten).argmin()
        attenlast = atten+1 if amax is None else np.abs(self.atten - amax).argmin()
        return self.atten[atten:max(atten+1, attenlast)].mean()

    def oldwsformat(self, atten, amax=None):
        """Q vals are GARBAGE!!! use for magnitude only"""
        atten = np.abs(self.atten-atten).argmin()
        attenlast = atten+1 if amax is None else np.abs(self.atten - amax).argmin()
        attenlast = max(atten+1, attenlast)

        freqs = self.freqs.ravel().copy()
        iVals = self.i[atten:attenlast].squeeze()
        qVals = self.q[atten:attenlast].squeeze()
        if qVals.ndim > 2:
            msg = 'Averaging over {} powers ({}) to gen WS data'
            getLogger(__name__).info(msg.format(qVals.shape[0], self.atten[atten:attenlast]))
            iVals = iVals.mean(0)
            qVals = qVals.mean(0)
        iVals = iVals.ravel().copy()
        qVals = qVals.ravel().copy()

        mags = np.sqrt(iVals ** 2 + qVals ** 2)

        deltas = np.diff(freqs)
        boundaryInds = np.where(deltas < 0)[0]
        boundaryDeltas = -deltas[boundaryInds]
        nOverlapPoints = (boundaryDeltas / self.freqStep).astype(int) + 1
        boundaryInds = boundaryInds + 1

        for i in range(len(boundaryInds)):
            lfMags = mags[boundaryInds[i] - nOverlapPoints[i]: boundaryInds[i]]
            hfMags = mags[boundaryInds[i]: boundaryInds[i] + nOverlapPoints[i]]
            hfWeights = np.linspace(0, 1, num=nOverlapPoints[i], endpoint=False)
            lfWeights = 1 - hfWeights
            # set mags to average the overlap regions
            mags[boundaryInds[i]: boundaryInds[i] + nOverlapPoints[i]] = lfWeights * lfMags + hfWeights * hfMags
            mags[boundaryInds[i] - nOverlapPoints[i]: boundaryInds[i]] = np.nan  # set one side of overlap to 0
            freqs[boundaryInds[i] - nOverlapPoints[i]: boundaryInds[i]] = np.nan

        mags = mags[~np.isnan(mags)]
        freqs = freqs[~np.isnan(freqs)]

        iVals = mags
        qVals = np.zeros_like(iVals)
        return np.transpose([freqs, iVals, qVals])


class SweepMetadata(object):
    def __init__(self, resid=None, wsfreq=None, flag=None, mlfreq=None, atten=None,
                 ml_isgood_score=None, ml_isbad_score=None, file='',
                 wsatten=np.nan):

        self.file = file
        self.feedline = None

        self.resIDs = resid
        self.wsfreq = wsfreq
        self.flag = flag

        self.wsatten = wsatten

        if resid is not None:
            assert self.resIDs.size==self.wsfreq.size==self.flag.size

        self.atten = atten
        self.mlfreq = mlfreq
        self.ml_isgood_score = ml_isgood_score
        self.ml_isbad_score = ml_isbad_score

        if atten is None:
            self.atten = np.full_like(self.resIDs, np.nan, dtype=float)
        if mlfreq is None:
            self.mlfreq = np.full_like(self.resIDs, np.nan, dtype=float)
        if ml_isgood_score is None:
            self.ml_isgood_score = np.full_like(self.resIDs, np.nan, dtype=float)
        if ml_isbad_score is None:
            self.ml_isbad_score = np.full_like(self.resIDs, np.nan, dtype=float)

        if file and resid is None:
            self._load()

        assert (self.resIDs.size==self.wsfreq.size==self.flag.size==
                self.atten.size==self.mlfreq.size==self.ml_isgood_score.size==self.ml_isbad_score.size)

        self._settypes()
        self.sort()

    def __str__(self):
        return self.file

    @property
    def goodmlfreq(self):
        return self.mlfreq[self.flag == ISGOOD]

    @property
    def netscore(self):
        x=self.ml_isgood_score - self.ml_isbad_score
        x[np.isnan(x)] = 0
        return x

    def plot_scores(self):
        use = ~np.isnan(self.ml_isgood_score)
        plt.figure(20)
        plt.subplot(2,1,1)
        plt.hist(self.netscore[use], np.linspace(-1, 1, 100), label='netscore')
        plt.xlabel('goodscore-badscore')
        plt.subplot(2,1,2)
        plt.hist(self.ml_isbad_score[use], np.linspace(0, 1, 100), label='badscore')
        plt.hist(self.ml_isgood_score[use], np.linspace(0, 1, 100), alpha=.5, label='goodscore')
        plt.xlabel('ml scores')
        plt.legend()
        plt.show(False)

    def sort(self):
        s = np.argsort(self.resIDs)
        self.resIDs = self.resIDs[s]
        self.wsfreq = self.wsfreq[s]
        self.flag = self.flag[s]
        self.mlfreq = self.mlfreq[s]
        self.atten = self.atten[s]
        self.ml_isgood_score = self.ml_isgood_score[s]
        self.ml_isbad_score = self.ml_isbad_score[s]

    def toarray(self):
        return np.array([self.resIDs, self.flag, self.wsfreq, self.mlfreq, self.atten, self.ml_isgood_score,
                         self.ml_isbad_score])

    def update_from_roach(self, lo, freqs=None, attens=None):
        #TODO move these to user attens/frequencies
        use = self.lomask(lo, LOCUT)
        if attens is not None:
            self.attens[use] = attens
            assert use.sum() == attens.size
        if freqs is not None:
            self.mlfreq[use] = freqs
            assert use.sum() == freqs.size

    def lomask(self, lo, locut):
        return (self.flag == ISGOOD) & (~np.isnan(self.mlfreq)) & (np.abs(self.mlfreq - lo)<LOCUT)

    def vet(self):
        if (np.abs(self.atten[~np.isnan(self.atten)]) > MAX_ATTEN).any():
            getLogger(__name__).warning('odd attens')
        if (np.abs(self.ml_isgood_score[~np.isnan(self.ml_isgood_score)]) > MAX_ML_SCORE).any():
            getLogger(__name__).warning('bad ml good score')
        if (np.abs(self.ml_isbad_score[~np.isnan(self.ml_isbad_score)]) > MAX_ML_SCORE).any():
            getLogger(__name__).warning('bad ml bad scores')

        assert self.resIDs.size == np.unique(self.resIDs).size, "Resonator IDs must be unique."

        assert (self.resIDs.size==self.wsfreq.size==self.flag.size==
                self.atten.size==self.mlfreq.size==self.ml_isgood_score.size==
                self.ml_isbad_score.size)

    def genheader(self):
        header = ('feedline={}\n'
                  'wsatten={}\n'
                  'rID\trFlag\twsFreq\tmlFreq\tatten\tmlGood\tmlBad')
        return header.format(self.feedline, self.wsatten)

    def save(self, file=''):
        sf = file.format(feedline=self.feedline) if file else self.file.format(feedline=self.feedline)
        self.vet()
        np.savetxt(sf, self.toarray().T, fmt="%8d %1u %16.7f %16.7f %5.1f %6.4f %6.4f",
                   header=self.genheader())

    def templar_data(self, lo, locut=None):
        locut = LOCUT if locut is not None else np.inf
        aResMask = self.lomask(lo, locut)
        freq = self.mlfreq[aResMask]
        s = np.argsort(freq)
        #TODO should this be moved to VET?
        assert freq.size == np.unique(freq).size, "Frequencies for templar must be be unique."
        return self.resIDs[aResMask][s], freq[s], self.atten[aResMask][s]

    def save_templar_freqfiles(self, lo, template='ps_freqs{roach}_FL{feedline}{band}.txt'):
        band = bandfor(lo)
        aFile = os.path.join(os.path.dirname(self.file),
                             template.format(roach=instruments.roachnum(self.feedline, band),
                                             feedline=self.feedline, band=band))
        np.savetxt(aFile, np.transpose(self.templar_data(lo)), fmt='%4i %10.9e %.2f')

    def _settypes(self):
        self.flag = self.flag.astype(int)
        self.resIDs = self.resIDs.astype(int)
        self.feedline = int(self.resIDs[0]/10000)

    def _load(self):
        d = np.loadtxt(self.file.format(feedline=self.feedline), unpack=True)
        #TODO convert to load metadata from file
        self.resIDs, self.flag, self.wsfreq, self.mlfreq, self.atten, self.ml_isgood_score, self.ml_isbad_score = d

        self.mlfreq[self.flag == ISBAD] = self.wsfreq[self.flag == ISBAD]
        self.ml_isgood_score[self.flag == ISBAD] = 0
        self.ml_isbad_score[self.flag == ISBAD] = 1
        self._settypes()


class FreqFile(object):
    def __init__(self, file, feedline=None, band=None, resids=None, freq=None, atten=None):

        self.file = file
        self.feedline = feedline
        self.band = band
        self.file = 'ps_freqs{roach}_FL{feedline}{band}.txt'

        if resids is None:
            self.load()

        self.resIDs, self.freqs, self.attens = resids, freq, atten
        self._coerce()

    def _coerce(self):
        assert self.resIDs.size == self.freqs.size == self.attens.size, "Frequencies in " + self.file + " must be unique."
        assert np.unique(self.resIDs).size == self.resIDs.size, 'Resonator IDs in ' + self.file + " must be unique."
        assert resID2fl(self.resIDs[0]) == self.feedline, 'Resonator ID/ feedline mismatch'
        s = self.freq.argsort()
        self.resIDs = self.resIDs[s]
        self.freqs = self.freqs[s]
        self.attens = self.attens[s]

    def save(self):
        np.savetxt(self.file, np.transpose([self.resIDs, self.freqs, self.attens]), fmt='%4i %10.9e %.2f')

    def load(self):
        self.resIDs, self.freqs, self.attens = np.loadtxt(self.file, unpack=True)
        self._coerce()


def loadold(allfile, goodfile, outfile='digWS_FL{feedline}_metadata.txt'):
    gid, gndx, gfreq = np.loadtxt(goodfile, unpack=True)
    aid, andx, afreq = np.loadtxt(allfile, unpack=True)

    flags = np.full(aid.size, ISBAD)

    assert np.unique(gid).size == gid.size and np.unique(aid).size == aid.size

    badids = np.setdiff1d(aid, gid)
    bad = np.isin(aid, badids)

    flags[~bad] = ISGOOD

    return SweepMetadata(resid=aid, wsfreq=afreq, flag=flags, file=outfile)
