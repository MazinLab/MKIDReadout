import numpy as np
import matplotlib.pyplot as plt
from mkidcore.corelog import getLogger


ISGOOD = 1
ISBAD = 0


class FreqSweep(object):
    def __init__(self, file):
        self.file = file
        data = np.load(self.file)
        self.atten = data['atten']  # 1d [nAttens] dB

        flip = np.diff(self.atten)[0] < 0

        self.freqs = data['freqs']  # 2d [nTones, nLOsteps] Hz

        self.i = data['I']  # 3d [nAttens, nTones, nLOsteps] ADC units
        self.q = data['Q']  # 3d [nAttens, nTones, nLOsteps] ADC units

        if '222' in self.file or'233' in self.file or '236' in self.file:
            self.atten = self.atten[::2]
            self.i = self.i[::2, :, :]
            self.q = self.q[::2, :, :]

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
                 ml_isgood_score=None, ml_isbad_score=None, file=''):

        self.file = file
        self.feedline = None

        self.resIDs = resid
        self.wsfreq = wsfreq
        self.flag = flag

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

    def save(self, file=''):
        sf = file.format(feedline=self.feedline) if file else self.file.format(feedline=self.feedline)

        if (np.abs(self.atten[~np.isnan(self.atten)])>100).any():
            getLogger(__name__).warning('odd attens')
        if (np.abs(self.ml_isgood_score[~np.isnan(self.ml_isgood_score)])>1).any():
            getLogger(__name__).warning('bad ml good score')
        if (np.abs(self.ml_isbad_score[~np.isnan(self.ml_isbad_score)]) > 1).any():
            getLogger(__name__).warning('bad ml bad scores')

        np.savetxt(sf, self.toarray().T, fmt="%8d %1u %16.7f %16.7f %5.1f %6.4f %6.4f",
                   header='feedline={}\nrID\trFlag\twsFreq\tmlFreq\tatten\tmlGood\tmlBad'.format(self.feedline))

    def _settypes(self):
        self.flag = self.flag.astype(int)
        self.resIDs = self.resIDs.astype(int)
        self.feedline = int(self.resIDs[0]/10000)

    def _load(self):
        d = np.loadtxt(self.file.format(feedline=self.feedline), unpack=True)
        self.resIDs, self.flag, self.wsfreq, self.mlfreq, self.atten, self.ml_isgood_score, self.ml_isbad_score = d

        self.mlfreq[self.flag == ISBAD] = self.wsfreq[self.flag == ISBAD]
        self.ml_isgood_score[self.flag == ISBAD] = 0
        self.ml_isbad_score[self.flag == ISBAD] = 1
        self._settypes()


def loadold(allfile, goodfile, outfile='digWS_FL{feedline}_metadata.txt'):
    gid, gndx, gfreq = np.loadtxt(goodfile, unpack=True)
    aid, andx, afreq = np.loadtxt(allfile, unpack=True)

    flags = np.full(aid.size, ISBAD)

    assert np.unique(gid).size == gid.size and np.unique(aid).size == aid.size

    badids = np.setdiff1d(aid, gid)
    bad = np.isin(aid, badids)

    flags[~bad] = ISGOOD

    return SweepMetadata(resid=aid, wsfreq=afreq, flag=flags, file=outfile)
