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
        self.freqs = data['freqs']  # 2d [nTones, nLOsteps] Hz
        self.i = data['I']  # 3d [nAttens, nTones, nLOsteps] ADC units
        self.q = data['Q']  # 3d [nAttens, nTones, nLOsteps] ADC units
        self.natten, self.ntone, self.nlostep = data['I'].shape
        self.freqStep = self.freqs[0,1] - self.freqs[0,0]

    def oldwsformat(self, atten):
        """Q vals are GARBAGE!!! use for magnitude only"""
        atten = np.abs(self.atten-atten).argmin()
        freqs = self.freqs.ravel()
        iVals = self.i[atten].ravel()
        qVals = self.q[atten].ravel()

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
        return self.ml_isgood_score - self.ml_isbad_score

    def plot_scores(self):
        plt.figure(20)
        plt.hist(self.netscore, np.linspace(-1, 1, 50), label='netscore')
        plt.show(True)

    def sort(self):
        s = np.argsort(self.wsfreq)
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
        self._settypes()

    # def savetemplarfile(self, file):
    #     """ resid  freq  atten"""
    #     u = self.flag == ISGOOD
    #     #see MKIDReadout/mkidreadout/configuration/findLOsAndMakeFreqLists.py and make support this file
    #     np.savetxt(file, np.array([self.resIDs[u], self.mlfreq[u], self.atten[u]]).T, fmt="%8d %16.7f %5.0f")


def loadold(allfile, goodfile, outfile='digWS_FL{feedline}_metadata.txt'):
    gid, gndx, gfreq = np.loadtxt(goodfile, unpack=True)
    aid, andx, afreq = np.loadtxt(allfile, unpack=True)

    flags = np.full(aid.size, ISBAD)

    assert np.unique(gid).size == gid.size and np.unique(aid).size == aid.size

    badids = np.setdiff1d(aid, gid)
    bad = np.isin(aid, badids)

    flags[~bad] = ISGOOD

    return SweepMetadata(resid=aid, wsfreq=afreq, flag=flags, file=outfile)
