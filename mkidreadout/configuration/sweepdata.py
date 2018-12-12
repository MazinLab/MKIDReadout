import numpy as np


ISGOOD = 1
ISBAD = 0

class SweepMetadata(object):
    def __init__(self, feedline, resid=None, wsfreq=None, flag=None, mlfreq=None, atten=None,
                 ml_isgood_score=None, ml_isbad_score=None, file=''):

        self.file = file
        self.feedline = feedline

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
            self.atten = np.full_like(self.resIDs, np.nan)
        if mlfreq is None:
            self.mlfreq = np.full_like(self.resIDs, np.nan)
        if ml_isgood_score is None:
            self.ml_isgood_score = np.full_like(self.resIDs, np.nan)
        if ml_isbad_score is None:
            self.ml_isbad_score = np.full_like(self.resIDs, np.nan)

        if file and resid is None:
            self._load()

        assert (self.resIDs.size==self.wsfreq.size==self.flag.size==
                self.atten.size==self.mlfreq.size==self.ml_isgood_score.size==self.ml_isbad_score.size)

        self.sort()

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
        sf = file if file else self.file
        np.savetxt(sf, self.toarray().T, fmt="%8d %1u %16.7f %16.7f %5.1f %6.4f %6.4f",
                   header='feedline={}\nrID\trFlag\twsFreq\tmlFreq\tatten\tmlGood\tmlBad'.format(self.feedline))

    def _load(self):
        d = np.loadtxt(self.file, unpack=True)
        self.resIDs, self.flag, self.wsfreq, self.mlfreq, self.atten, self.ml_isgood_score, self.ml_isbad_score = d
        self.flag = self.flag.astype(int)
        self.resIDs = self.resIDs.astype(int)