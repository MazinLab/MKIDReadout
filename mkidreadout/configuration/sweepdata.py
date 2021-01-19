import matplotlib.pyplot as plt
import numpy as np

from mkidcore.corelog import getLogger
try:
    import glob, parse
except ImportError:
    getLogger(__name__).warning('Could not find glob/parse modules')

#from mkidcore.objects import Beammap
from mkidcore.pixelflags import beammap as beamMapFlags

# flags
ISGOOD = 0b1
ISREVIEWED = 0b10
ISBAD = 0

MAX_ML_SCORE = 1
MAX_ATTEN = 100

LOCUT = 1e9

A_RANGE_CUTOFF = 6e9


def genResIDsForFreqs(freqs, flnum):
    # TODO where does this live
    return np.arange(freqs.size) + flnum * 10000 + (freqs > A_RANGE_CUTOFF) * 5000


#This function is already implemented in beammap/utils.py
def resID2fl(resID):
    return int(resID / 10000)


def bandfor(lo):
    return 'a' if lo < A_RANGE_CUTOFF else 'b'


class FreqSweep(object):
    def __init__(self, file):
        self.file = file
        data = np.load(self.file)
        self.atten = data['atten']  # 1d [nAttens] dB

        if len(self.atten) > 1:
            flip = np.diff(self.atten)[0] < 0
        else:
            flip = False

        self.freqs = data['freqs']  # 2d [nTones, nLOsteps] Hz

        if len(self.atten) == 1 and data['I'].ndim == 2:
            self.i = np.expand_dims(data['I'], 0)
            self.q = np.expand_dims(data['Q'], 0)

        else:
            self.i = data['I']  # 3d [nAttens, nTones, nLOsteps] ADC units
            self.q = data['Q']  # 3d [nAttens, nTones, nLOsteps] ADC units

        if flip:
            self.atten = self.atten[::-1]
            self.i = self.i[::-1, :, :]
            self.q = self.q[::-1, :, :]

        self.natten, self.ntone, self.nlostep = self.i.shape #data['I'].shape
        self.freqStep = self.freqs[0, 1] - self.freqs[0, 0]

        sortedAttenInds = np.argsort(self.atten)
        self.atten = self.atten[sortedAttenInds]
        self.i = self.i[sortedAttenInds, :, :]
        self.q = self.q[sortedAttenInds, :, :]
        try:
            self.lostart = data['loStart']
            self.loend = data['loEnd']
            self.lo = (self.lostart + self.loend)/2
        except KeyError:
            pass

    def oldwsformat_effective_atten(self, atten, amax=None):
        atten = np.abs(self.atten - atten).argmin()
        attenlast = atten + 1 if amax is None else np.abs(self.atten - amax).argmin()
        return self.atten[atten:max(atten + 1, attenlast)].mean()

    def oldwsformat(self, atten, amax=None):
        """Q vals are GARBAGE!!! use for magnitude only"""
        atten = np.abs(self.atten - atten).argmin()
        attenlast = atten + 1 if amax is None else np.abs(self.atten - amax).argmin()
        attenlast = max(atten + 1, attenlast)

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

    @property
    def iqvel(self):
        if not hasattr(self, '_iqvel'):
            self._iqvel = np.sqrt(np.diff(self.i, axis=2)**2 + np.diff(self.q, axis=2)**2)

        return self._iqvel


class SweepMetadata(object):
    def __init__(self, resid=None, wsfreq=None, flag=None, mlfreq=None, mlatten=None,
                 ml_isgood_score=None, ml_isbad_score=None, file='',
                 wsatten=np.nan, phases=None, iqRatios=None):

        #TODO add channel, range (a|b)
        self.file = file
        self.feedline = None

        self.resIDs = resid
        self.wsfreq = wsfreq
        self.flag = flag
        self.wsatten = wsatten

        self.mlatten = mlatten
        self.mlfreq = mlfreq
        self.ml_isgood_score = ml_isgood_score
        self.ml_isbad_score = ml_isbad_score

        self.phases = phases
        self.iqRatios = iqRatios

        if mlatten is None:
            self.mlatten = np.full_like(self.resIDs, np.nan, dtype=float)
        if mlfreq is None:
            self.mlfreq = np.full_like(self.resIDs, np.nan, dtype=float)
        if ml_isgood_score is None:
            self.ml_isgood_score = np.full_like(self.resIDs, np.nan, dtype=float)
        if ml_isbad_score is None:
            self.ml_isbad_score = np.full_like(self.resIDs, np.nan, dtype=float)
        if phases is None:
            self.phases = np.full_like(self.resIDs, np.nan, dtype=float)
        if iqRatios is None:
            self.iqRatios = np.full_like(self.resIDs, np.nan, dtype=float) 

        self.freq = self.mlfreq.copy()
        self.atten = self.mlatten.copy()

        if file and resid is None:
            self._load()

        self._vet()
        #self.sort()

    def __str__(self):
        return self.file

    @property
    def goodmlfreq(self):
        return self.mlfreq[self.flag & ISGOOD]

    @property
    def netscore(self):
        x = self.ml_isgood_score - self.ml_isbad_score
        x[np.isnan(x)] = 0
        return x

    def plot_scores(self):
        use = ~np.isnan(self.ml_isgood_score)
        plt.figure(20)
        plt.subplot(2, 1, 1)
        plt.hist(self.netscore[use], np.linspace(-1, 1, 100), label='netscore')
        plt.xlabel('goodscore-badscore')
        plt.subplot(2, 1, 2)
        plt.hist(self.ml_isbad_score[use], np.linspace(0, 1, 100), label='badscore')
        plt.hist(self.ml_isgood_score[use], np.linspace(0, 1, 100), alpha=.5, label='goodscore')
        plt.xlabel('ml scores')
        plt.legend()
        plt.show(False)

    def set(self, resID, atten=None, freq=None, save=False, reviewed=False):
        mask = resID == self.resIDs
        if not mask.any():
            getLogger(__name__).warning('Unable to set values for unknown resID: {}'.format(resID))
            return False
        if atten is not None:
            self.atten[mask] = atten
        if freq is not None:
            self.freq[mask] = freq
        if reviewed:
            self.flag[mask] |= ISREVIEWED
        if save:
            self.save()
        return True

    def sort(self):
        s = np.argsort(self.resIDs)
        self.resIDs = self.resIDs[s]
        self.wsfreq = self.wsfreq[s]
        self.flag = self.flag[s]
        self.mlfreq = self.mlfreq[s]
        self.mlatten = self.mlatten[s]
        self.atten = self.atten[s]
        self.ml_isgood_score = self.ml_isgood_score[s]
        self.ml_isbad_score = self.ml_isbad_score[s]
        self.freq = self.freq[s]

    def toarray(self):
        return np.array([self.resIDs, self.flag, self.wsfreq, self.mlfreq, self.mlatten, self.freq,
                         self.atten, self.ml_isgood_score, self.ml_isbad_score, self.phases, self.iqRatios])

    def update_from_roach(self, resIDs, freqs=None, attens=None):
        if attens is not None:
            assert resIDs.size == attens.size
        if freqs is not None:
            assert resIDs.size == freqs.size
        for i, r in enumerate(resIDs):
            #TODO vectorize!
            use = self.resIDs == r
            if attens is not None:
                self.atten[use] = attens[i]
            if freqs is not None:
                self.freq[use] = freqs[i]

    def lomask(self, lo):
        return ((self.flag & ISGOOD) & (~np.isnan(self.freq)) & (np.abs(self.freq - lo) < LOCUT) & (self.atten > 0)).astype(bool)

    def vet(self):
        if (np.abs(self.atten[~np.isnan(self.atten)]) > MAX_ATTEN).any():
            getLogger(__name__).warning('odd attens')
        if (np.abs(self.ml_isgood_score[~np.isnan(self.ml_isgood_score)]) > MAX_ML_SCORE).any():
            getLogger(__name__).warning('bad ml good score')
        if (np.abs(self.ml_isbad_score[~np.isnan(self.ml_isbad_score)]) > MAX_ML_SCORE).any():
            getLogger(__name__).warning('bad ml bad scores')

        assert self.resIDs.size == np.unique(self.resIDs).size, "Resonator IDs must be unique."

        assert (self.resIDs.size == self.wsfreq.size == self.flag.size ==
                self.atten.size == self.mlfreq.size == self.ml_isgood_score.size ==
                self.ml_isbad_score.size)

    def genheader(self, useSBSup=False):
        if useSBSup:
            header = ('feedline={}\n'
                      'wsatten={}\n'
                      'rID\trFlag\twsFreq\tmlFreq\tmlatten\tfreq\tatten\tmlGood\tmlBad\tphases\tiqratios')
        else:
            header = ('feedline={}\n'
                      'wsatten={}\n'
                      'rID\trFlag\twsFreq\tmlFreq\tmlatten\tfreq\tatten\tmlGood\tmlBad')
        return header.format(self.feedline, self.wsatten)

    def save(self, file='', saveSBSupData=False):
        sf = file.format(feedline=self.feedline) if file else self.file.format(feedline=self.feedline)
        self.vet()
        if saveSBSupData:
            np.savetxt(sf, self.toarray().T, fmt="%8d %1u %16.7f %16.7f %5.1f %16.7f %5.1f %6.4f %6.4f %6.4f %6.4f",
                       header=self.genheader(True))
        else:
            np.savetxt(sf, self.toarray().T[:, :-2], fmt="%8d %1u %16.7f %16.7f %5.1f %16.7f %5.1f %6.4f %6.4f",
                   header=self.genheader(False))

    def templar_data(self, lo):
        aResMask = self.lomask(lo)  #TODO URGENT add range assignment to each resonator
        freq = self.freq[aResMask]
        # Do not sort or force things to be unique, doing so would break the implicity channel order
        return self.resIDs[aResMask], freq, self.atten[aResMask], self.phases[aResMask], self.iqRatios[aResMask]

    def legacy_save(self, file=''):
        sf = file.format(feedline=self.feedline) if file else self.file.format(feedline=self.feedline)
        np.savetxt(sf, np.transpose(self.templar_data(0)[:3]), fmt='%4i %10.9e %.2f')  #TODO remove 0 after sorting out
        # lo issue

    def _vet(self):

        assert (self.resIDs.size == self.wsfreq.size == self.flag.size == self.atten.size == self.freq.size ==
                self.mlatten.size == self.mlfreq.size == self.ml_isgood_score.size == self.ml_isbad_score.size)

        self.flag = self.flag.astype(int)
        self.resIDs = self.resIDs.astype(int)

        for x in (self.freq, self.mlfreq, self.wsfreq):
           use = (~np.isnan(x)) & ((self.flag & ISGOOD) == ISGOOD)
           if x[use].size != np.unique(x[use]).size:
               getLogger(__name__).warning("Found non-unique frequencies")

        self.feedline = resID2fl(self.resIDs[0])

    def _load(self):
        d = np.loadtxt(self.file.format(feedline=self.feedline), unpack=True)
        if d.ndim == 1: #allows files with single res
            d = np.expand_dims(d, axis=1)
        # TODO convert to load metadata from file
        try:
            if d.shape[0] == 11:
                self.resIDs, self.flag, self.wsfreq, self.mlfreq, self.mlatten, \
                self.freq, self.atten, self.ml_isgood_score, self.ml_isbad_score, self.phases, self.iqRatios = d
            elif d.shape[0] == 9:
                self.resIDs, self.flag, self.wsfreq, self.mlfreq, self.mlatten, \
                self.freq, self.atten, self.ml_isgood_score, self.ml_isbad_score = d
                self.phases = np.full_like(self.resIDs, 0, dtype=float)
                self.iqRatios = np.full_like(self.resIDs, 1, dtype=float)
            elif d.shape[0] == 7:
                self.resIDs, self.flag, self.wsfreq, self.mlfreq, self.mlatten, \
                self.ml_isgood_score, self.ml_isbad_score = d
                self.freq = self.mlfreq.copy()
                self.atten = self.mlatten.copy()
                self.phases = np.full_like(self.resIDs, 0, dtype=float)
                self.iqRatios = np.full_like(self.resIDs, 1, dtype=float)
            elif d.shape[0] == 5:
                self.resIDs, self.freq, self.atten, self.phases, self.iqRatios = d
                #_, u = np.unique(self.freq, return_index=True)
                #self.resIDs = self.resIDs[u]
                #self.freq = self.freq[u]
                #self.atten = self.atten[u]
                self.wsfreq = self.freq.copy()
                self.mlfreq = self.freq.copy()
                self.mlatten = self.atten.copy()
                self.flag = np.full_like(self.resIDs, ISGOOD, dtype=int)
                self.ml_isgood_score = np.full_like(self.resIDs, np.nan, dtype=float)
                self.ml_isbad_score = np.full_like(self.resIDs, np.nan, dtype=float)
            else:
                self.resIDs, self.freq, self.atten = d
                #_, u = np.unique(self.freq, return_index=True)
                #self.resIDs = self.resIDs[u]
                #self.freq = self.freq[u]
                #self.atten = self.atten[u]
                self.wsfreq = self.freq.copy()
                self.mlfreq = self.freq.copy()
                self.mlatten = self.atten.copy()
                self.flag = np.full_like(self.resIDs, ISGOOD, dtype=int)
                self.ml_isgood_score = np.full_like(self.resIDs, np.nan, dtype=float)
                self.ml_isbad_score = np.full_like(self.resIDs, np.nan, dtype=float)
                self.phases = np.full_like(self.resIDs, 0, dtype=float)
                self.iqRatios = np.full_like(self.resIDs, 1, dtype=float)
        except:
            raise ValueError('Unknown number of columns')

        self.freq[np.isnan(self.freq)] = self.mlfreq[np.isnan(self.freq)]
        self.freq[np.isnan(self.freq)] = self.wsfreq[np.isnan(self.freq)]
        self.atten[np.isnan(self.atten)] = self.mlatten[np.isnan(self.atten)]

        self.flag = self.flag.astype(int)
        self.mlfreq[self.flag & ISBAD] = self.wsfreq[self.flag & ISBAD]
        self.ml_isgood_score[self.flag & ISBAD] = 0
        self.ml_isbad_score[self.flag & ISBAD] = 1
        self._vet()
        
    def powerDownUnbeammappedRes(self, beammap):
        badResIDs = beammap.resIDs[beammap.flags != beamMapFlags['good']]
        for i,r in enumerate(self.resIDs):
            if np.any(r == badResIDs):
                self.atten[i] = 99



def loadold(allfile, goodfile, outfile='digWS_FL{feedline}_metadata.txt'):
    gid, gndx, gfreq = np.loadtxt(goodfile, unpack=True)
    aid, andx, afreq = np.loadtxt(allfile, unpack=True)

    flags = np.full(aid.size, ISBAD)

    assert np.unique(gid).size == gid.size and np.unique(aid).size == aid.size

    badids = np.setdiff1d(aid, gid)
    bad = np.isin(aid, badids)

    flags[~bad] = ISGOOD

    return SweepMetadata(resid=aid, wsfreq=afreq, flag=flags, file=outfile)

def getSweepFilesFromPat(sweepFilePat):
    sweepGlobPat = sweepFilePat.replace('{roach}', '???') 
    sweepGlobPat = sweepGlobPat.replace('{feedline}', '*')  
    sweepGlobPat = sweepGlobPat.replace('{range}', '?')

    sweepFiles = glob.glob(sweepGlobPat)

    sweepFmt = sweepFilePat.replace('*', '{}')
    sweepFmt = sweepFmt.replace('{roach}', '{roach:d}')
    sweepFmt = sweepFmt.replace('{feedline}', '{feedline:d}')

    paramDicts = []

    for sweepFile in sweepFiles:
        sweepParamDict = parse.parse(sweepFmt, sweepFile).named
        paramDicts.append(sweepParamDict)

    return sweepFiles, paramDicts
  

def matchSweepToMetadataPat(sweepFilePat, mdFilePat):
    """
    Returns a list of sweep files and corresponding metadata according to 
    the file patterns specified in 'sweepFilePat' and 'mdFilePat'. Patterns
    may use {roach}, {feedline}, and/or {range} tags to match
    corresponding files, as well as standard glob syntax such as * and ?.

    Does NOT currently make any assumptions about which roach boards are 
    associated with which feedlines/ranges (so format specifiers should
    have at least one matching tag).

    parameters
    ----------
        sweepFilePat: string
            Pattern to match sweep npz files (something like,
            '/path/to/files/psData_{roach}.npz')
        mdFilePat: string
            Pattern to match metadata files

    returns
    -------
        list of sweep files
        corresponding list of metadata files
        dictionary w/ 'roach', 'feedline' and/or 'range' 
            specifiers for each set of files

    """
    sweepGlobPat = sweepFilePat.replace('{roach}', '???')
    mdGlobPat = mdFilePat.replace('{roach}', '???')
    sweepGlobPat = sweepGlobPat.replace('{feedline}', '*')
    mdGlobPat = mdGlobPat.replace('{feedline}', '*')
    sweepGlobPat = sweepGlobPat.replace('{range}', '?')
    mdGlobPat = mdGlobPat.replace('{range}', '?')

    sweepFiles = np.array(glob.glob(sweepGlobPat))
    mdFiles = np.array(glob.glob(mdGlobPat))

     #find md file corresponding to each sweep
    mdFilesOrdered = []
    paramDicts = [] #list of parsed out params for each file
    sweepFmt = sweepFilePat.replace('*', '{}')
    mdFmt = mdFilePat.replace('*', '{}')

    sweepFmt = sweepFmt.replace('{roach}', '{roach:d}')
    sweepFmt = sweepFmt.replace('{feedline}', '{feedline:d}')
    mdFmt = mdFmt.replace('{roach}', '{roach:d}')
    mdFmt = mdFmt.replace('{feedline}', '{feedline:d}')

    for sweepFile in sweepFiles:
        sweepParamDict = parse.parse(sweepFmt, sweepFile).named
        mdMatchesSweep = np.ones(len(mdFiles)).astype(bool)
        for i, mdFile in enumerate(mdFiles):
            mdParamDict = parse.parse(mdFmt, mdFile).named
            for mdKey, mdVal in mdParamDict.items():
                if mdKey in sweepParamDict:
                    if sweepParamDict[mdKey] != mdVal:
                        mdMatchesSweep[i] = False #make sure all overlapping keys match

        if np.sum(mdMatchesSweep) > 1:
            raise Exception('Multiple metadata files matching for {}'.format(sweepFile))
        if np.sum(mdMatchesSweep) == 0:
            getLogger(__name__).warning('No metadata found for {}. Skipping.'.format(sweepFile))
            mdFilesOrdered.append(None)
            paramDicts.append(None)
            continue

        matchingMD = mdFiles[mdMatchesSweep][0]
        mdParamDict = parse.parse(mdFmt, matchingMD).named
        sweepParamDict.update(mdParamDict)
        if not 'roach' in sweepParamDict:
            sweepParamDict['roach'] = '???'
        if not 'feedline' in sweepParamDict:
            sweepParamDict['feedline'] = '?'
        if not 'range' in sweepParamDict:
            sweepParamDict['range'] = '?'
        paramDicts.append(sweepParamDict)
        mdFilesOrdered.append(matchingMD)

    #weirdness here is to filter out 'None'
    mdFNP = np.array(mdFilesOrdered, dtype=object)
    if len(mdFNP[mdFNP.astype(bool)]) != len(np.unique(mdFNP[mdFNP.astype(bool)])):
        raise Exception('Duplicate MD files')

    return list(sweepFiles), mdFilesOrdered, paramDicts
