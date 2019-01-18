import numpy as np
import os
from mkidreadout.configuration.sweepdata import FreqSweep, SweepMetadata, ISGOOD, ISBAD

class MLData(object):
    def __init__(self, freqSweepFn, metadataFn):
        self.freqSweep = FreqSweep(freqSweepFn)
        self.metadata = SweepMetadata(metadataFn)
        
        #to define: self.freqs, self.Is, self.Qs, self.attens, self.resIDs
        #   self.opt_attens, self.opt_freqs, self.scores
        freqSpan = np.array([self.freqSweep.freqs[0,0], self.freqSweep.freqs[-1,-1]])
        self.mdResMask = (self.metadata.flag == ISGOOD) & (self.metadata.wsfreq > freqSpan[0]) & (self.metadata.wsfreq < freqSpan[1])
        self.nRes = np.sum(self.mdResMask)
        self.resIDs = self.metadata.resIDs[self.mdResMask]
        self.initfreqs = self.metadata.wsfreq[self.mdResMask]
        self.opt_attens = self.metadata.atten[self.mdResMask]
        self.opt_freqs = self.metadata.mlfreq[self.mdResMask]
        self.scores = np.zeros(self.nRes)
        if not np.all(np.isnan(self.opt_attens)):
            attenblock = np.tile(self.freqSweep.atten, (len(self.opt_attens),1))
            self.opt_iAttens = np.where(opt_attens==attenblock.T)[0]

        generateMLWindows(self.initfreqs)

    def generateMLWindows(self, resFreqs):
        assert len(resFreqs)==self.nRes, 'Mismatch between provided freq list and number of res in metadata file'
        self.freqs = np.zeros((self.nRes, self.freqsweep.lostep))
        self.Is = np.zeros((self.nRes, self.freqsweep.natten, self.freqsweep.lostep))
        self.Qs = np.zeros((self.nRes, self.freqsweep.natten, self.freqsweep.lostep))
        winCenters = self.freqSweep.freqs[:, self.freqsweep.lostep/2]
        for i in range(self.nRes):
            freqWinInd = np.argmin(np.abs(resFreqs[i] - winCenters))
            self.freqs[i] = self.freqSweep.freqs[freqWinInd, :]
            self.Is[i] = self.freqSweep.i[:, freqWinInd, :]
            self.Qs[i] = self.freqSweep.q[:, freqWinInd, :]

    def saveInferenceData(self, flag='_ml'):
        self.metadata.attens[self.mdResMask] = self.opt_attens
        self.metadata.mlfreq[self.mdResMask] = self.opt_freqs
        self.metadata.ml_isgood_score[self.mdResMask] = self.scores #TODO: implement ml bad scores
        self.metadata.save(self.metadata.file.split('.')[0] + flag + '.txt')
    

