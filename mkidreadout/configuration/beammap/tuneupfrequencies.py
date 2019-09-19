import numpy as np

from mkidreadout.configuration.sweepdata import SweepMetadata


class Correlator(object):
    def __init__(self, oldPath='', newPath='', beammapPath=''):
        self.old = SweepMetadata(file=oldPath)
        self.new = SweepMetadata(file=newPath)
        # self.beammap = Beammap(file=beammapPath)
        self._cleanData()

    def _cleanData(self):
        oldmask = (~np.isnan(self.old.atten)) & (self.old.flag == 1)
        self.oldFreq = self.old.freq[oldmask]
        self.oldRes = self.old.resIDs[oldmask]

        newmask = (~np.isnan(self.new.atten)) & (self.new.flag == 1)
        self.newFreq = self.new.freq[newmask]
        self.newRes = self.new.resIDs[newmask]

    def findFrequencyShift(self):
        self.shifts = np.arange(-1e9, (1e9+1000), 1e6)
        avgResidual = []
        stdResidual = []
        for i in self.shifts:
            if abs(i) % 1e8 == 0:
                print(i)
            match = self.matchFrequencies(i)
            residuals = match[0]-match[1]
            avgResidual.append(np.mean(residuals))
            stdResidual.append(np.std(residuals))
        self.avgResidual = abs(np.array(avgResidual))
        self.stdResidual = np.array(stdResidual)

    def matchFrequencies(self, shift):
        if len(self.newFreq) > len(self.oldFreq):
            longer = self.newFreq+shift
            shorter = self.oldFreq
        else:
            longer = self.oldFreq+shift
            shorter = self.newFreq

        matches = np.zeros((2, len(longer)))
        matches[0] = longer
        for i in range(len(longer)):
            residual = abs(shorter-longer[i])
            mask = residual == min(residual)
            matches[1][i] = shorter[mask][0]

        return matches



