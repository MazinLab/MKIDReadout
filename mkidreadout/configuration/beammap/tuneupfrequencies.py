import numpy as np
import matplotlib.pyplot as plt

from mkidreadout.configuration.sweepdata import SweepMetadata
from mkidcore.objects import Beammap


class Correlator(object):
    def __init__(self, oldPath='', newPath='', beammapPath=''):
        self.old = SweepMetadata(file=oldPath)
        self.new = SweepMetadata(file=newPath)
        # self.beammap = Beammap(file=beammapPath)
        self._cleanData()
        self.fitData()

    def _cleanData(self):
        oldmask = (~np.isnan(self.old.atten)) & (self.old.flag == 1)
        self.oldFreq = self.old.freq[oldmask]
        self.oldRes = self.old.resIDs[oldmask]

        newmask = (~np.isnan(self.new.atten)) & (self.new.flag == 1)
        self.newFreq = self.new.freq[newmask]
        self.newRes = self.new.resIDs[newmask]

    def fitData(self):
        o = np.polyfit(self.oldFreq, self.oldRes, deg=4)
        n = np.polyfit(self.newFreq, self.newRes, deg=4)
        self.oFunc = np.poly1d(o)
        self.nFunc = np.poly1d(n)

    def plot(self):
        plt.figure(1)
        plt.plot(self.oldFreq, self.oldRes, '.', label="Old Data")
        plt.plot(self.newFreq, self.newRes, '.', label="New Data")
        plt.legend()
        plt.savefig('data.png')

        plt.figure(2)
        plt.plot(self.newFreq, self.oFunc(self.newFreq), '-', label="Old Interpolation")
        plt.plot(self.newFreq, self.nFunc(self.newFreq), '-', label="New Interpolation")
        plt.legend()
        plt.savefig('interpolation.png')