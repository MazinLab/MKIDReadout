import numpy as np
import matplotlib.pyplot as plt
import mkidreadout.configuration.beammap.flags as flags
import mkidreadout.configuration.beammap.utils as utils

class Beammap:
    """
    Simple wrapper for beammap file. 
    Attributes:
        resIDs
        flags
        xCoords
        yCoords
    """
    
    def __init__(self):
        self.resIDs = np.empty(0)
        self.flags = np.empty(0)
        self.xCoords = np.empty(0)
        self.yCoords = np.empty(0)
        pass

    def setData(self, bmData):
        """
        Sets resIDs, flags, xCoords, yCoords to data in bmData
        INPUTS:
            bmData - Nx4 numpy array in same format as beammap file
        """
        self.resIDs = np.array(bmData[:,0])
        self.flags = np.array(bmData[:,1])
        self.xCoords = np.array(bmData[:,2])
        self.yCoords = np.array(bmData[:,3])

    def load(self, filename):
        """
        Loads beammap data from filename
        """
        self.resIDs, self.flags, self.xCoords, self.yCoords = np.loadtxt(filename, unpack=True)

    def save(self, filename, forceIntegerCoords=False):
        """
        Saves beammap data to file.
        INPUTS:
            filename - full path of save file
            forceIntegerCoords - if true floors coordinates and saves as integers
        """
        if forceIntegerCoords:
            fmt = '%4i %4i %4i %4i'
        else:
            fmt = '%4i %4i %0.5f %0.5f'
        np.savetxt(filename, np.transpose([self.resIDs, self.flags, self.xCoords, self.yCoords]), fmt=fmt)

    def copy(self):
        """
        Returns a deep copy of itself
        """
        newBeammap = Beammap()
        newBeammap.resIDs = np.copy(self.resIDs)
        newBeammap.flags = np.copy(self.flags)
        newBeammap.xCoords = np.copy(self.xCoords)
        newBeammap.yCoords = np.copy(self.yCoords)
        return newBeammap

