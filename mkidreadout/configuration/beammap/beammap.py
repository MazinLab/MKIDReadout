import numpy as np
import matplotlib.pyplot as plt
import pkg_resources as pkg
import os
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
    
    def __init__(self, beammap):
        """
        Constructor.
        
        INPUTS:
            beammap - either a path to beammap file, instrument name, or 
                beammap object. 
                    If path, loads data from beammap file. 
                    If instrument (either 'mec' or 'darkness'), loads corresponding
                        default beammap. 
                    If instance of Beammap, creates a copy

        """
        if isinstance(beammap, str):
            if os.path.isfile(beammap):
                self.load(beammap)
            elif beammap.lower()=='mec':
                self.load(pkg.resource_filename(__name__, 'mec.bmap'))
            elif beammap.lower()=='darkness':
                self.load(pkg.resource_filename(__name__, 'darkness.bmap'))
            else:
                raise Exception('Must specify beammap file or instrument name')
                
        elif isinstance(beammap, Beammap):
            self.resIDs = beammap.resIDs.copy()
            self.flags = beammap.flags.copy()
            self.xCoords = beammap.xCoords.copy()
            self.yCoords = beammap.yCoords.copy()

        else:
            raise Exception('Input must be either Beammap instance or string')


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
        return Beammap(self)

