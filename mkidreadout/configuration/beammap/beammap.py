import numpy as np
import pkg_resources as pkg
from glob import glob
import copy

_DEFAULT_ARRAY_SIZES = {'mec': (140, 146), 'darkness': (80, 125)}


class Beammap(object):
    """
    Simple wrapper for beammap file. 
    Attributes:
        resIDs
        flags
        xCoords
        yCoords
    """
    def __init__(self, file=None, xydim=None, default='MEC'):
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
        if file is not None:
            self._load(file)
            try:
                self.ncols, self.nrows = xydim
            except TypeError:
                raise ValueError('xydim is a required parameter when loading a beammap from a file')
        else:
            try:
                self._load(pkg.resource_filename(__name__, '{}.bmap'.format(default.lower())))
                self.ncols, self.nrows = _DEFAULT_ARRAY_SIZES[default.lower()]
            except IOError:

                opt = ', '.join([f.rstrip('.bmap').upper()
                                 for f in glob(pkg.resource_filename(__name__, '*.bmap'))])
                raise ValueError('Unknown default beampmap "{}". Options: {}'.format(default, opt))

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

    def _load(self, filename):
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
        return copy.deepcopy(self)

