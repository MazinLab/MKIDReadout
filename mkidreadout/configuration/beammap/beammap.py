import numpy as np
import pkg_resources as pkg
from glob import glob
import copy
import os

_DEFAULT_ARRAY_SIZES={'mec':(100,100), 'darkness': (150,150)}

class _BeamDict(dict):
    def __missing__(self, key):
        bfile = os.path.join(os.path.dirname(__file__), key.lower()+'.bmap')
        self[key] = bfile
        return bfile

DEFAULT_BMAP_FILES  = _BeamDict()


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
        self.file = ''
        if file is not None:
            self._load(file)
            try:
                self.nrows, self.ncols = xydim
            except TypeError:
                raise ValueError('xydim is required when loading from file')
        else:
            try:
                self._load(pkg.resource_filename(__name__, '{}.bmap'.format(default.lower())))
                self.nrows, self.ncols = _DEFAULT_ARRAY_SIZES[default.lower()]
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
        self.file = filename

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

    @property
    def failmask(self, ):
        #x = np.ones((self.nrows, self.ncols), dtype=bool)
        # for i in range(len(self.resIDs)):
        #     try:
        #         x[int(self.yCoords[i]), int(self.xCoords[i])] = self.flags[i] != 0
        #     except IndexError:
        #         pass

        mask = np.ones((self.nrows, self.ncols), dtype=bool)
        use = (int(self.yCoords) < self.nrows) & (int(self.xCoords) < self.ncols)
        mask[int(self.yCoords[use]), int(self.xCoords[use])] = self.flags[use].nonzero()

        return mask

    def __str__(self):
        return 'File: "{}"\nWell Mapped: {}'.format(self.file, self.nrows * self.ncols - self.flags.nonzero().sum())
