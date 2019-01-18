import numpy as np
import pkg_resources as pkg
from glob import glob
import copy
import glob
from mkidreadout.instruments import DEFAULT_ARRAY_SIZES
import mkidcore.config
import os
from mkidcore.corelog import getLogger


class _BeamDict(dict):
    def __missing__(self, key):
        bfile = os.path.join(os.path.dirname(__file__), key.lower()+'.bmap')
        self[key] = bfile
        return bfile


DEFAULT_BMAP_CFGFILES = _BeamDict()


class Beammap(object):
    """
    Simple wrapper for beammap file. 
    Attributes:
        resIDs
        flags
        xCoords
        yCoords
    """
    yaml_tag = u'!bmap'

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
        self.file = file
        self.resIDs = None
        self.flags = None
        self.xCoords = None
        self.yCoords = None
        self.frequencies = None

        if file is not None:
            self._load(file)
            try:
                self.ncols, self.nrows = xydim
            except TypeError:
                raise ValueError('xydim is a required parameter when loading a beammap from a file')
            if (self.ncols * self.nrows) != len(self.resIDs):
                raise Exception('The dimensions of the beammap entered do not match the beammap read in')
        else:
            try:
                self._load(pkg.resource_filename(__name__, '{}.bmap'.format(default.lower())))
                self.ncols, self.nrows = DEFAULT_ARRAY_SIZES[default.lower()]
            except IOError:
                opt = ', '.join([f.rstrip('.bmap').upper() for f in glob(pkg.resource_filename(__name__, '*.bmap'))])
                raise ValueError('Unknown default beampmap "{}". Options: {}'.format(default, opt))

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_mapping(cls.yaml_tag, dict(file=node.file, nrows=node.nrows, ncols=node.ncols))

    @classmethod
    def from_yaml(cls, constructor, node):
        d = mkidcore.config.extract_from_node(('file', 'nrows', 'ncols', 'default'), node)
        if 'default' in d:
            return cls(default=d['default'])
        else:
            return cls(file=d['file'], xydim=(int(d['ncols']), int(d['nrows'])))

    def setData(self, bmData):
        """
        Sets resIDs, flags, xCoords, yCoords, (and optionally frequencies) to data in bmData
        INPUTS:
            bmData - Nx4 or Nx5 numpy array in same format as beammap file
        """
        if bmData.shape[1] == 4:
            self.resIDs = np.array(bmData[:, 0])
            self.flags = np.array(bmData[:, 1])
            self.xCoords = np.array(bmData[:, 2])
            self.yCoords = np.array(bmData[:, 3])
        elif bmData.shape[1] == 5:
            self.resIDs = np.array(bmData[:, 0])
            self.flags = np.array(bmData[:, 1])
            self.xCoords = np.array(bmData[:, 2])
            self.yCoords = np.array(bmData[:, 3])
            self.frequencies = np.array(bmData[:, 4])
        else:
            raise Exception("This data is not in the proper format")

    def _load(self, filename):
        """
        Loads beammap data from filename
        """
        self.file = filename
        getLogger(__name__).debug('Reading {}'.format(self.file))
        self.resIDs, self.flags, self.xCoords, self.yCoords = np.loadtxt(filename, unpack=True)

    def loadFrequencies(self, filepath):
        powerSweeps = glob.glob(filepath)
        psData = np.loadtxt(powerSweeps[0])
        for i in range(len(powerSweeps) - 1):
            sweep = np.loadtxt(powerSweeps[i + 1])
            psData = np.concatenate((psData, sweep))
        # psData has the form [Resonator ID, Frequency (Hz), Attenuation (dB)]
        self.frequencies = np.full(self.resIDs.shape, np.nan)
        for j in range(len(psData)):
            idx = np.where(self.resIDs == psData[j][0])[0]
            self.frequencies[idx] = (psData[j][1] / (10 ** 6))

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

    def resIDat(self, x, y):
        return self.resIDs[(np.floor(self.xCoords) == x) & (np.floor(self.yCoords) == y)]

    def getResonatorsAtCoordinate(self, x, y):
        resonators = [self.getResonatorData(r) for r in  self.resIDat(x,y)]
        return resonators

    def get(self, attribute='', flNum=None):
        """
        :params attribute and flNum:
        :return the values of the attribute for a single feedline (denoted by the first number of its resID:
        for use in the beammap shifting code
        """
        if attribute:
            x = self.getBeammapAttribute(attribute)
        else:
            x = None
            raise Exception("This attribute does not exist")
        if flNum:
            mask = flNum == np.floor(self.resIDs / 10000)
        else:
            mask = np.ones_like(self.resIDs, dtype=bool)
        if x.shape == mask.shape:
            return x[mask]
        else:
            raise Exception('Your attribute contained no data')

    def getBeammapAttribute(self, attribute=''):
        """
        :param attribute:
        :return list of attribute values, the length of the beammap object:
        This is for use in the get function
        """
        if attribute.lower() == 'resids':
            return self.resIDs
        elif attribute.lower() == 'flags':
            return self.flags
        elif attribute.lower() == 'xcoords':
            return self.xCoords
        elif attribute.lower() == 'ycoords':
            return self.yCoords
        elif attribute.lower() == 'frequencies':
            return self.frequencies
        else:
            raise Exception('This is not a valid Beammap attribute')

    def getResonatorData(self, resID):
        index = np.where(self.resIDs == resID)[0][0]  #TODO Noah don't use where!
        resonator = [int(self.resIDs[index]), int(self.flags[index]), int(self.xCoords[index]), int(self.yCoords[index]),
                     float(self.frequencies[index])]
        return resonator

    def beammapDict(self):
        return {'resID': self.resIDs, 'freqCh': self.freqs, 'xCoord': self.xCoords,
                'yCoord': self.yCoords, 'flag': self.flags}

    @property
    def failmask(self):
        mask = np.ones((self.nrows, self.ncols), dtype=bool)
        use = (self.yCoords.astype(int) < self.nrows) & (self.xCoords.astype(int) < self.ncols)
        mask[self.yCoords[use].astype(int), self.xCoords[use].astype(int)] = self.flags[use] != 0
        return mask

    @property
    def residmap(self):
        map = np.zeros((self.ncols, self.nrows), dtype=self.resIDs.dtype)
        use = (self.yCoords.astype(int) < self.nrows) & (self.xCoords.astype(int) < self.ncols)
        map[self.xCoords[use].astype(int), self.yCoords[use].astype(int)] = self.resIDs
        return map

    @property
    def flagmap(self):
        map = np.zeros((self.ncols, self.nrows), dtype=self.flags.dtype)
        use = (self.yCoords.astype(int) < self.nrows) & (self.xCoords.astype(int) < self.ncols)
        map[self.xCoords[use].astype(int), self.yCoords[use].astype(int)] = self.flags
        return map

    def __str__(self):
        return 'File: "{}"\nWell Mapped: {}'.format(self.file, self.nrows * self.ncols - (self.flags!=0).sum())



mkidcore.config.yaml.register_class(Beammap)


class DesignArray(object):
    """
    Wrapper for a design array. Given an array whose elements are design frequencies, gives a
    list of x- and y- coordinates and corresponding design frequencies. Typically used with the
    Beammap Shifter class to allow for easy searching of modified (fitted) design frequencies
    attr:
    designXCoords
    designYcoords
    designFrequencies
    designArray (this is the 'shaped' version of what will be turned into lists)
    """
    def __init__(self):
        self.designArray = np.empty(0)
        self.designXCoords = np.empty(0)
        self.designYCoords = np.empty(0)
        self.designFrequencies = np.empty(0)

    def load(self, designData):
        """
        Loads in data
        :param designData:
        :return: Set the designArray attribute
        """
        self.designArray = designData

    def reshape(self):
        """
        Reshapes the array into lists for easy searching
        :return:
        """
        xCoords = []
        yCoords = []
        designFrequencies = []
        for y in range(len(self.designArray)):
            for x in range(len(self.designArray[y])):
                xCoords.append(x)
                yCoords.append(y)
                designFrequencies.append(self.designArray[y][x])

        self.designXCoords = np.array(xCoords)
        self.designYCoords = np.array(yCoords)
        self.designFrequencies = np.array(designFrequencies)

    def getDesignFrequencyFromCoords(self, coordinate):
        """
        Given an (x, y) coordinate, return the design frequency there
        :param coordinate:
        :return:
        """
        xCoord = coordinate[0]
        yCoord = coordinate[1]
        index = int(np.where((self.designXCoords == xCoord) & (self.designYCoords == yCoord))[0])
        designFrequencyAtCoordinate = self.designFrequencies[index]
        return designFrequencyAtCoordinate
