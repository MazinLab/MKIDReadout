import numpy as np
import pkg_resources as pkg
from glob import glob
import copy
import glob
import matplotlib.pyplot as plt
import mkidreadout.configuration.beammap.flags as flags
#import mkidreadout.configuration.beammap.utils as utils


class Beammap(object):
    """
    Simple wrapper for beammap file. 
    Attributes:
        resIDs
        flags
        xCoords
        yCoords
    """
    def __init__(self, file=None, default='MEC'):
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
        else:
            try:
                self._load(pkg.resource_filename(__name__, '{}.bmap'.format(default.lower())))
            except IOError:
                opt = ', '.join([f.rstrip('.bmap').upper() for f in glob(pkg.resource_filename(__name__, '*.bmap'))])
                raise ValueError('Unknown default beampmap "{}". Options: {}'.format(default, opt))

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

    def getResonatorsAtCoordinate(self, xCoordinate, yCoordinate):
        indices = np.where((np.floor(self.xCoords) == xCoordinate) & (np.floor(self.yCoords) == yCoordinate))[0]
        resonators = []
        for idx in indices:
            resonators.append(self.getResonatorData(self.resIDs[idx]))
        return np.array(resonators)

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
        index = np.where(self.resIDs == resID)[0][0]
        resonator = [int(self.resIDs[index]), int(self.flags[index]), int(self.xCoords[index]), int(self.yCoords[index]),
                          float(self.frequencies[index])]
        return resonator

    def beammapDict(self):
        return {'resID': self.resIDs, 'freqCh': self.freqs, 'xCoord': self.xCoords,
                'yCoord': self.yCoords, 'flag': self.flags}


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
