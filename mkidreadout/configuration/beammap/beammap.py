import numpy as np
import glob
import matplotlib.pyplot as plt
import mkidreadout.configuration.beammap.flags as flags
#import mkidreadout.configuration.beammap.utils as utils

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
        self.frequencies = np.empty(0)
        pass

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

    def load(self, filename):
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
        newBeammap = Beammap()
        newBeammap.resIDs = np.copy(self.resIDs)
        newBeammap.flags = np.copy(self.flags)
        newBeammap.xCoords = np.copy(self.xCoords)
        newBeammap.yCoords = np.copy(self.yCoords)
        return newBeammap

    def getResonatorsAtCoordinate(self, xCoordinate, yCoordinate):
        indices = np.where((self.xCoords == xCoordinate) & (self.yCoords == yCoordinate))[0]
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


class DesignArray:

    def __init__(self):
        self.designArray = np.empty(0)
        self.designXCoords = np.empty(0)
        self.designYCoords = np.empty(0)
        self.designFrequencies = np.empty(0)

    def load(self, designData):
        self.designArray = designData

    def reshape(self):
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
        xCoord = coordinate[0]
        yCoord = coordinate[1]
        index = int(np.where((self.designXCoords == xCoord) & (self.designYCoords == yCoord))[0])
        designFrequencyAtCoordinate = self.designFrequencies[index]
        return designFrequencyAtCoordinate
