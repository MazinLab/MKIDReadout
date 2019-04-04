import numpy as np
import pkg_resources as pkg
from glob import glob
import copy
import glob
from mkidreadout.instruments import DEFAULT_ARRAY_SIZES
import mkidcore.config
import os
from mkidcore.corelog import getLogger
from mkidcore.objects import Beammap


class _BeamDict(dict):
    def __missing__(self, key):
        bfile = os.path.join(os.path.dirname(__file__), key.lower()+'.bmap')
        self[key] = bfile
        return bfile


DEFAULT_BMAP_CFGFILES = _BeamDict()


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
