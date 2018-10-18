"""
cleanV2.py

Takes rough beammap with overlapping pixels and pixels out of bounds
Reconciles doubles by placing them in nearest empty pix
Places o.o.b pixels in random position, changes flag to 1
Places failed pixels randomly.
Outputs final cleaned beammap with ID, flag, x, y; ready to go into dashboard

TODO: refactor to actually use the Beammap class

"""

from __future__ import print_function
import numpy as np
import os
import logging
import argparse
import ipdb
import matplotlib.pyplot as plt
import itertools
from mkidreadout.utils.arrayPopup import plotArray
from mkidreadout.utils.readDict import readDict
from mkidreadout.configuration.beammap.flags import beamMapFlags
from mkidreadout.configuration.beammap.utils import isInCorrectFL, getFLFromID, getFLFromCoords, isResonatorOnCorrectFeedline
from mkidreadout.configuration.beammap.beammap import Beammap
from mkidreadout.configuration.beammap import shift

MEC_FL_WIDTH = 14
DARKNESS_FL_WIDTH = 25

N_FL_MEC = 10
N_FL_DARKNESS = 5

logging.basicConfig()
log = logging.getLogger('beammap.clean')


def getOverlapGrid(xCoords, yCoords, flags, nXPix, nYPix):
    pixToUseMask = (flags==beamMapFlags['good']) | (flags==beamMapFlags['double'])
    xCoords = xCoords.astype(np.int)[pixToUseMask]
    yCoords = yCoords.astype(np.int)[pixToUseMask]
    bmGrid = np.zeros((nXPix, nYPix))
    for x,y in zip(xCoords,yCoords):
        if x >= 0 and y >= 0 and x < nXPix and y < nYPix:
            bmGrid[x, y] += 1
    return bmGrid





class BMCleaner:
    def __init__(self, roughBeammap, nRows, nCols, flip, instrument):
        self.nRows = nRows
        self.nCols = nCols
        self.flip = flip
        self.instrument = instrument.lower()

        if self.instrument.lower()=='mec':
            self.nFL = N_FL_MEC
            self.flWidth = MEC_FL_WIDTH
        elif self.instrument.lower()=='darkness':
            self.nFL = N_FL_DARKNESS
            self.flWidth = DARKNESS_FL_WIDTH
        else:
            raise Exception('Provided instrument not implemented!')
    
        self.resIDs = roughBeammap.resIDs.astype(int)
        self.flags = roughBeammap.flags.astype(int)
        self.preciseXs = roughBeammap.xCoords
        self.preciseYs = roughBeammap.yCoords
        self.flooredXs = None
        self.flooredYs = None
        self.placedXs = None
        self.placedYs = None
        self.bmGrid = None

    def fixPreciseCoordinates(self, arraySlack=1, flSlack=1):
        '''
        Checks precise coordinates to make sure they are within the bounds of the array
        and in the correct feedline. Flags OOB pixels as bad and sets to NaN.
        
        INPUTS:
            arraySlack - OOB pixels (array boundaries) within arraySlack of boundary are
                placed at boundary and flagged good.
            flSlack - OOB pixels (array boundaries) within arraySlack of FL boundary are
                placed at boundary and flagged good.
        '''
        self._fixOOBPixels(arraySlack)
        self._fixInitialFeedlinePlacement(flSlack)


    def placeOnGrid(self):
        '''
        Places all resonators on nRowsxnCols size grid. Only do this after precise coordinates are finalized.
        Results are stored in self.placedXs and self.placedYs
        '''
        #lock in coordinates, set up placement tools/arrays
        self.flooredXs = self.preciseXs.astype(np.int)
        self.flooredYs = self.preciseYs.astype(np.int)
        self.placedXs = np.floor(self.preciseXs)
        self.placedYs = np.floor(self.preciseYs)
        self.bmGrid = getOverlapGrid(self.preciseXs, self.preciseYs, self.flags, self.nCols, self.nRows)


    def _fixInitialFeedlinePlacement(self, slack=1):
        wayOutMask = ~isInCorrectFL(self.resIDs, self.preciseXs, self.preciseYs, self.instrument, slack, self.flip)
        self.flags[((self.flags==beamMapFlags['good']) | (self.flags==beamMapFlags['double'])) & wayOutMask] = beamMapFlags['wrongFeedline']
        self.preciseXs[self.flags==beamMapFlags['wrongFeedline']] = np.nan
        self.preciseYs[self.flags==beamMapFlags['wrongFeedline']] = np.nan
        log.info('%d pixels in wrong feedline. Flagged as bad', np.sum(self.flags==beamMapFlags['wrongFeedline']))

        if slack>0:
            flPos = getFLFromID(self.resIDs) - 1
            if self.flip:
                flPos = 9 - flPos
            rightEdgeMask = flPos < getFLFromCoords(self.preciseXs, self.preciseYs, self.instrument, flip=False) - 1
            leftEdgeMask = flPos > getFLFromCoords(self.preciseXs, self.preciseYs, self.instrument, flip=False) - 1
            assert np.all((rightEdgeMask & leftEdgeMask) == 0), 'Error moving pixels to correct FL'
            log.info(str(np.sum(rightEdgeMask)+np.sum(leftEdgeMask)) + ' placed in correct feedline')
            if self.instrument=='mec':
                self.preciseXs[rightEdgeMask] = (flPos[rightEdgeMask]+1)*self.flWidth-0.01
                self.preciseXs[leftEdgeMask] = (flPos[leftEdgeMask])*self.flWidth+0.01
            elif self.instrument=='darkness':
                self.preciseYs[rightEdgeMask] = (flPos[rightEdgeMask]+1)*self.flWidth-0.01
                self.preciseYs[leftEdgeMask] = (flPos[leftEdgeMask])*self.flWidth+0.01
            else:
                raise Exception('Provided instrument not implemented!')
            validCoordMask = (~np.isnan(self.preciseXs)) & (~np.isnan(self.preciseYs))
            assert np.all(isInCorrectFL(self.resIDs[validCoordMask], self.preciseXs[validCoordMask], self.preciseYs[validCoordMask], self.instrument, 0, self.flip)), 'bad FL cleanup!'


    def _fixOOBPixels(self, slack=1):
        wayOutMask = (self.preciseXs + slack < 0) | (self.preciseXs - slack > self.nCols - 1) | (self.preciseYs + slack < 0) | (self.preciseYs - slack > self.nRows - 1)
        self.preciseXs[wayOutMask] = np.nan
        self.preciseYs[wayOutMask] = np.nan
        self.flags[wayOutMask] = beamMapFlags['wrongFeedline']

        self.preciseXs[(self.preciseXs < 0) & ~wayOutMask] = 0
        self.preciseXs[(self.preciseXs >= self.nCols) & ~wayOutMask] = self.nCols - 0.01
        self.preciseYs[(self.preciseYs < 0) & ~wayOutMask] = 0
        self.preciseYs[(self.preciseYs >= self.nRows) & ~wayOutMask] = self.nRows - 0.01

    
    def resolveOverlaps(self):
        '''
        Resolves overlaps out to one nearest neighbor. Results stored in self.placedXs and self.placedYs. self.bmGrid is also
        modified.
        '''
        overlapCoords = np.asarray(np.where(self.bmGrid > 1)).T
        unoccupiedCoords = np.asarray(np.where(self.bmGrid==0)).T 
        nOverlapsResolved = 0

        for coord in overlapCoords:
            coordMask = (coord[0] == self.flooredXs) & (coord[1] == self.flooredYs) & ((self.flags == beamMapFlags['good']) | (self.flags == beamMapFlags['double']))
            coordInds = np.where(coordMask)[0] #indices of overlapping coordinates in beammap
    
            uONNCoords = np.asarray(np.where(self.bmGrid[coord[0]-1:coord[0]+2, coord[1]-1:coord[1]+2]==0)).T + coord - np.array([1,1])
            uONNCoords = uONNCoords[isInCorrectFL(self.resIDs[coordInds[0]]*np.ones(len(uONNCoords)), uONNCoords[:,0], uONNCoords[:,1], self.instrument, 0, self.flip), :]
                
    
            precXOverlap = self.preciseXs[coordMask] - 0.5
            precYOverlap = self.preciseYs[coordMask] - 0.5
            precOverlapCoords = np.array(zip(precXOverlap, precYOverlap)) #list of precise coordinates overlapping with coord
    
            # no nearest neigbors, so pick the closest one and flag the rest as bad
            if len(uONNCoords)==0:
                distsFromCenter = (precOverlapCoords[:,0] - coord[0])**2 + (precOverlapCoords[:,1] - coord[1])**2
                minDistInd = np.argmin(distsFromCenter) #index in precOverlapCoords
                coordInds = np.delete(coordInds, minDistInd) #remove correct coordinate from overlap
                self.bmGrid[coord[0], coord[1]] = 1
                self.flags[coordInds] = beamMapFlags['duplicatePixel']
                self.placedXs[coordInds] = np.nan
                self.placedYs[coordInds] = np.nan
                continue
    
            #distMat[i,j] is distance from the ith overlap coord to the jth unoccupied nearest neighbor
            distMat = np.zeros((len(precOverlapCoords), len(uONNCoords))) 
            for i in range(len(precOverlapCoords)):
                distMat[i, :] = (precOverlapCoords[i][0] - uONNCoords[:,0])**2 + (precOverlapCoords[i][1] - uONNCoords[:,1])**2
    
            for i in range(np.sum(coordMask)-1):
                minDistInd = np.unravel_index(np.argmin(distMat), distMat.shape)
                toMoveCoordInd = coordInds[minDistInd[0]] #index in beammap
                nnToFillCoord = uONNCoords[minDistInd[1]]
    
                self.placedXs[toMoveCoordInd] = nnToFillCoord[0]
                self.placedYs[toMoveCoordInd] = nnToFillCoord[1]
                self.bmGrid[nnToFillCoord[0], nnToFillCoord[1]] += 1
                self.bmGrid[coord[0], coord[1]] -= 1
    
                distMat = np.delete(distMat, minDistInd[1], axis=1) #delete column corresponding to NN just filled
                distMat = np.delete(distMat, minDistInd[0], axis=0) #delete row corresponding to moved coordinate
                coordInds = np.delete(coordInds, minDistInd[0])
                uONNCoords = np.delete(uONNCoords, minDistInd[1], axis=0)
                precOverlapCoords = np.delete(precOverlapCoords, minDistInd[0], axis=0)
                precXOverlap = np.delete(precXOverlap, minDistInd[0])
                precYOverlap = np.delete(precYOverlap, minDistInd[0])
    
                nOverlapsResolved += 1
    
                if distMat.shape[1]==0:
                    distsFromCenter = (precOverlapCoords[:,0] - coord[0])**2 + (precOverlapCoords[:,1] - coord[1])**2
                    minDistInd = np.argmin(distsFromCenter) #index in precOverlapCoords
                    coordInds = np.delete(coordInds, minDistInd) #remove correct coordinate from overlap
                    self.bmGrid[coord[0], coord[1]] = 1
                    self.flags[coordInds] = beamMapFlags['duplicatePixel']
                    self.placedXs[coordInds] = np.nan
                    self.placedYs[coordInds] = np.nan
                    break
                    
        log.info('Successfully resolved %d overlaps', nOverlapsResolved)
        log.info('Failed to resolve %d overlaps', np.sum(self.flags==beamMapFlags['duplicatePixel']))

    def resolveOverlapWithFrequency(self, shifterObject):
        """
        Given a shifted beammap object that has frequency data, resolve doubles
        """
        beammap = shifterObject.shiftedBeammap
        designXcoords = shifterObject.designXCoords
        designYcoords = shifterObject.designYCoords
        designFrequencies = shifterObject.designFrequencies

        if not hasattr(beammap, 'frequencies'):
            raise Exception("This beammap does not have frequency data, this operation cannot be done")

        overlapGrid = getOverlapGrid(beammap.xCoords, beammap.yCoords, beammap.flags, self.nCols, self.nRows)
        overlapCoords = np.asarray(np.where(overlapGrid > 1)).T
        numberOfOverlapsResolved = 0

        for coord in overlapCoords:
            doubles = self.getResonatorsAtCoordinate(coord[0], coord[1], beammap)
            freqAtCoord = self.getDesignFrequencyFromCoords(designFrequencies, designXcoords, designYcoords, coord[0], coord[1])

            # Create a list of coordinates that the resonators could be at (i.e. on the array and correct FL, and they do not already have a resonator on them)
            coordsToSearch = self.generateCoordsToSearch(coord)
            onArrayMask = ((coordsToSearch[:, 0] >= 0) & (coordsToSearch[:, 0] < self.nCols) & (coordsToSearch[:, 1] >= 0) & (coordsToSearch[:, 1] < self.nRows))
            coordsToSearch = coordsToSearch[onArrayMask]
            onFeedlineMask = np.zeros(len(coordsToSearch))
            for i in range(len(coordsToSearch)):
                onFeedlineMask[i] = isResonatorOnCorrectFeedline(doubles[0][0], coordsToSearch[i][0], coordsToSearch[i][1], self.instrument)
            coordsToSearch = coordsToSearch[onArrayMask.astype(bool)]
            occupationMask = np.zeros(len(coordsToSearch))
            for i in range(len(coordsToSearch)):
                if overlapGrid[coordsToSearch[i][0], coordsToSearch[i][1]] == 0:
                    occupationMask[i] = True
                else:
                    occupationMask[i] = False
            coordsToSearch = coordsToSearch[occupationMask.astype(bool)]
            coordsToSearch = np.append(coordsToSearch, coord).reshape((len(coordsToSearch)+1, 2))  # Adds the overlap coordinate back to the search coords (they can be placed there)
            freqsToSearch = (self.getDesignFrequencyFromCoords(designFrequencies, designXcoords, designYcoords, coordinate[0], coordinate[1]) for coordinate in coordsToSearch) # Design frequencies at each of the search coordinates

            # Creates an array where each row corresponds to a resonator and each element corresponds to the residual at a coordinate in the coordsToSearch array
            residuals = np.zeros((len(doubles),len(coordsToSearch)))
            for i in range(len(doubles)):
                for j in range(len(freqsToSearch)):
                    residuals[i][j] = abs(doubles[i][4] - freqsToSearch[j])

            # Gets an array of all possible combinations of placements of resonators in the available pixels
            placementPossibilities = self.permuteResiduals(residuals)
            totalResiduals = []
            for i in placementPossibilities:
                temparray = np.zeros(len(i)+1)
                sumOfResiduals = 0
                for j in range(len(i)):
                    temparray[j] = i[j]
                    sumOfResiduals += i[j]
                temparray[-1] = sumOfResiduals
                totalResiduals.append(temparray)
            # totalResiduals has a row for each permutation of position placements, the first columns are the residuals, the final column is the sum of residuals at that configuration
            totalResiduals = np.array(totalResiduals)

            # Find where the total frequency residual is minimized (i.e. we placed the resonators in the double at the 'best' places)
            index = np.where(totalResiduals[:, -1] == np.min(totalResiduals[:, -1]))[0][0]
            minimumCombinedResidual = totalResiduals[index]

            # Finds which coordinate each residual corresponds to for each resonator
            indicesOfMinimumResidual = np.zeros(len(minimumCombinedResidual)-1)
            for i in range(len(minimumCombinedResidual)-1):
                indicesOfMinimumResidual[i] = np.where(residuals[i] == minimumCombinedResidual[i])[0]
            if len(np.unique(indicesOfMinimumResidual)) == indicesOfMinimumResidual:
                overlapGrid[coord[0], coord[1]] -= len(indicesOfMinimumResidual)
                for i in range(len(doubles)):
                    newCoordinate = coordsToSearch[int(indicesOfMinimumResidual[i])]
                    overlapGrid[newCoordinate[0], newCoordinate[1]] += 1
                    resonator = doubles[i]
                    self.updateResonatorCoordinate(resonator, beammap, newCoordinate)
                numberOfOverlapsResolved += 1


    def placeFailedPixels(self):
        '''
        Places all bad pixels (NaN coordinates) in arbitrary locations on correct feedline. Should ensure
        1-to-1 mapping between resID and coordinates.
        '''
        toPlaceMask = np.isnan(self.placedXs) | np.isnan(self.placedYs)
        unoccupiedCoords = np.asarray(np.where(self.bmGrid==0)).T
        
        for i in range(self.nFL):
            toPlaceMaskCurFL = toPlaceMask & ((self.resIDs/10000 - 1) == i)
            unoccupiedCoordsCurFL = unoccupiedCoords[isInCorrectFL(10000*(i+1)*np.ones(len(unoccupiedCoords)), unoccupiedCoords[:,0], unoccupiedCoords[:,1], self.instrument, flip=self.flip)]
            self.placedXs[toPlaceMaskCurFL] = unoccupiedCoordsCurFL[:,0]
            self.placedYs[toPlaceMaskCurFL] = unoccupiedCoordsCurFL[:,1]

    def placeFailedPixelsQuick(self): 
        '''
        Places all bad pixels just outside array bounds. Use when placeFailedPixels fails but you need a 
        quick beammap without NaNs!
        '''
        toPlaceXs = np.isnan(self.placedXs) 
        toPlaceYs = np.isnan(self.placedYs)
        self.placedXs[toPlaceXs] = self.nCols
        self.placedYs[toPlaceYs] = self.nRows

    def getResonatorsAtCoordinate(self, xCoordinate, yCoordinate, beamMap):
        indices = np.where((beamMap.xCoords == xCoordinate) & (beamMap.yCoords == yCoordinate))[0]
        resonators = []
        for idx in indices:
            resonators.append(beamMap.getResonatorData(beamMap.resIDs[idx]))
        return np.array(resonators)

    def getDesignFrequencyFromCoords(self, designFrequencies, designArrayXvalues, designArrayYvalues, xCoord, yCoord):
        index = int(np.where((designArrayXvalues == xCoord) & (designArrayYvalues == yCoord))[0])
        designFrequencyAtCoordinate = designFrequencies[index]
        return designFrequencyAtCoordinate

    def generateCoordsToSearch(self, coordinate):
        coordList = np.zeros((9, 2))
        coordList[0] = [coordinate[0] - 1, coordinate[1] - 1]
        coordList[1] = [coordinate[0] - 1, coordinate[1]]
        coordList[2] = [coordinate[0] - 1, coordinate[1] + 1]
        coordList[3] = [coordinate[0], coordinate[1] - 1]
        coordList[4] = [coordinate[0], coordinate[1]]
        coordList[5] = [coordinate[0], coordinate[1] + 1]
        coordList[6] = [coordinate[0] + 1, coordinate[1] - 1]
        coordList[7] = [coordinate[0] + 1, coordinate[1]]
        coordList[8] = [coordinate[0] + 1, coordinate[1] + 1]
        return coordList.astype(int)

    def permuteResiduals (self, residuals):
        if len(residuals) <= 1:
            raise Exception ("This is trying to resolve a non-overlap")
        elif len(residuals) == 2:
            residualCombinations = itertools.product(residuals[0],residuals[1])
        elif len(residuals) == 3:
            residualCombinations = itertools.product(residuals[0],residuals[1],residuals[2])
        elif len(residuals) == 4:
            residualCombinations = itertools.product(residuals[0],residuals[1],residuals[2], residuals[3])
        elif len(residuals) == 5:
            residualCombinations = itertools.product(residuals[0],residuals[1],residuals[2], residuals[3], residuals[5])
        return residualCombinations

    def updateResonatorCoordinate (self, resonator, beammap, newCoordinate):
        index = np.where(beammap.resIDs == resonator[0])[0]
        beammap.xCoords[index] = newCoordinate[0]
        beammap.yCoords[index] = newCoordinate[1]

    def saveBeammap(self, path):
        '''
        Saves beammap in standard 4 column text file format
            INPUTS:
                path - full path of beammap file
        '''
        assert np.all(np.isnan(self.placedXs)==False), 'NaNs in final beammap!'
        assert np.all(np.isnan(self.placedYs)==False), 'NaNs in final beammap!'
        log.info('N good pixels: ' + str(np.sum(self.flags==beamMapFlags['good'])))
        log.info('N doubles: ' + str(np.sum(self.flags==beamMapFlags['double'])))
        log.info('N failed pixels (read out but not beammapped): ' + str(np.sum((self.flags!=beamMapFlags['good']) & (self.flags!=beamMapFlags['double']) & (self.flags!=beamMapFlags['noDacTone']))))
        np.savetxt(path, np.transpose([self.resIDs, self.flags, self.placedXs.astype(int), self.placedYs.astype(int)]), fmt='%4i %4i %4i %4i')
     

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('cfgFile', nargs=1, type=str, default='/mnt/data0/MKIDReadoout/configuration/clean.cfg', help='Configuration file')
    #default config file location
    args = parser.parse_args()
    configFileName = args.cfgFile[0]
    log.setLevel(logging.INFO)
    
    # Open config file
    configData = readDict()
    configData.read_from_file(configFileName)
    beammapDirectory = configData['beammapDirectory']
    finalBMFile = configData['finalBMFile']
    rawBMFile = configData['rawBMFile']
    psFiles = configData['powersweeps']
    designFile = configData['designMapFile']

    #put together full input/output BM paths
    finalPath = os.path.join(beammapDirectory,finalBMFile)
    rawPath = os.path.join(beammapDirectory,rawBMFile)
    frequencySweepPath = os.path.join(beammapDirectory,"ps_*.txt")


    #load location data from rough BM file
    rawBM = Beammap()
    rawBM.load(rawPath)
    rawBM.loadFrequencies(frequencySweepPath)
   
    cleaner = BMCleaner(rawBM, int(configData['numRows']), int(configData['numCols']), configData['flip'], configData['instrument'])
    shifter = shift.BeammapShifter(designFile, rawBM, configData['instrument'])
    shifter.run()
    cleaner.resolveOverlapWithFrequency(shifter)

    cleaner.fixPreciseCoordinates() #fix wrong feedline and oob coordinates
    cleaner.placeOnGrid() #initial grid placement

    #plt.ion()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.imshow(cleaner.bmGrid.T)
    ax1.set_title('Initial (floored) beammap (value = res per pix coordinate)')

    # resolve overlaps and place failed pixels
    cleaner.resolveOverlaps()
    cleaner.placeFailedPixels()
    cleaner.saveBeammap(finalPath)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.imshow(cleaner.bmGrid.T)
    ax2.set_title('Beammap after fixing overlaps (value = res per pix coordinate)')

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    goodPixMask = (cleaner.flags==beamMapFlags['good']) | (cleaner.flags==beamMapFlags['double'])
    placedXsGood = cleaner.placedXs[goodPixMask]
    placedYsGood = cleaner.placedYs[goodPixMask]
    preciseXsGood = cleaner.preciseXs[goodPixMask] - 0.5
    preciseYsGood = cleaner.preciseYs[goodPixMask] - 0.5
    print(placedXsGood - preciseXsGood)

    #u = np.zeros((nRows, nCols))
    #v = np.zeros((nRows, nCols))
    #for i in range(np.sum(goodPixMask)):
    #    if placedXsGood[i] < nCols and placedYsGood[i] < nRows and np.abs(placedXsGood[i] - preciseXsGood[i]) < 0.5 and np.abs(placedYsGood[i] - preciseYsGood[i]) < 0.5:
    #        u[int(placedYsGood[i]), int(placedXsGood[i])] = placedXsGood[i] - preciseXsGood[i]
    #        v[int(placedYsGood[i]), int(placedXsGood[i])] = placedYsGood[i] - preciseYsGood[i]
    #    
    #ax3.streamplot(np.arange(0, nCols), np.arange(0, nRows), u, v, density=5)

    ax3.quiver(preciseXsGood, preciseYsGood, placedXsGood - preciseXsGood, placedYsGood - preciseYsGood, angles='xy', scale_units='xy', scale=1)
    ax3.plot(preciseXsGood, preciseYsGood, '.')
    ax3.plot(placedXsGood, placedYsGood, '.')
    ax3.set_title('Assignment from RawMap coordinates to final, quantized pixel coordinate')

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    smallVMask = ((placedXsGood - preciseXsGood) < 0.5) & ((placedYsGood - preciseYsGood) < 0.5)
    ax4.quiver(placedXsGood[smallVMask], placedYsGood[smallVMask], placedXsGood[smallVMask] - preciseXsGood[smallVMask], placedYsGood[smallVMask] - preciseYsGood[smallVMask], angles='xy', scale_units='xy', scale=None, width=0.001)
    ax4.set_title('Quiver Plot showing final_coordinates -> precise_coordinates (arrows not to scale)')    

    plt.show()
