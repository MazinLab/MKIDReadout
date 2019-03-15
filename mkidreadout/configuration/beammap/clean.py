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
from mkidreadout.configuration.beammap.utils import isInCorrectFL, getFLFromID, getFLFromCoords, isResonatorOnCorrectFeedline, generateCoords
from mkidcore.objects import Beammap
from mkidreadout.configuration.beammap import shift

MEC_FL_WIDTH = 14
DARKNESS_FL_WIDTH = 25

N_FL_MEC = 10
N_FL_DARKNESS = 5

logging.basicConfig()
log = logging.getLogger('beammap.clean')


def getOverlapGrid(xCoords, yCoords, flags, nXPix, nYPix):
    pixToUseMask = (flags == beamMapFlags['good']) | (flags == beamMapFlags['double'])
    xCoords = xCoords.astype(np.int)[pixToUseMask]
    yCoords = yCoords.astype(np.int)[pixToUseMask]
    bmGrid = np.zeros((nXPix, nYPix))
    for x,y in zip(xCoords, yCoords):
        if x >= 0 and y >= 0 and x < nXPix and y < nYPix:
            bmGrid[x, y] += 1
    return bmGrid

class BMCleaner(object):
    def __init__(self, beamMap, nRows, nCols, flip, instrument, designMapPath=None):
        self.nRows = nRows
        self.nCols = nCols
        self.flip = flip
        self.instrument = instrument.lower()
        self.designFile = designMapPath
        self.beamMap = beamMap.copy()
        self.preciseXs = self.beamMap.xCoords
        self.preciseYs = self.beamMap.yCoords

        if self.instrument.lower() == 'mec':
            self.nFL = N_FL_MEC
            self.flWidth = MEC_FL_WIDTH
        elif self.instrument.lower() == 'darkness':
            self.nFL = N_FL_DARKNESS
            self.flWidth = DARKNESS_FL_WIDTH
        else:
            raise Exception('Provided instrument not implemented!')
        
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
        self.flooredXs = self.beamMap.xCoords.astype(np.int)
        self.flooredYs = self.beamMap.yCoords.astype(np.int)
        self.placedXs = np.floor(self.beamMap.xCoords)
        self.placedYs = np.floor(self.beamMap.yCoords)
        self.bmGrid = getOverlapGrid(self.beamMap.xCoords, self.beamMap.yCoords, self.beamMap.flags.astype(int), self.nCols, self.nRows)


    def _fixInitialFeedlinePlacement(self, slack=1):
        wayOutMask = ~isInCorrectFL(self.beamMap.resIDs.astype(int), self.beamMap.xCoords, self.beamMap.yCoords, self.instrument, slack, self.flip)
        self.beamMap.flags.astype(int)[((self.beamMap.flags.astype(int) == beamMapFlags['good']) | (self.beamMap.flags.astype(int) == beamMapFlags['double'])) & wayOutMask] = beamMapFlags['wrongFeedline']
        self.beamMap.xCoords[self.beamMap.flags.astype(int) == beamMapFlags['wrongFeedline']] = np.nan
        self.beamMap.yCoords[self.beamMap.flags.astype(int) == beamMapFlags['wrongFeedline']] = np.nan
        log.info('%d pixels in wrong feedline. Flagged as bad', np.sum(self.beamMap.flags.astype(int)==beamMapFlags['wrongFeedline']))

        if slack > 0:
            flPos = getFLFromID(self.beamMap.resIDs.astype(int)) - 1
            if self.flip:
                flPos = 9 - flPos
            rightEdgeMask = flPos < getFLFromCoords(self.beamMap.xCoords, self.beamMap.yCoords, self.instrument, flip=False) - 1
            leftEdgeMask = flPos > getFLFromCoords(self.beamMap.xCoords, self.beamMap.yCoords, self.instrument, flip=False) - 1
            assert np.all((rightEdgeMask & leftEdgeMask) == 0), 'Error moving pixels to correct FL'
            log.info(str(np.sum(rightEdgeMask)+np.sum(leftEdgeMask)) + ' placed in correct feedline')
            if self.instrument == 'mec':
                self.beamMap.xCoords[rightEdgeMask] = (flPos[rightEdgeMask]+1)*self.flWidth-0.01
                self.beamMap.xCoords[leftEdgeMask] = (flPos[leftEdgeMask])*self.flWidth+0.01
            elif self.instrument == 'darkness':
                self.beamMap.yCoords[rightEdgeMask] = (flPos[rightEdgeMask]+1)*self.flWidth-0.01
                self.beamMap.yCoords[leftEdgeMask] = (flPos[leftEdgeMask])*self.flWidth+0.01
            else:
                raise Exception('Provided instrument not implemented!')
            validCoordMask = (~np.isnan(self.beamMap.xCoords)) & (~np.isnan(self.beamMap.yCoords))
            assert np.all(isInCorrectFL(self.beamMap.resIDs.astype(int)[validCoordMask], self.beamMap.xCoords[validCoordMask], self.beamMap.yCoords[validCoordMask], self.instrument, 0, self.flip)), 'bad FL cleanup!'


    def _fixOOBPixels(self, slack=1):
        wayOutMask = (self.beamMap.xCoords + slack < 0) | (self.beamMap.xCoords - slack > self.nCols - 1) | (self.beamMap.yCoords + slack < 0) | (self.beamMap.yCoords - slack > self.nRows - 1)
        self.beamMap.xCoords[wayOutMask] = np.nan
        self.beamMap.yCoords[wayOutMask] = np.nan
        self.beamMap.flags.astype(int)[wayOutMask] = beamMapFlags['wrongFeedline']

        self.beamMap.xCoords[(self.beamMap.xCoords < 0) & ~wayOutMask] = 0
        self.beamMap.xCoords[(self.beamMap.xCoords >= self.nCols) & ~wayOutMask] = self.nCols - 0.01
        self.beamMap.yCoords[(self.beamMap.yCoords < 0) & ~wayOutMask] = 0
        self.beamMap.yCoords[(self.beamMap.yCoords >= self.nRows) & ~wayOutMask] = self.nRows - 0.01

    
    def resolveOverlaps(self):
        '''
        Resolves overlaps out to one nearest neighbor. Results stored in self.placedXs and self.placedYs. self.bmGrid is also
        modified.
        '''
        overlapCoords = np.asarray(np.where(self.bmGrid > 1)).T
        # unoccupiedCoords = np.asarray(np.where(self.bmGrid == 0)).T
        nOverlapsResolved = 0

        for coord in overlapCoords:
            coordMask = (coord[0] == self.flooredXs) & (coord[1] == self.flooredYs) & ((self.beamMap.flags.astype(int) == beamMapFlags['good']) | (self.beamMap.flags.astype(int) == beamMapFlags['double']))
            coordInds = np.where(coordMask)[0] #indices of overlapping coordinates in beammap
    
            uONNCoords = np.asarray(np.where(self.bmGrid[coord[0]-1:coord[0]+2, coord[1]-1:coord[1]+2]==0)).T + coord - np.array([1,1])
            uONNCoords = uONNCoords[isInCorrectFL(self.beamMap.resIDs.astype(int)[coordInds[0]]*np.ones(len(uONNCoords)), uONNCoords[:,0], uONNCoords[:,1], self.instrument, 0, self.flip), :]
                
    
            precXOverlap = self.beamMap.xCoords[coordMask] - 0.5
            precYOverlap = self.beamMap.yCoords[coordMask] - 0.5
            precOverlapCoords = np.array(zip(precXOverlap, precYOverlap)) #list of precise coordinates overlapping with coord
    
            # no nearest neigbors, so pick the closest one and flag the rest as bad
            if len(uONNCoords)==0:
                distsFromCenter = (precOverlapCoords[:,0] - coord[0])**2 + (precOverlapCoords[:,1] - coord[1])**2
                minDistInd = np.argmin(distsFromCenter) #index in precOverlapCoords
                coordInds = np.delete(coordInds, minDistInd) #remove correct coordinate from overlap
                self.bmGrid[coord[0], coord[1]] = 1
                self.beamMap.flags.astype(int)[coordInds] = beamMapFlags['duplicatePixel']
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
                    self.beamMap.flags.astype(int)[coordInds] = beamMapFlags['duplicatePixel']
                    self.placedXs[coordInds] = np.nan
                    self.placedYs[coordInds] = np.nan
                    break
                    
        log.info('Successfully resolved %d overlaps', nOverlapsResolved)
        log.info('Failed to resolve %d overlaps', np.sum(self.beamMap.flags.astype(int)==beamMapFlags['duplicatePixel']))

    def resolveOverlapWithFrequency(self):
        """

        Uses the beammap that was given to the BMCleaner class to first run the shifting/frequency fitting code, then uses
        the unformation from that to resolve overlaps using the frequency information.

        Modifies the beammap in BMCleaner with a shift applied, overlaps resolved, and coordinates locked onto a grid.
        """
        if not hasattr(self.beamMap, 'frequencies'):
            raise Exception("This beammap does not have frequency data, this operation cannot be done")

        shiftObject = self.runShiftingCode()

        self.beamMap = shiftObject.shiftedBeammap
        designMap = shiftObject.designArray

        self.fixPreciseCoordinates()
        self.placeOnGrid()

        # self.placedXs = np.full(beammap.xCoords.shape, fill_value=np.nan)
        # self.placedYs = np.full(beammap.yCoords.shape, fill_value=np.nan)
        overlapCoords = np.asarray(np.where(self.bmGrid > 1)).T
        nOverlapsResolved = 0
        nPixelsPlaced = 0
        originalCoord = []
        placedCoord = []

        for coord in overlapCoords:
            doubles = self.beamMap.getResonatorsAtCoordinate(coord[0], coord[1])

            # Creates an list of coordinates to try to place the members of the overlap on. Creates a square around the
            # coordinate of the overlap, then removes all locations in the wrong feedline, off the array, or that are
            # already occupied. Adds back the coordinate of the overlap, because one of them could be placed there.
            xUncertainty = 0
            yUncertainty = 0
            coordsToSearch = generateCoords(coord, xUncertainty, yUncertainty)
            nCoords = len(coordsToSearch)
            while nCoords < len(doubles):
                xUncertainty += 1
                yUncertainty += 1
                coordsToSearch = generateCoords(coord, xUncertainty, yUncertainty)

                # First masks, removes search coordinates that are off of the array or the feedline
                onArrayMask = ((coordsToSearch[:, 0] >= 0) & (coordsToSearch[:, 0] < self.nCols) & (coordsToSearch[:, 1] >= 0) & (coordsToSearch[:, 1] < self.nRows))
                coordsToSearch = coordsToSearch[onArrayMask.astype(bool)]
                onFeedlineMask = np.zeros(len(coordsToSearch))
                for i in range(len(coordsToSearch)):
                    onFeedlineMask[i] = isResonatorOnCorrectFeedline(doubles[0][0], coordsToSearch[i][0], coordsToSearch[i][1], self.instrument)
                coordsToSearch = coordsToSearch[onFeedlineMask.astype(bool)]

                # Mask for if any of the search coordinates already have a resonator there
                occupationMask = np.zeros(len(coordsToSearch))
                for i in range(len(coordsToSearch)):
                    if self.bmGrid[coordsToSearch[i][0], coordsToSearch[i][1]] == 0:
                        occupationMask[i] = True
                    else:
                        occupationMask[i] = False
                coordsToSearch = coordsToSearch[occupationMask.astype(bool)]

                # Adds the overlap coordinate back to the search coords (they can be placed there)
                coordsToSearch = np.append(coordsToSearch, coord).reshape((len(coordsToSearch)+1, 2))
                nCoords = len(coordsToSearch)

            # Generates the design frequency at the coordinates that are being tested
            freqsToSearch = list(designMap.getDesignFrequencyFromCoords(coordinate) for coordinate in coordsToSearch) # Design frequencies at each of the search coordinates
            # Creates the possible combinations to place the resonators at
            possiblePlacements = itertools.permutations((range(len(freqsToSearch))), len(doubles))
            placementsToSearch = [i for i in possiblePlacements]
            residuals = np.zeros(len(placementsToSearch))

            # Generates the 'total frequency residual' of all resonators in the possible configurations
            for i in range(len(placementsToSearch)):
                for j in range(len(doubles)):
                    residuals[i] += abs(doubles[j][4]-freqsToSearch[placementsToSearch[i][j]])

            # Finds where the 'total frequency residual' occurs. This gives the placement combination which minimizes
            # the difference in frequency between the pixel placement and design frequency at those points
            index = np.where(residuals == min(residuals))[0][0]
            bestPlacement = placementsToSearch[index]

            # Generates the coordinates at which each resonator will be placed
            newCoordinates = np.zeros((len(doubles), 2))
            for i in range(len(bestPlacement)):
                newCoordinates[i] = coordsToSearch[bestPlacement[i]]

            # Updates the overlap grid to reflect the new occupancy of coordinates. Updates the in-function beammap.
            # These updates will be applied to the class beammap
            self.bmGrid[coord[0], coord[1]] = 0
            for i in range(len(doubles)):
                resonator = doubles[i]
                newCoordinate = newCoordinates[i].astype(int)
                originalCoord.append(coord)
                placedCoord.append(newCoordinate)
                self.bmGrid[newCoordinate[0], newCoordinate[1]] += 1
                index = np.where(self.beamMap.resIDs == resonator[0])[0]
                self.beamMap.xCoords[index] = newCoordinate[0]
                self.beamMap.yCoords[index] = newCoordinate[1]
                self.beamMap.flags[index] = beamMapFlags['good']
                nPixelsPlaced += 1

            nOverlapsResolved += 1

        log.info('Successfully resolved %d overlaps', nOverlapsResolved)
        log.info('Successfully placed %d pixels', nPixelsPlaced)

        self.placedXs = self.beamMap.xCoords
        self.placedYs = self.beamMap.yCoords

        # For all of the resonators that were analyzed in this part of the code, figure out where they started (after
        # being shifted in space) and where they were moved (or not moved to).
        original, placed = np.array(originalCoord), np.array(placedCoord)
        distanceMoved = np.sqrt(((original[:, 0]-placed[:, 0])**2)+((original[:, 1]-placed[:, 1])**2))
        didMove = (distanceMoved != 0)
        within1 = (distanceMoved < 2)
        nPixelsWithin1 = distanceMoved[within1.astype(bool)]


        # Plots the shifted, locked onto a grid original location of the resolved overlaps and the lovations the
        # pixels were placed at. Red dot = pixel was not moved, Blue arrow points from the overlap to where the
        # pixel was placed
        log.info('%d pixels were not moved', len(distanceMoved[~didMove.astype(bool)]))
        log.info('%d pixels were moved', len(distanceMoved[didMove.astype(bool)]))
        log.info('%d pixels were placed within 1 pixel of their original placement', nPixelsWithin1)

        # Plots the 'process' of overlap resolution. Arrows start where the pixel began to where it was placed. Dots
        # are shown where the 'best placement' for a resonator that was not moved.
        # NOTE: If a pixel has a dot, it MUST also have an arrow coming from it.
        # plt.scatter(original[:, 0][~didMove.astype(bool)], original[:, 1][~didMove.astype(bool)], c='green', marker='.')
        # plt.quiver(original[:, 0][didMove.astype(bool)], original[:, 1][didMove.astype(bool)],
        #            (placed[:, 0]-original[:, 0])[didMove.astype(bool)], (placed[:, 1]-original[:, 1])[didMove.astype(bool)],
        #            color='blue', angles='xy', scale_units='xy', scale=1, headlength=3, headwidth=2)
        # plt.show(block=False)

    def runShiftingCode(self):
        shifter = shift.BeammapShifter(self.designFile, self.beamMap, self.instrument)
        shifter.run()
        return shifter  # This is an object which contains a beammap that will be used in future beammap cleaning steps

    def placeFailedPixels(self):
        '''
        Places all bad pixels (NaN coordinates) in arbitrary locations on correct feedline. Should ensure
        1-to-1 mapping between resID and coordinates.
        '''
        toPlaceMask = np.isnan(self.placedXs) | np.isnan(self.placedYs)
        unoccupiedCoords = np.asarray(np.where(self.bmGrid==0)).T
        
        for i in range(self.nFL):
            toPlaceMaskCurFL = toPlaceMask & ((self.beamMap.resIDs.astype(int)/10000 - 1) == i)
            unoccupiedCoordsCurFL = unoccupiedCoords[isInCorrectFL(10000*(i+1)*np.ones(len(unoccupiedCoords)), unoccupiedCoords[:, 0], unoccupiedCoords[:,1], self.instrument, flip=self.flip)]
            self.placedXs[toPlaceMaskCurFL] = unoccupiedCoordsCurFL[:, 0]
            self.placedYs[toPlaceMaskCurFL] = unoccupiedCoordsCurFL[:, 1]

    def placeFailedPixelsQuick(self): 
        '''
        Places all bad pixels just outside array bounds. Use when placeFailedPixels fails but you need a 
        quick beammap without NaNs!
        '''
        toPlaceXs = np.isnan(self.placedXs) 
        toPlaceYs = np.isnan(self.placedYs)
        self.placedXs[toPlaceXs] = self.nCols
        self.placedYs[toPlaceYs] = self.nRows

    def saveBeammap(self, path):
        '''
        Saves beammap in standard 4 column text file format
            INPUTS:
                path - full path of beammap file
        '''
        assert np.all(np.isnan(self.placedXs)==False), 'NaNs in final beammap!'
        assert np.all(np.isnan(self.placedYs)==False), 'NaNs in final beammap!'
        log.info('N good pixels: ' + str(np.sum(self.beamMap.flags.astype(int)==beamMapFlags['good'])))
        log.info('N doubles: ' + str(np.sum(self.beamMap.flags.astype(int)==beamMapFlags['double'])))
        log.info('N failed pixels (read out but not beammapped): ' + str(np.sum((self.beamMap.flags.astype(int)!=beamMapFlags['good']) & (self.beamMap.flags.astype(int)!=beamMapFlags['double']) & (self.beamMap.flags.astype(int)!=beamMapFlags['noDacTone']))))
        np.savetxt(path, np.transpose([self.beamMap.resIDs.astype(int), self.beamMap.flags.astype(int), self.placedXs.astype(int), self.placedYs.astype(int)]), fmt='%4i %4i %4i %4i')
     

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('cfgFile', nargs=1, type=str, default='/mnt/data0/MEC/20181218/clean.cfg', help='Configuration file')
    #default config file location
    args = parser.parse_args()
    configFileName = args.cfgFile[0]
    resolutionType = args
    log.setLevel(logging.INFO)

    configData = readDict()
    configData.read_from_file(configFileName)
    beammapDirectory = configData['beammapDirectory']
    finalBMFile = configData['finalBMFile']
    rawBMFile = configData['rawBMFile']
    useFreqs = configData['useFreqs']
    psFiles = configData['powersweeps']
    designFile = configData['designMapFile']
    numRows = configData['numRows']
    numCols = configData['numCols']
    flipParam = configData['flip']
    inst = configData['instrument']

    rawBeamMap = Beammap(rawBMFile, (146, 140), 'MEC')
    if useFreqs:
        rawBeamMap.loadFrequencies(psFiles)
        cleaner = BMCleaner(beamMap=rawBeamMap, nRows=numRows, nCols=numCols,
                            flip=flipParam, instrument=inst, designMapPath=designFile)
    else:
        cleaner = BMCleaner(beamMap=rawBeamMap, nRows=numRows, nCols=numCols,
                            flip=flipParam, instrument=inst, designMapPath=designFile)
    cleaner.fixPreciseCoordinates()
    cleaner.placeOnGrid()

    #plt.ion()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.imshow(cleaner.bmGrid.T)
    ax1.set_title('Initial (floored) beammap (value = res per pix coordinate)')

    # resolve overlaps and place failed pixels

    if useFreqs:
        cleaner.resolveOverlapWithFrequency()
        cleaner.placeFailedPixels()
        cleaner.saveBeammap(finalBMFile)
    else:
        cleaner.resolveOverlaps()
        cleaner.placeFailedPixels()
        cleaner.saveBeammap(finalBMFile)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.imshow(cleaner.bmGrid.T)
    ax2.set_title('Beammap after fixing overlaps (value = res per pix coordinate)')

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    goodPixMask = (cleaner.beamMap.flags==beamMapFlags['good']) | (cleaner.beamMap.flags==beamMapFlags['double'])
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