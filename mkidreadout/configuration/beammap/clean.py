"""
cleanV2.py

Takes rough beammap with overlapping pixels and pixels out of bounds
Reconciles doubles by placing them in nearest empty pix
Places o.o.b pixels in random position, changes flag to 1
Places failed pixels randomly.
Outputs final cleaned beammap with ID, flag, x, y; ready to go into dashboard

"""

from __future__ import print_function
import numpy as np
import os
import logging
import argparse
import ipdb
import matplotlib.pyplot as plt
from mkidreadout.utils.arrayPopup import plotArray
from mkidreadout.utils.readDict import readDict
from mkidreadout.configuration.beammap.flags import beamMapFlags
from mkidreadout.configuration.beammap.utils import isInCorrectFL, getFLFromID, getFLFromCoords

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
    def __init__(self, roughBMData, nRows, nCols, flip, instrument):
        self.nRows = int(configData['numRows'])
        self.nCols = int(configData['numCols'])
        self.flip = bool(configData['flip'])
        self.roughBMFile = str(configData['roughBMFile'])
        self.finalBMFile = str(configData['finalBMFile'])
        self.beammapDirectory = str(configData['beammapDirectory'])
        self.instrument = str(configData['instrument']).lower()

        if self.instrument.lower()=='mec':
            self.nFL = N_FL_MEC
            self.flWidth = MEC_FL_WIDTH
        elif self.instrument.lower()=='darkness':
            self.nFL = N_FL_DARKNESS
            self.flWidth = DARKNESS_FL_WIDTH
    
        self.resIDs = np.array(roughBM[:,0],dtype=np.int)
        self.flags = np.array(roughBM[:,1],dtype=np.int)
        self.preciseXs = roughBM[:,2]
        self.preciseYs = roughBM[:,3]
        self.flooredXs = None
        self.flooredYs = None
        self.placedXs = None
        self.placedYs = None
        self.bmGrid = None

    def fixPreciseCoordinates(self):
        #fix coordinates
        self._fixOOBPixels()
        self._fixInitialFeedlinePlacement()


    def placeOnGrid(self):
        '''
        Places all resonators on nRowsxnCols size grid. Only do this after precise coordinates are finalized
        '''
        #lock in coordinates, set up placement tools/arrays
        self.flooredXs = self.preciseXs.astype(np.int)
        self.flooredYs = self.preciseYs.astype(np.int)
        self.placedXs = np.floor(self.preciseXs)
        self.placedYs = np.floor(self.preciseYs)
        self.bmGrid = getOverlapGrid(self.preciseXs, self.preciseYs, self.flags, self.nCols, self.nRows)


    def _fixInitialFeedlinePlacement(self, slack=0):
        wayOutMask = ~isInCorrectFL(self.resIDs, self.preciseXs, self.preciseYs, self.instrument, slack, self.flip)
        self.flags[((self.flags==beamMapFlags['good']) | (self.flags==beamMapFlags['double'])) & wayOutMask] = beamMapFlags['wrongFeedline']
        self.preciseXs[self.flags==beamMapFlags['wrongFeedline']] = np.nan
        self.preciseYs[self.flags==beamMapFlags['wrongFeedline']] = np.nan
        log.info('%d pixels in wrong feedline. Flagged as bad', np.sum(self.flags==beamMapFlags['wrongFeedline']))

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
            #if coord[0]==61 and coord[1]==22:
            #    ipdb.set_trace()
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

    def placeFailedPixels(self):
        toPlaceMask = np.isnan(self.placedXs) | np.isnan(self.placedYs)
        unoccupiedCoords = np.asarray(np.where(self.bmGrid==0)).T
        
        for i in range(self.nFL):
            toPlaceMaskCurFL = toPlaceMask & ((self.resIDs/10000 - 1) == i)
            unoccupiedCoordsCurFL = unoccupiedCoords[isInCorrectFL(10000*(i+1)*np.ones(len(unoccupiedCoords)), unoccupiedCoords[:,0], unoccupiedCoords[:,1], self.instrument, flip=self.flip)]
            self.placedXs[toPlaceMaskCurFL] = unoccupiedCoordsCurFL[:,0]
            self.placedYs[toPlaceMaskCurFL] = unoccupiedCoordsCurFL[:,1]

    def placeFailedPixelsQuick(self): 
        toPlaceXs = np.isnan(self.placedXs) 
        toPlaceYs = np.isnan(self.placedYs)
        self.placedXs[toPlaceXs] = self.nCols
        self.placedYs[toPlaceYs] = self.nRows
    
    def saveBeammap(self, path):
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
    roughBMFile = configData['roughBMFile']
    finalBMFile = configData['finalBMFile']

    #put together full input/output BM paths
    roughPath = os.path.join(beammapDirectory,roughBMFile)
    finalPath = os.path.join(beammapDirectory,finalBMFile)
    
    #load location data from rough BM file
    roughBM = np.loadtxt(roughPath)
   
    cleaner = BMCleaner(roughBM, configData['numRows'], configData['numCols'], configData['flip'], configData['instrument'])

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
    #cleaner.placeFailedPixelsQuick()
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
            
    


