'''
Automates the functionality in the pixels_movingscan GUI. Finds the optimal scale, angle, and
offset from the raw beammap data, applies these, and saves the beammap file. Note that cleanBeammap.py 
should still be run after this. 

Author: Neelay Fruitwala

Usage: python alignGrid.py <configFile>
    configFile is almost identical to the GridConfig.dict file used by pixels_movingscan - it specifies
    the master beamlist and doubles lists that come out of the clickthrough GUI, as well as the raw beammap
    output files. The only difference is that it has parameters for nXPix and nYPix.

'''

import os, sys
import numpy as np
import scipy.ndimage as sciim
import matplotlib.pyplot as plt
from readDict import readDict

class BMAligner:

    def __init__(self, beamListFn, nXPix, nYPix, usFactor=10):
        self.beamListFn = beamListFn
        self.usFactor = usFactor
        self.nXPix = nXPix
        self.nYPix = nYPix
        self.resIDs, self.flags, self.rawXs, self.rawYs = np.loadtxt(beamListFn, unpack=True)
        self.makeRawImage()

    def makeRawImage(self):
        self.rawImage = np.zeros((int(np.max(self.rawXs)*self.usFactor+1), int(np.max(self.rawYs)*self.usFactor+1)))
        for i, resID in enumerate(self.resIDs):
            if self.flags[i] == 0:
                self.rawImage[int(round(self.rawXs[i]*self.usFactor)), int(round(self.rawYs[i]*self.usFactor))] = 1

    def fftRawImage(self):
        self.rawImageFFT = np.abs(np.fft.fft2(self.rawImage))
        self.rawImageFreqs = [np.fft.fftfreq(self.rawImageFFT.shape[0]), np.fft.fftfreq(self.rawImageFFT.shape[1])]
    
    def findAngleAndScale(self):
        maxFiltFFT = sciim.maximum_filter(self.rawImageFFT, size=10)
        locMaxMask = (self.rawImageFFT==maxFiltFFT)
        
        maxInds = np.argsort(self.rawImageFFT, axis=None)
        maxInds = maxInds[::-1]

        # find four nonzero k-vectors
        nMaxFound = 0
        i = 0
        kvecList = np.zeros((2,2))
        while nMaxFound < 2:
            maxCoords = np.unravel_index(maxInds[i], self.rawImageFFT.shape)

            if locMaxMask[maxCoords]==1 and maxCoords!=(0,0):
                kvec = np.array([self.rawImageFreqs[0][maxCoords[0]], self.rawImageFreqs[1][maxCoords[1]]])
                if nMaxFound == 0:
                    kvecList[0] = kvec
                    nMaxFound += 1
                else:
                    if np.abs(np.dot(kvecList[0], kvec)) < 0.05*np.linalg.norm(kvec)**2:
                        kvecList[1] = kvec
                        nMaxFound += 1
                    

            i += 1
                
        xKvecInd = np.argmax(np.abs(kvecList[:,0]))
        yKvecInd = np.argmax(np.abs(kvecList[:,1]))

        assert xKvecInd!=yKvecInd, 'x and y kvecs are the same!'

        xKvec = kvecList[xKvecInd]
        yKvec = kvecList[yKvecInd]

        if xKvec[0]<0:
            xKvec *= -1

        if yKvec[1]<0:
            yKvec *= -1


        anglex = np.arctan2(xKvec[1], xKvec[0])
        angley = np.arctan2(yKvec[1], yKvec[0]) - np.pi/2

        assert (anglex - angley)/anglex < 0.01, 'x and y kvecs are not perpendicular!'

        self.angle = (anglex + angley)/2
        self.xScale = 1/(self.usFactor*np.linalg.norm(xKvec))
        self.yScale = 1/(self.usFactor*np.linalg.norm(yKvec))

        print 'angle:', self.angle
        print 'x scale:', self.xScale
        print 'y scale:', self.yScale
        
    def rotateAndScaleCoords(self, xVals=None, yVals=None):
        c = np.cos(-self.angle)
        s = np.sin(-self.angle)
        rotMat = np.array([[c, -s], [s, c]])
        if xVals is None:
            rawCoords = np.stack((self.rawXs, self.rawYs))
        else:
            rawCoords = np.stack((xVals, yVals))
        coords = np.dot(rotMat,rawCoords)
        coords = np.transpose(coords)
        coords[:,0] /= self.xScale
        coords[:,1] /= self.yScale
        if xVals is None:
            self.coords = coords
        else:
            return coords

    def findOffset(self, nMCIters=10000, maxSampDist=10, roundCoords=False):
        # find a good starting point for search, using median of 100 minimum "good" points
        goodInds = np.where(self.flags==0)[0]
        sortedX = np.sort(self.coords[goodInds,0])
        sortedY = np.sort(self.coords[goodInds,1])
        baselineXOffs = np.median(sortedX[:self.nXPix*3/4])
        baselineYOffs = np.median(sortedY[:self.nYPix*3/4])
        print 'Baseline X Offset:', baselineXOffs
        print 'Baseline Y Offset:', baselineYOffs
        curXOffs = baselineXOffs
        curYOffs = baselineYOffs
        optXOffs = curXOffs
        optYOffs = curYOffs
        startXOffs = baselineXOffs
        startYOffs = baselineYOffs
        optNGoodPix = 0
        optI = 0

        optSearchIters = 1000 #after this many iters around max go back to baseline

        # do monte-carlo search around the baseline to find which offset has the most filled in pixels
        for i in range(nMCIters):
            if roundCoords:
                shiftedXs = np.round(self.coords[:,0] - curXOffs).astype(int)
                shiftedYs = np.round(self.coords[:,1] - curYOffs).astype(int)
            else:
                shiftedXs = (self.coords[:,0] - curXOffs).astype(int)
                shiftedYs = (self.coords[:,1] - curYOffs).astype(int)
                
            validInds = np.where(np.logical_and(np.logical_and(shiftedXs>0, shiftedXs<self.nXPix), 
                np.logical_and(shiftedYs>0, shiftedYs<self.nYPix)))[0]
            shiftedXs = shiftedXs[validInds]
            shiftedYs = shiftedYs[validInds]
            goodPixMask = np.zeros((self.nXPix, self.nYPix))
            goodPixCoords = (shiftedXs, shiftedYs)
            goodPixMask[goodPixCoords] = 1
            nGoodPix = np.sum(goodPixMask)
            if nGoodPix > optNGoodPix:
                optNGoodPix = nGoodPix
                optXOffs = curXOffs
                optYOffs = curYOffs
                optI = i
                startXOffs = optXOffs
                startYOffs = optYOffs
                print 'Found new optimum at', optXOffs, optYOffs, 'with', optNGoodPix, 'good Pixels. i =', i
            if i - optI > optSearchIters: #search around maximum for a bit then go back to baseline
                startXOffs = baselineXOffs
                startYOffs = baselineYOffs
            curXOffs = startXOffs + 2*maxSampDist*np.random.random() - maxSampDist
            curYOffs = startYOffs + 2*maxSampDist*np.random.random() - maxSampDist

        self.xOffs = optXOffs
        self.yOffs = optYOffs

        print 'Optimal offset:', self.xOffs, self.yOffs

        #if roundCoords:
        #    self.coords[:,0] = (np.round(self.coords[:,0] - self.xOffs)).astype(int)
        #    self.coords[:,1] = (np.round(self.coords[:,1] - self.yOffs)).astype(int)
        #else:
        #    self.coords[:,0] = (self.coords[:,0] - self.xOffs).astype(int)
        #    self.coords[:,1] = (self.coords[:,1] - self.yOffs).astype(int)
        self.coords[:,0] = (self.coords[:,0] - self.xOffs)
        self.coords[:,1] = (self.coords[:,1] - self.yOffs)

    def applyOffset(self, coords):
        newCoords = np.zeros(coords.shape)
        newCoords[:,0] = coords[:,0] - self.xOffs
        newCoords[:,1] = coords[:,1] - self.yOffs
        return newCoords

    def saveRawMap(self, rawMapFn):
        np.savetxt(rawMapFn, np.transpose([self.resIDs, self.flags, self.coords[:,0], self.coords[:,1]]), fmt='%4i %4i %.5f %.5f')

    def makeDoublesRawMap(self, doublesListFn, rawDoublesMapFn):
        resID, x1, y1, x2, y2 = np.loadtxt(doublesListFn, unpack=True)
        coords1 = self.rotateAndScaleCoords(x1, y1)
        coords2 = self.rotateAndScaleCoords(x2, y2)
        coords1 = self.applyOffset(coords1)
        coords2 = self.applyOffset(coords2)
        np.savetxt(rawDoublesMapFn, np.transpose([resID, coords1[:,0], coords1[:,1], 
            coords2[:,0], coords2[:,1]]), fmt='%4i %.5f %.5f %.5f %.5f')
        
    def plotCoords(self):
        goodInds = np.where(self.flags==0)[0]
        badInds = np.where(self.flags!=0)[0]
        plt.plot(self.coords[goodInds,0], self.coords[goodInds,1], '.', color='b')
        plt.plot(self.coords[badInds,0], self.coords[badInds,1], '.', color='r')
        plt.show()
            
if __name__=='__main__':
    if len(sys.argv)<2:
        print 'Usage: "python alignGrid.py <configFile>", where <configFile> is in MKID_DATA_DIR'
        exit(1)

    mdd = os.environ['MKID_DATA_DIR']
    cfgFn = os.path.join(mdd, sys.argv[1])
    paramDict = readDict()
    paramDict.read_from_file(cfgFn)

    aligner = BMAligner(paramDict['masterPositionList'], paramDict['nXPix'], paramDict['nYPix'])
    aligner.makeRawImage()
    aligner.fftRawImage()
    aligner.findAngleAndScale()
    aligner.rotateAndScaleCoords()
    aligner.findOffset()
    aligner.plotCoords()
    aligner.saveRawMap(paramDict['outputFilename'])
    aligner.makeDoublesRawMap(paramDict['masterDoublesList'], paramDict['outputDoubleName'])



                
            
