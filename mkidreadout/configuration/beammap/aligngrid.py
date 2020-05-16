"""
Automates the functionality in the pixels_movingscan GUI. Finds the optimal scale, angle, and
offset from the temporal beammap data, applies these, and saves the beammap file. Note that clean.py
should still be run after this.

Algorithm outline:
    1. Take an FFT of the lattice of temporal pixel coordinates (x/y sweep timestamps from sweep.py)
    2. Find the fundamental frequencies in x and y planes (click on them w/ GUI)
    3. Use these to determine scale factor and angle
TODO: implement skew

Author: Neelay Fruitwala

Usage: python alignGrid.py <configFile>
    configFile is almost identical to the GridConfig.dict file used by pixels_movingscan - it specifies
    the master beamlist and doubles lists that come out of the clickthrough GUI, as well as the temporal beammap
    output files. The only difference is that it has parameters for nXPix and nYPix.

"""
from __future__ import print_function

import os
import sys
import warnings

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sciim

from mkidcore.instruments import DEFAULT_ARRAY_SIZES
from mkidcore.config import load
from mkidcore.corelog import getLogger
from mkidcore.pixelflags import beammap as beamMapFlags
from mkidreadout.configuration.beammap.utils import DARKNESS_FL_WIDTH, MEC_FL_WIDTH, N_FL_DARKNESS, N_FL_MEC, \
    getFLFromID, isInCorrectFL


class BMAligner(object):
    def __init__(self, beamdir, beamListFn, cachename, instrument, flip=False, usFactor=50):
        self.beamListFn = os.path.join(beamdir, beamListFn)
        self.usFactor = usFactor
        self.instrument = instrument.lower()
        self.flip = flip
        self.resIDs, self.flags, self.temporalXs, self.temporalYs = np.loadtxt(self.beamListFn, unpack=True)
        self.makeTemporalImage()
        self.temporalImage = None
        self.temporalImageFFT = None
        self.temporalImageFreqs = None
        self.temporalImageFFTFile = os.path.join(beamdir, cachename)

        self.nXPix, self.nYPix = DEFAULT_ARRAY_SIZES[config.beammap.instrument.lower()]

        if instrument.lower()=='mec':
            self.flWidth = MEC_FL_WIDTH
            self.nFL = N_FL_MEC
        elif instrument.lower()=='darkness':
            self.flWidth = DARKNESS_FL_WIDTH
            self.nFL = N_FL_DARKNESS

        else:
            raise Exception('Instrument ' + instrument + ' not implemented yet!')

    def makeTemporalImage(self):
        self.temporalImage = np.zeros((int(np.max(self.temporalXs[np.where(np.isfinite(self.temporalXs))])*self.usFactor+2), int(np.max(self.temporalYs[np.where(np.isfinite(self.temporalYs))])*self.usFactor+2)))
        for i, resID in enumerate(self.resIDs):
            if self.flags[i] == beamMapFlags['good']:
                if np.isfinite(self.temporalXs[i]) and np.isfinite(self.temporalYs[i]):
                    self.temporalImage[int(round(self.temporalXs[i]*self.usFactor)), int(round(self.temporalYs[i]*self.usFactor))] = 1

    def fftTemporalImage(self, save=True):
        getLogger(__name__).info('Computing FFT of temporal image')
        self.temporalImageFFT = np.abs(np.fft.fft2(self.temporalImage))
        self.temporalImageFreqs = [np.fft.fftfreq(self.temporalImageFFT.shape[0]), np.fft.fftfreq(self.temporalImageFFT.shape[1])]
        if save:
            np.savez(self.temporalImageFFTFile,
                     temporalImageFFT=self.temporalImageFFT, temporalImageFreqs=self.temporalImageFreqs)

    def loadFFT(self, path=None):
        if path is None:
            path = self.temporalImageFFTFile
        if not os.path.exists(path):
            self.fftTemporalImage()

        getLogger(__name__).info('Loading FFT from ' + str(path))
        fftDict = np.load(path, allow_pickle=True)
        try:
            self.temporalImageFFT = fftDict['temporalImageFFT']
            self.temporalImageFreqs = fftDict['temporalImageFreqs']
        except KeyError:
            getLogger(__name__).warning('FFT file has old format variable names')
            self.temporalImageFFT = fftDict['rawImageFFT']
            self.temporalImageFreqs = fftDict['rawImageFreqs']
    
    def findKvecsAuto(self):
        maxFiltFFT = sciim.maximum_filter(self.temporalImageFFT, size=10)
        locMaxMask = (self.temporalImageFFT==maxFiltFFT)
        
        maxInds = np.argsort(self.temporalImageFFT, axis=None)
        maxInds = maxInds[::-1]

        # find four nonzero k-vectors
        nMaxFound = 0
        i = 0
        kvecList = np.zeros((2,2))
        while nMaxFound < 2:
            maxCoords = np.unravel_index(maxInds[i], self.temporalImageFFT.shape)
            kvec = np.array([self.temporalImageFreqs[0][maxCoords[0]], self.temporalImageFreqs[1][maxCoords[1]]])

            if locMaxMask[maxCoords]==1 and (kvec[0]**2+kvec[1]**2)>0.05:
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

        self.xKvec = xKvec
        self.yKvec = yKvec

    def findKvecsManual(self):
        shiftedFreqs = [0, 0]
        shiftedFreqs[0] = np.fft.fftshift(self.temporalImageFreqs[0])
        shiftedFreqs[1] = np.fft.fftshift(self.temporalImageFreqs[1])
        shiftedImage = np.fft.fftshift(self.temporalImageFFT)

        kvecGui = KVecGUI(shiftedImage, shiftedFreqs, 30*self.usFactor)
        self.xKvec = kvecGui.kx
        self.yKvec = kvecGui.ky

    def findAngleAndScale(self):
        anglex = np.arctan2(self.xKvec[1], self.xKvec[0])
        angley = np.arctan2(self.yKvec[1], self.yKvec[0]) - np.pi/2

        #assert (anglex - angley)/anglex < 0.01, 'x and y kvecs are not perpendicular!'

        self.angle = (anglex + angley)/2
        self.xScale = 1/(self.usFactor*np.linalg.norm(self.xKvec))
        self.yScale = 1/(self.usFactor*np.linalg.norm(self.yKvec))

        getLogger(__name__).info('angle: {}'.format(self.angle))
        getLogger(__name__).info('x scale: {}'.format(self.xScale))
        getLogger(__name__).info('y scale: {}'.format(self.yScale))

    def rotateAndScaleCoords(self, xVals=None, yVals=None):
        c = np.cos(-self.angle)
        s = np.sin(-self.angle)
        rotMat = np.array([[c, -s], [s, c]])
        if xVals is None:
            temporalCoords = np.stack((self.temporalXs, self.temporalYs))
        else:
            temporalCoords = np.stack((xVals, yVals))
        coords = np.dot(rotMat,temporalCoords)
        coords = np.transpose(coords)
        coords[:,0] /= self.xScale
        coords[:,1] /= self.yScale
        if xVals is None:
            self.coords = coords
        else:
            return coords

    def findOffset(self, nMCIters=10000, maxSampDist=50, roundCoords=False):
        # find a good starting point for search, using median of 100 minimum "good" points
        if self.instrument.lower() == 'mec' and self.flip:
            self.coords[:,0] = -self.coords[:,0]
            self.coords[:,1] = -self.coords[:,1] #flip y too to keep beammap consistent
        elif self.instrument.lower() == 'darkness' and self.flip:
            self.coords[:,1] = -self.coords[:,1]

        goodMask = self.flags==0
        goodResIDs = self.resIDs[goodMask]
        goodCoords = self.coords[goodMask,:]
        sortedXInds = np.argsort(goodCoords[:,0])
        sortedYInds = np.argsort(goodCoords[:,1])
        sortedX = goodCoords[sortedXInds,0]
        sortedY = goodCoords[sortedYInds,1]

        if self.instrument.lower() == 'mec':
            yStart = 0
            firstFL = getFLFromID(goodResIDs[sortedXInds[100]]) #remove outliers - we expect at least 500 res per FL
            xStart = (firstFL - 1)*self.flWidth
            getLogger(__name__).info('xStart: {}'.format(xStart))

        elif self.instrument.lower() == 'darkness':
            xStart = 0
            firstFL = getFLFromID(goodResIDs[sortedYInds[100]])
            yStart = (firstFL - 1)*self.flWidth

        baselineXOffs = np.median(sortedX[:self.nXPix*3/4]) - xStart
        baselineYOffs = np.median(sortedY[:self.nYPix*3/4]) - yStart
        getLogger(__name__).info('Baseline X Offset: {}'.format(baselineXOffs))
        getLogger(__name__).info('Baseline Y Offset: {}'.format(baselineYOffs))
        curXOffs = baselineXOffs
        curYOffs = baselineYOffs
        optXOffs = curXOffs
        optYOffs = curYOffs
        startXOffs = baselineXOffs
        startYOffs = baselineYOffs
        optNGoodPix = 0
        optI = 0

        optSearchIters = 2000 #after this many iters around max go back to baseline

        # do monte-carlo search around the baseline to find which offset has the most filled in pixels
        for i in range(nMCIters):
            if roundCoords:
                shiftedXs = np.round(self.coords[:,0] - curXOffs).astype(int)
                shiftedYs = np.round(self.coords[:,1] - curYOffs).astype(int)
            else:
                shiftedXs = (self.coords[:,0] - curXOffs).astype(int)
                shiftedYs = (self.coords[:,1] - curYOffs).astype(int)
                
            validMask = isInCorrectFL(self.resIDs, shiftedXs, shiftedYs, self.instrument, flip=False)
            validMask = validMask & (shiftedXs>=0) & (shiftedXs<self.nXPix) & (shiftedYs>=0) & (shiftedYs<self.nYPix)
                
            shiftedXs = shiftedXs[validMask]
            shiftedYs = shiftedYs[validMask]
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
                msg = 'Found new optimum at {:.1f}, {:.1f} with {:.0f} good pixles. i={}'
                getLogger(__name__).info(msg.format(optXOffs, optYOffs, optNGoodPix, i))
            if i - optI > optSearchIters: #search around maximum for a bit then go back to baseline
                startXOffs = baselineXOffs
                startYOffs = baselineYOffs
            curXOffs = startXOffs + 2*maxSampDist*np.random.random() - maxSampDist
            curYOffs = startYOffs + 2*maxSampDist*np.random.random() - maxSampDist

        self.xOffs = optXOffs
        self.yOffs = optYOffs

        getLogger(__name__).info('Optimal offset: {:.1f}, {:.1f}'.format(self.xOffs, self.yOffs))

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

    def saveTemporalMap(self, temporalMapFn):
        np.savetxt(temporalMapFn, np.transpose([self.resIDs, self.flags, self.coords[:,0], self.coords[:,1]]), fmt='%4i %4i %.5f %.5f')

    def makeDoublesTemporalMap(self, doublesListFn, temporalDoublesMapFn):
        resID, x1, y1, x2, y2 = np.loadtxt(doublesListFn, unpack=True)
        coords1 = self.rotateAndScaleCoords(x1, y1)
        coords2 = self.rotateAndScaleCoords(x2, y2)
        coords1 = self.applyOffset(coords1)
        coords2 = self.applyOffset(coords2)
        np.savetxt(temporalDoublesMapFn, np.transpose([resID, coords1[:,0], coords1[:,1],
            coords2[:,0], coords2[:,1]]), fmt='%4i %.5f %.5f %.5f %.5f')
        
    def plotCoords(self):
        goodInds = np.where(self.flags==0)[0]
        badInds = np.where(self.flags!=0)[0]
        plt.plot(self.coords[goodInds,0], self.coords[goodInds,1], '.', color='b')
        plt.plot(self.coords[badInds,0], self.coords[badInds,1], '.', color='r')
        plt.show()


class KVecGUI():
    def __init__(self, fftImage, fftFreqs, plotRange=None):
        zeroKxLoc = np.where(fftFreqs[0]==0)[0][0]
        zeroKyLoc = np.where(fftFreqs[1]==0)[0][0]
        if plotRange is not None:
            fftFreqs[0] = fftFreqs[0][zeroKxLoc-plotRange:zeroKxLoc+plotRange]
            fftFreqs[1] = fftFreqs[1][zeroKyLoc-plotRange:zeroKyLoc+plotRange]
            fftImage = fftImage[zeroKxLoc-plotRange:zeroKxLoc+plotRange, zeroKyLoc-plotRange:zeroKyLoc+plotRange]
            zeroKxLoc = plotRange
            zeroKyLoc = plotRange

        self.zeroKxLoc = zeroKxLoc
        self.zeroKyLoc = zeroKyLoc
        self.fftImage = fftImage
        self.fftFreqs = fftFreqs

        self.curAxis = 'x'

        getLogger(__name__).info('Click first bright spot to the right of center (red dot)')

        self.plotImage()

    def plotImage(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.ax.imshow(np.transpose(self.fftImage))
        self.ax.add_patch(patches.Circle((self.zeroKxLoc, self.zeroKyLoc), radius=10, color='red'))

        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        

        plt.show(block=True)


    def onClick(self, event):
        if self.fig.canvas.manager.toolbar._active is None:
            if self.curAxis=='x':
                self.kx = np.array([self.fftFreqs[0][int(round(event.xdata))], self.fftFreqs[1][int(round(event.ydata))]])
                getLogger(__name__).info('kx: {}'.format(self.kx))
                self.curAxis='y'
                getLogger(__name__).info('Click first bright spot below center (red dot)')
            elif self.curAxis=='y':
                self.ky = np.array([self.fftFreqs[0][int(round(event.xdata))], self.fftFreqs[1][int(round(event.ydata))]])
                getLogger(__name__).info('ky: {}'.format(self.ky))
                self.curAxis='x'
                getLogger(__name__).info('Done.')
                getLogger(__name__).info('If you want to re-select kx, click first bright spot to the '
                                         'right of center (red dot), otherwise close the plot')


if __name__=='__main__':
    if len(sys.argv)<2:
        print('Usage: "python alignGrid.py <configFile>", where <configFile> is in MKID_DATA_DIR')
        exit(1)

    cfgfilename=sys.argv[1]
    if not os.path.isfile(cfgfilename):
        mdd = os.environ['MKID_DATA_DIR']
        cfgfilename = os.path.join(mdd, cfgfilename)
    config = load(cfgfilename)

    print(config.paths.masterdoubleslist)
    aligner = BMAligner(os.path.join(config.paths.beammapdirectory, config.paths.mastertemporalbeammap),
                        config.beammap.numcols, config.beammap.numrows, config.beammap.instrument, config.beammap.flip)
    aligner.makeTemporalImage()
    # aligner.fftTemporalImage()
    aligner.loadFFT()
    aligner.findKvecsManual()
    aligner.findAngleAndScale()
    aligner.rotateAndScaleCoords()
    aligner.findOffset(50000)
    aligner.plotCoords()
    aligner.saveTemporalMap(os.path.join(config.paths.beammapdirectory, config.paths.alignedbeammap))
    if config.paths.masterdoubleslist is not None:
        aligner.makeDoublesTemporalMap(os.path.join(config.paths.beammapdirectory, config.paths.masterdoubleslist),
                                       os.path.join(config.paths.beammapdirectory, config.paths.outputdoublename))

