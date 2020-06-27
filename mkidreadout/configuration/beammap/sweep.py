"""
#Todo implement handling of unused feedlines in combo

Author: Alex Walter, Rupert Dodkins
Date: Dec 11, 2019

This file contains all the calls to create a beammap file with format:
resID   flag    loc_x   loc_y
[int    int     int     int]

This is accomplished in 5 stages.

1) --xcor or --simple-locate creates an unverified temporal beammap split across the number of feedlines with format:
resID   flag    time_x  time_y
[int    int     float   float]

2) --manual {#fl} opens the GUI for the user to check the location and apply the relevant flags

3) --combo takes the relevant FL from each file and combines them

4) --align see top of aligngrid.py

5) --clean see top of clean.py

Classes in this file:
BeamSweep1D(imageList, pixelComputationMask=None, minCounts=5, maxCountRate=2499)
ManualTemporalBeammap(x_images, y_images, initial_bmap, stage1_bmap, stage2_bmap)
TemporalBeammap(configFN)
BeamSweepGaussFit(imageList, initialGuessImage)

Usage:
    From the commandline:
    $ python sweep.py [sweep.yml] <stage>

$ python sweep.py --help  #for details on calling the different steps

"""

import os, sys

import matplotlib
import numpy as np
import multiprocessing as mp
import argparse
import time, datetime, calendar


matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from mkidcore.config import load
from mkidcore.corelog import getLogger, create_log
from mkidcore.hdf.mkidbin import parse, extract
from mkidcore.objects import Beammap
from mkidcore.pixelflags import beammap as beamMapFlags
from mkidcore.instruments import DEFAULT_ARRAY_SIZES, MEC_FEEDLINE_INFO, DARKNESS_FEEDLINE_INFO

import mkidreadout.config
from mkidreadout.configuration.beammap import aligngrid as bmap_align
from mkidreadout.configuration.beammap import clean as bmap_clean
import mkidreadout.configuration.beammap.utils as bmu
from mkidreadout.configuration.beammap.flags import timestream_flags

N_CPU = 4


def bin2imgs((binfile, nrows, ncols)):
    """ Grab both intensity and phase maps from bin data """
    log.info("Making intensity and phase maps for {}".format(binfile))
    photons = parse(binfile)

    intensitymap = np.zeros((nrows, ncols))
    phasemap = np.zeros((nrows, ncols))
    for x, y, p in zip(photons['x'], photons['y'], photons['phase']):  #
        intensitymap[y, x] += 1
        phasemap[y, x] += p
    phasemap /= intensitymap

    return intensitymap, phasemap

def bin2img((binfile, nrows, ncols)):
    """ Grab intensity maps from bin data """
    log.info("Making intensity map for {}. discarding phase info".format(binfile))
    photons = parse(binfile)

    intensitymap = np.histogram2d(photons['y'], photons['x'], bins=[range(nrows + 1), range(ncols + 1)])[0]

    return intensitymap

def bin2imgfile(starttime, inttime, bindir, initialbmfile, nrows, ncols):
    photons = extract(bindir, starttime, inttime, initialbmfile, ncols, nrows)
    return np.histogram2d(photons['y'], photons['x'], bins=[range(nrows + 1), range(ncols + 1)])[0]
    

def raster2img(binDir, ditherLogFile, ditherTimestamp, axis, nrows, ncols, dithercache=True): 
    """
    ditherlog is tuple of (starts, ends, pos)
    axis is 'x' or 'y'
    """
    posTolerance = 0.002
    minCoords = 10
    
    if axis == 'x':
        axInd = 0
    elif axis == 'y':
        axInd = 1
    else:
        raise Exception('Invalid direction')

    ditherFrameDict = getDitherFrames(binDir, ditherLogFile, ditherTimestamp, nrows, ncols, dithercache)
    pos = ditherFrameDict['pos']
    images = ditherFrameDict['frames']

    uniqueCoords = np.sort(pos[:, axInd])
    uniqueMask = np.abs(np.diff(uniqueCoords)) > posTolerance
    uniqueMask = np.append([True], uniqueMask)
    uniqueCoords = uniqueCoords[uniqueMask]


    frames = []
    for coord in uniqueCoords:
        coordMask = np.abs(coord - pos[:, axInd]) < posTolerance
        if np.sum(coordMask) > minCoords:
            frames.append(np.sum(images[coordMask, :, :], axis=0))
    
    return np.asarray(frames)


def getDitherFrames(binDir, ditherLogFile, ditherTimestamp, nrows, ncols, useCache=True):
    useMP = False
    ditherlog = bmu.getDitherInfo(ditherTimestamp, ditherLogFile)
    loghash = hash((tuple(ditherlog[0]), tuple(ditherlog[1]), tuple(ditherlog[2])))&sys.maxsize
    cacheFn = os.path.join(os.path.dirname(ditherLogFile), 'ditherFrames_' + str(loghash) + '.npz')
    if useCache:
        if os.path.isfile(cacheFn):
            return np.load(cacheFn)
    
    if useMP:
        pool = mp.Pool(N_CPU)

    starts = np.asarray(ditherlog[0])
    ends = np.asarray(ditherlog[1])
    pos = np.asarray(ditherlog[2])

    images = np.empty((len(starts), nrows, ncols))
    inds = range(len(starts))
    assert np.all(starts == np.sort(starts)), 'dither start times out of order!'
    assert np.all(ends == np.sort(ends)), 'dither end times out of order!'
    photoncache = parse(os.path.join(binDir, str(int(starts[0]-1))+'.bin')) 
    photoncache = np.append(photoncache, parse(os.path.join(binDir, str(int(starts[0]))+'.bin')))
    photoncacheLastFile = int(starts[0])

    curYr = datetime.datetime.utcnow().year
    yrStart = datetime.date(curYr, 1, 1)
    tsOffs = calendar.timegm(yrStart.timetuple()) #UTC time in seconds at start of year; reference point for bin timestamps
    #parsebin timestamps are in microseconds since year start

    for i, startTime, endTime in zip(inds, starts, ends):
        startTimestamp = 1.e6*(startTime - tsOffs) #microseconds since year start
        endTimestamp = 1.e6*(endTime - tsOffs)
        print 'startTimestamp', startTimestamp
        print 'endTimestamp', endTimestamp
        print photoncache['tstamp'][0], photoncache['tstamp'][-1]
        if startTimestamp < photoncache['tstamp'][0] - 100:
            raise Exception('photon cache start bug')
        while endTimestamp > photoncache['tstamp'][-1]:
            print 'parsing new files:', photoncacheLastFile + 1
            if useMP:
                fileList = [os.path.join(binDir, str(photoncacheLastFile + i + 1) + '.bin') for i in range(N_CPU)]
                newphotons = pool.map(parse, fileList)
                for phots in newphotons:
                    photoncache = np.append(photoncache, phots)
                photoncacheLastFile += N_CPU
            else:
                photoncache = np.append(photoncache, parse(os.path.join(binDir, str(photoncacheLastFile + 1)+'.bin')))
                photoncacheLastFile += 1

        startInd = np.argmin(np.abs(photoncache['tstamp'] - startTimestamp))
        endInd = np.argmin(np.abs(photoncache['tstamp'] - endTimestamp))
        photons = photoncache[startInd:endInd]
        photoncache = photoncache[startInd:] #remove all photons before startTime
        images[i] = np.histogram2d(photons['y'], photons['x'], bins=[range(nrows + 1), range(ncols + 1)])[0]
        #getLogger(__name__).debug('Making frame {}/{}'.format(i, len(starts)))
        print 'Making frame {}/{}'.format(i, len(starts))

    ditherFrameDict = {'frames':images, 'starts':starts, 'ends':ends, 'pos':pos}
    np.savez(cacheFn, **ditherFrameDict)
    if useMP:
        pool.close()
    return ditherFrameDict







class FitBeamSweep(object):
    """
    Uses a fit to find peak in lightcurve (currently either gaussian or CoM)
    Can be used to refine the output of the cross-correlation.

    
    """

    def __init__(self, imageList, locEstimates=None):
        self.imageList = imageList
        self.initialGuessImage = locEstimates
        self.peakLocs = np.empty(imageList[0].shape)
        self.peakLocs[:] = np.nan

    def fitTemporalPeakLocs(self, fitType, fitWindow=20):
        """
        INPUTS:
            fitWindow - Only find peaks within this window of the initial guess

        Returns:
            peakLocs - map of peak locations for each pixel
        """
        fitType = fitType.lower()
        if fitType!='gaussian' and fitType!= 'com':
            raise Exception('fitType must be either Gaussian or CoM!')
        for y in range(self.imageList[0].shape[0]):
            for x in range(self.imageList[0].shape[1]):
                timestream = self.imageList[:, y, x]
                if self.initialGuessImage is None or np.logical_not(np.isfinite(self.initialGuessImage[y, x])):
                    peakGuess= np.nan
                else:
                    peakGuess=self.initialGuessImage[y, x]

                if fitType == 'gaussian':
                    self.peakLocs[y,x] = bmu.fitPeak(timestream, peakGuess, fitWindow)[0]
                elif fitType == 'com':
                    self.peakLocs[y,x] = bmu.getPeakCoM(timestream, peakGuess, fitWindow)
        return self.peakLocs


class CorrelateBeamSweep(object):
    """
    This class is for computing a temporal beammap using a list of images
    
    It uses a complicated cross-correlation function to find the pixel locations
    """

    def __init__(self, imageList, pixelComputationMask=None, minCounts=5, maxCountRate=2499):
        """
        Be careful, this function doesn't care about the units of time in the imageList
        The default minCounts, maxCountRate work well when the images are binned as 1 second exposures        

        INPUTS:
            imageList - list of images
            pixelComputationMask - It takes too much memory to calculate the beammap for the whole array at once. 
                                   This is a 2D array of integers (same shape as an image) with the value at each pixel 
                                   that corresponds to the group we want to compute it with.
            minCounts - integer of minimum counts during total exposure for it to be a good pixel
            maxCountRate - Check that the countrate is less than this in every image frame
        """
        self.imageList = np.asarray(imageList)

        # Use these parameters to determine what's a good pixel
        self.minCounts = minCounts  # counts during total exposure
        self.maxCountRate = maxCountRate  # counts per image frame

        nPix = np.prod(self.imageList[0].shape)
        nTime = len(self.imageList)
        bkgndList = 1.0 * np.median(self.imageList, axis=0)
        nCountsList = 1.0 * np.sum(self.imageList, axis=0)
        maxCountsList = 1.0 * np.amax(self.imageList, axis=0)
        badPix = np.where(np.logical_not(
            (nCountsList > minCounts) * (maxCountsList < maxCountRate) * (bkgndList < nCountsList / nTime)))

        if pixelComputationMask is None:
            nGoodPix = nPix - len(badPix[0])
            # nGroups=np.prod(imageList.shape)*(np.prod(imageList[0].shape)-1)/(200*3000*2999)*nGoodPix/nPix     # 300 timesteps x 3000 pixels takes a lot of memory...
            nGroups = nTime * nGoodPix * (nGoodPix - 1) / (600 * 3000 * 2999.)
            nGroups = nGoodPix / 1200.
            nGroups = max(nGroups, 1.)
            pixelComputationMask = np.random.randint(0, int(round(nGroups)), imageList[0].shape)
            # pixelComputationMask=np.repeat(range(5),2000).reshape(imageList[0].shape)
        self.compMask = np.asarray(pixelComputationMask)
        if len(badPix[0]) > 0:
            self.compMask[badPix] = np.amax(self.compMask) + 1  # remove bad pixels from the computation
            self.compGroups = (np.unique(self.compMask))[:-1]
        else:
            self.compGroups = np.unique(self.compMask)

    def getAbsOffset(self, shiftedTimes, auto=True, locLimit=None):
        """
        The autocorrelation function can only calculate relative time differences
        between pixels. This function defines the absolute time reference (ie. the
        location of the peak)

        INPUTS:
            shiftedTimes: a list of pixel time streams shifted to match up
            auto: if False then ask user to click on a plot
        """
        if not np.isfinite(locLimit) or locLimit<0 or locLimit>=len(shiftedTimes): locLimit=-1
        offset = np.argmax(np.sum(shiftedTimes[:locLimit], axis=0))
        if auto: return offset

        getLogger('Sweep').info("Please click the correct peak")
        fig, ax = plt.subplots()
        for p_i in range(len(shiftedTimes)):
            ax.plot(shiftedTimes[p_i])
        ax.plot(np.sum(shiftedTimes, axis=0), 'k-')
        ln = ax.axvline(offset, c='b')

        def onclick(event):
            if fig.canvas.manager.toolbar._active is None:
                offset = event.xdata
                getLogger('Sweep').info(offset)
                ln.set_xdata(offset)
                plt.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        return offset

    def findRelativePixelLocations(self, locLimit=None):
        """
        Use auto correllation and least squares to find best time offsets between pixels
        """
        try:
            locs = np.empty(self.imageList[0].shape)
        except TypeError:
            return []
        locs[:] = np.nan

        for g in self.compGroups:
            # for g in [0]:
            getLogger('Sweep').info('Starting group {}'.format(g))
            compPixels = np.where(self.compMask == g)

            timestreams = np.transpose(self.imageList[:, compPixels[0], compPixels[1]])  # shape [nPix, nTime]
            correlationList, goodPix = bmu.crossCorrelateTimestreams(timestreams, self.minCounts, self.maxCountRate)
            if len(goodPix) == 0: continue
            correlationLocs = np.argmax(correlationList, axis=1)

            correlationQaulity = 1.0 * np.amax(correlationList, axis=1) / np.sum(correlationList, axis=1)
            # correlationQuality = np.sum((correlationList[:,:len(goodPix)/2] - (correlationList[:,-1:-len(goodPix)/2-1:-1]))**2.,axis=1)     #subtract the mirror, square, and sum. If symmetric then it should be near 0
            # pdb.set_trace()
            del correlationList

            getLogger('Sweep').info("Making Correlation matrix...")
            corrMatrix = np.zeros((len(goodPix), len(goodPix)))
            corrMatrix[np.triu_indices(len(goodPix), 1)] = correlationLocs - len(self.imageList) / 2
            corrMatrix[np.tril_indices(len(goodPix), -1)] = -1 * np.transpose(corrMatrix)[
                np.tril_indices(len(goodPix), -1)]
            del correlationLocs
            corrQualityMatrix = np.ones((len(goodPix), len(goodPix)))
            corrQualityMatrix[np.triu_indices(len(goodPix), 1)] = correlationQaulity
            corrQualityMatrix[np.tril_indices(len(goodPix), -1)] = -1 * np.transpose(corrQualityMatrix)[
                np.tril_indices(len(goodPix), -1)]
            del correlationQaulity
            getLogger('Sweep').info("Done...")

            getLogger('Sweep').info("Finding Best Relative Locations...")
            a = bmu.minimizePixelLocationVariance(corrMatrix)
            bestPixelArgs, totalVar = bmu.determineSelfconsistentPixelLocs2(corrMatrix, a)
            bestPixels = goodPix[bestPixelArgs]
            bestPixels = bestPixels[: len(bestPixels) / 20]
            best_a = bmu.minimizePixelLocationVariance(corrMatrix[:, np.where(np.in1d(goodPix, bestPixels))[0]])
            getLogger('Sweep').info("Done...")

            getLogger('Sweep').info("Finding Timestream Peak Locations...")
            shifts = np.rint(best_a[bestPixelArgs]).astype(np.int)
            shifts = shifts[: len(bestPixels)]
            shifts = shifts[:, None] + np.arange(len(timestreams[0]))
            shifts[np.where(shifts < 0)] = -1
            shifts[np.where(shifts >= len(timestreams[0]))] = -1
            bkgndList = 1.0 * np.median(timestreams[bestPixels], axis=1)
            nCountsList = 1.0 * np.sum(timestreams[bestPixels], axis=1)
            shiftedTimes = np.zeros((len(bestPixels), len(timestreams[0]) + 1))
            shiftedTimes[:, :-1] = (timestreams[bestPixels] - bkgndList[:, None]) / nCountsList[:,
                                                                                    None]  # padded timestream with 0
            shiftedTimes = shiftedTimes[np.arange(len(bestPixels))[:, None], shifts]  # shift each timestream
            # offset = np.argmax(np.sum(shiftedTimes,axis=0))
            offset = self.getAbsOffset(shiftedTimes, locLimit=locLimit)
            del shiftedTimes

            best_a += offset

            getLogger('Sweep').info("Done...")

            locs[compPixels[0][goodPix], compPixels[1][goodPix]] = best_a

        locs[locs < 0] = 0
        locs[locs >= len(self.imageList)] = len(self.imageList)
        return locs


class ManualTemporalBeammap(object):
    def __init__(self, x_images, y_images, initial_bmap, stage1_bmap, stage2_bmap, fitType = None,
                 xp_images = None, yp_images=None):
        """
        Class for manually clicking through beammap.
        Saves a temporal beammap with filename temporalBeammapFN-HHMMSS.txt
        A 'temporal' beammap is one that doesn't have x/y but instead the peak location in time from the swept light beam.

        INPUTS:
            x_images - list of images for sweep(s) in x-direction
            y_images -
            initial_bmap - path+filename of initial beammap used for making the images
            stage1_bmap - path+filename of the temporal beammap (time at peak instead of x/y value)
                             If the temporalBeammap doesn't exist then it will be instantiated with nans
                             Flags set to noDacTone (currently 1) will be skipped. b.saveTemporalBeammap() will produce
                             a series of files with all but one FL flagged to noDacTone

            stage2_bmap  - output file
            fitType - Type of fit to use when finding exact peak location from click. Current options are
                             com and gaussian. Ignored if None (default).
        """
        self.x_images = x_images
        self.y_images = y_images
        self.xp_images = xp_images
        self.yp_images = yp_images
        self.nTime_x = len(self.x_images)
        self.nTime_y = len(self.y_images)
        self.goodregion = []  # corner coordinates

        self.initial_bmap = initial_bmap  # mec.bmap
        self.stage1_bmap = stage1_bmap
        self.stage2_bmap = stage2_bmap

        if os.path.isfile(self.stage2_bmap):  # has a clicked file already been made? Load that instead
            self.stage1_bmap = self.stage2_bmap
        self.resIDsMap, self.flagMap, self.x_loc, self.y_loc = bmu.shapeBeammapIntoImages(self.initial_bmap,
                                                                                          self.stage1_bmap)
        self.flagMap[np.all(self.x_images==0, axis=0) | np.all(self.y_images==0, axis=0)] = beamMapFlags['noDacTone']
        if self.stage1_bmap is None or not os.path.isfile(self.stage1_bmap):
            self.flagMap[np.where(self.flagMap != beamMapFlags['noDacTone'])] = beamMapFlags['failed']

        self.fitType = fitType.lower()

        self.goodPix = np.where(self.flagMap != beamMapFlags['noDacTone'])
        self.nGoodPix = len(self.goodPix[0])
        getLogger('Sweep').info('Pixels with light: {}'.format(self.nGoodPix))
        self.curPixInd = 0
        self.curPixValue = np.amax(beamMapFlags.values()) + 1

        self._want_to_close = False
        self.plotFlagMap()
        self.plotXYscatter()
        self.plotTimestream()

        plt.show()

    def saveTemporalBeammap(self):
        getLogger('Sweep').info('Saving: {}'.format(self.stage2_bmap))
        allResIDs = self.resIDsMap.flatten()
        flags = self.flagMap.flatten()
        x = self.x_loc.flatten()
        y = self.y_loc.flatten()
        args = np.argsort(allResIDs)
        data = np.asarray([allResIDs[args], flags[args], x[args], y[args]]).T
        np.savetxt(self.stage2_bmap, data, fmt='%7d %3d %7f %7f')

    def plotTimestream(self):
        self.fig_time, (self.ax_time_x, self.ax_time_y) = plt.subplots(2)
        y = self.goodPix[0][self.curPixInd]
        x = self.goodPix[1][self.curPixInd]

        xStream = self.x_images[:, y, x]
        ln_xStream, = self.ax_time_x.plot(xStream)
        x_loc = self.x_loc[y, x]
        if np.logical_not(np.isfinite(x_loc)): x_loc = -1
        ln_xLoc = self.ax_time_x.axvline(x_loc, c='r')

        yStream = self.y_images[:, y, x]
        ln_yStream, = self.ax_time_y.plot(yStream)
        y_loc = self.y_loc[y, x]
        if np.logical_not(np.isfinite(y_loc)): y_loc = -1
        ln_yLoc = self.ax_time_y.axvline(y_loc, c='r')

        self.ax_xp = self.ax_time_x.twinx()
        ln_xphasestream, = self.ax_xp.plot(self.xp_images[:, y, x], alpha=0.35)
        self.ax_yp = self.ax_time_y.twinx()
        ln_yphasestream, = self.ax_yp.plot(self.yp_images[:, y, x], alpha=0.35)

        # self.ax_time_x.set_title('Pix '+str(self.curPixInd)+' resID'+str(int(self.resIDsMap[y,x]))+' ('+str(x)+', '+str(y)+')')
        flagStr = [key for key, value in beamMapFlags.items() if value == self.flagMap[y, x]][0]
        self.ax_time_x.set_title(
            'Pix ' + str(self.curPixInd) + '; resID' + str(int(self.resIDsMap[y, x])) + '; (' + str(x) + ', ' + str(
                y) + '); flag ' + str(flagStr))
        self.ax_time_x.set_ylabel('X counts')
        self.ax_xp.set_ylabel('X phase')
        self.ax_time_y.set_ylabel('Y counts')
        self.ax_yp.set_ylabel('Y phase')
        self.ax_time_y.set_xlabel('Timesteps')
        self.ax_time_x.set_xlim(-2, self.nTime_x + 2)
        self.ax_time_y.set_xlim(-2, self.nTime_y + 2)
        self.ax_time_x.set_ylim(0, None)
        self.ax_time_y.set_ylim(0, None)

        self.timestreamPlots = [ln_xLoc, ln_yLoc, ln_xStream, ln_yStream, ln_xphasestream, ln_yphasestream]

        self.fig_time.canvas.mpl_connect('button_press_event', self.onClickTime)
        self.fig_time.canvas.mpl_connect('close_event', self.onCloseTime)
        self.fig_time.canvas.mpl_connect('key_press_event', self.onKeyTime)
        plt.tight_layout()

    def onKeyTime(self, event):
        getLogger('Sweep').info('Pressed '+event.key)
        # if event.key not in ('right', 'left'): return
        if event.key in ['right', 'c']:
            self.curPixInd += 1
            self.curPixInd %= self.nGoodPix
            self.updateTimestreamPlot()
            self.updateXYPlot(1)
            self.updateFlagMapPlot()
        elif event.key in [' ']:
            self.curPixInd += 1
            self.curPixInd %= self.nGoodPix
            self.updateTimestreamPlot(fastforward=True)
            self.updateXYPlot(1)
            self.updateFlagMapPlot()
        elif event.key in ['r']:
            self.curPixInd -= 1
            self.curPixInd %= self.nGoodPix
            self.updateTimestreamPlot(rewind=True)
            self.updateXYPlot(1)
            self.updateFlagMapPlot()
        elif event.key in ['left']:
            self.curPixInd -= 1
            self.curPixInd %= self.nGoodPix
            self.updateTimestreamPlot()
            self.updateXYPlot(1)
            self.updateFlagMapPlot()
        elif event.key in ['b']:
            y = self.goodPix[0][self.curPixInd]
            x = self.goodPix[1][self.curPixInd]
            self.x_loc[y, x] = np.nan
            self.y_loc[y, x] = np.nan
            getLogger('Sweep').info('Pix {} ({}, {}) Marked bad'.format(int(self.resIDsMap[y, x]),x,y))
            self.updateFlagMap(self.curPixInd)
            self.curPixInd += 1
            self.curPixInd %= self.nGoodPix
            self.updateTimestreamPlot()
            self.updateXYPlot(2)
            self.updateFlagMapPlot()
        elif event.key in ['d']:
            y = self.goodPix[0][self.curPixInd]
            x = self.goodPix[1][self.curPixInd]
            if self.flagMap[y, x] != beamMapFlags['double']:
                self.flagMap[y, x] = beamMapFlags['double']
                getLogger('Sweep').info('Pix {} ({}, {}) Marked as double and moved offset to larger phase dip'.format(int(self.resIDsMap[y, x]), x, y))
                self.y_loc[y, x] = np.argmin(self.yp_images[:, y, x])
                self.x_loc[y, x] = np.argmin(self.xp_images[:, y, x])
            else:
                self.flagMap[y, x] = beamMapFlags['good']
                getLogger('Sweep').info('Pix {} ({}, {}) Un-Marked as double'.format(int(self.resIDsMap[y, x]), x, y))
            self.updateFlagMap(self.curPixInd)
            self.updateTimestreamPlot(5)

    def onCloseTime(self, event):
        if not self._want_to_close:
            self.curPixInd += 1
            self.curPixInd %= self.nGoodPix
            self.updateTimestreamPlot()
            self.plotTimestream()
            plt.show()
        # event.ignore()
        # self.fig_time.show()

    def updateTimestreamPlot(self, lineNum=4, fastforward = False, rewind = False):
        y = self.goodPix[0][self.curPixInd]
        x = self.goodPix[1][self.curPixInd]

        if fastforward or rewind:
            skip_timestream = True
            counter = 0
            if len(self.goodregion) != 4:
                log.info('Skipping xy position check until region selected')
            while skip_timestream and counter < self.nGoodPix:
                counter += 1
                y = self.goodPix[0][self.curPixInd]
                x = self.goodPix[1][self.curPixInd]
                log.info('\n-- Checking pixel {} --'.format(self.curPixInd))

                xy_good = bmu.check_xy(self.x_loc[y, x], self.y_loc[y, x], self.goodregion)  # returns True if self.goodregion is []
                if not xy_good:
                    self.x_loc[y, x] = np.nan
                    self.y_loc[y, x] = np.nan
                    getLogger('Sweep').info('pixel failed xy check. setting flag to bad')
                    self.updateFlagMap(self.curPixInd)
                    self.updateXYPlot(2)
                    self.updateFlagMapPlot()

                xstream_flag = bmu.check_timestream(self.x_images[:, y, x], self.x_loc[y, x])
                xstream_good = xstream_flag == timestream_flags['good'] #or (xstream_flag == timestream_flags['double'] and xp_good)

                ystream_flag = bmu.check_timestream(self.y_images[:, y, x], self.y_loc[y, x])
                ystream_good = ystream_flag == timestream_flags['good'] #or (ystream_flag == timestream_flags['double'] and yp_good)

                xp_good, yp_good = bmu.check_phasestreams(self.xp_images[:, y, x], self.yp_images[:, y, x],
                                                      self.x_loc[y, x], self.y_loc[y, x])
                if not xp_good:
                    getLogger('Sweep').info('x failed phase location check')
                if not yp_good:
                    getLogger('Sweep').info('y failed phase location check')

                skip_timestream = xy_good and xstream_good and ystream_good and xp_good and yp_good
                if skip_timestream:
                    if fastforward:
                        self.curPixInd += 1
                    else:
                        self.curPixInd -= 1
                    self.curPixInd %= self.nGoodPix

        if lineNum == 0 or lineNum >= 4:
            offset = self.x_loc[y, x]
            if np.logical_not(np.isfinite(offset)): offset = -1
            self.timestreamPlots[0].set_xdata(offset)
        if lineNum == 1 or lineNum >= 4:
            offset = self.y_loc[y, x]
            if np.logical_not(np.isfinite(offset)): offset = -1
            self.timestreamPlots[1].set_xdata(offset)
        if lineNum == 2 or lineNum >= 4:
            self.timestreamPlots[2].set_ydata(self.x_images[:, y, x])
            # self.ax_time_x.autoscale(True,'y',True)
            self.ax_time_x.set_ylim(0, 1.05 * np.amax(self.x_images[:, y, x]))
            self.ax_time_x.set_xlim(-2, self.nTime_x + 2)
            self.timestreamPlots[4].set_ydata(self.xp_images[:, y, x])
            self.ax_xp.set_ylim(1.05 * np.nanmin(self.xp_images[:, y, x]), 0)
        if lineNum == 3 or lineNum >= 4:
            self.timestreamPlots[3].set_ydata(self.y_images[:, y, x])
            # self.ax_time_y.autoscale(True,'y',True)
            self.ax_time_y.set_ylim(0, 1.05 * np.amax(self.y_images[:, y, x]))
            self.ax_time_y.set_xlim(-2, self.nTime_y + 2)
            self.timestreamPlots[5].set_ydata(self.yp_images[:, y, x])
            self.ax_yp.set_ylim(1.05 * np.nanmin(self.yp_images[:, y, x]), 0)
        if lineNum == 2 or lineNum == 3 or lineNum == 4 or lineNum == 5:
            flagStr = [key for key, value in beamMapFlags.items() if value == self.flagMap[y, x]][0]
            self.ax_time_x.set_title(
                'Pix ' + str(self.curPixInd) + '; resID' + str(int(self.resIDsMap[y, x])) + '; (' + str(x) + ', ' + str(
                    y) + '); flag ' + str(flagStr))
        self.fig_time.canvas.draw()

    def onClickTime(self, event):
        if self.fig_time.canvas.manager.toolbar._active is None:
            # update time plot and x/y_loc
            y = self.goodPix[0][self.curPixInd]
            x = self.goodPix[1][self.curPixInd]
            offset = event.xdata
            if offset < 0: offset = np.nan

            if event.inaxes == self.ax_time_x or event.inaxes == self.ax_xp:
                offset = bmu.snapToPeak(self.x_images[:, y, x], offset)
                if self.fitType == 'gaussian':
                    fitParams=bmu.fitPeak(self.x_images[:,y,x],offset,20)
                    offset=fitParams[0]
                    getLogger('Sweep').info('Gaussian fit params: ' + str(fitParams))
                elif self.fitType == 'com':
                    offset=bmu.getPeakCoM(self.x_images[:,y,x],offset)
                    getLogger('Sweep').info('Using CoM: ' + str(offset), 10)
                self.x_loc[y, x] = offset
                getLogger('Sweep').info('x: {}'.format(offset))
                self.updateTimestreamPlot(0)

            elif event.inaxes == self.ax_time_y or event.inaxes == self.ax_yp:
                offset = bmu.snapToPeak(self.y_images[:, y, x], offset)
                if self.fitType == 'gaussian':
                    fitParams=bmu.fitPeak(self.y_images[:,y,x],offset,20)
                    offset=fitParams[0]
                    getLogger('Sweep').info('Gaussian fit params: ' + str(fitParams))
                elif self.fitType == 'com':
                    offset=bmu.getPeakCoM(self.y_images[:,y,x],offset, 10)
                    getLogger('Sweep').info('Using CoM: ' + str(offset))
                self.y_loc[y, x] = offset
                getLogger('Sweep').info('y: {}'.format(offset))
                self.updateTimestreamPlot(1)

            self.updateXYPlot(2)
            self.updateFlagMap(self.curPixInd)

    def updateXYPlot(self, lines):
        if lines == 0 or lines == 2:
            self.ln_XY.set_data(self.x_loc[self.flagMap==0].flatten(), self.y_loc[self.flagMap==0].flatten())
            self.ln_XY_bad.set_data(self.x_loc[self.flagMap!=0].flatten(), self.y_loc[self.flagMap!=0].flatten())
        if lines == 1 or lines == 2:
            y = self.goodPix[0][self.curPixInd]
            x = self.goodPix[1][self.curPixInd]
            self.ln_XYcur.set_data([self.x_loc[y, x]], [self.y_loc[y, x]])

        self.ax_XY.set_title(self.goodregionon) if len(self.goodregion) == 4 else self.ax_XY.set_title(
            self.goodregionoff)

        self.ax_XY.autoscale(True)
        self.fig_XY.canvas.draw()

    def updateFlagMap(self, curPixInd):
        y = self.goodPix[0][curPixInd]
        x = self.goodPix[1][curPixInd]
        if np.isfinite(self.x_loc[y, x]) * np.isfinite(self.y_loc[y, x]):
            if self.flagMap[y, x] != beamMapFlags['double']: self.flagMap[y, x] = beamMapFlags['good']
        elif np.logical_not(np.isfinite(self.x_loc[y, x])) * np.isfinite(self.y_loc[y, x]):
            self.flagMap[y, x] = beamMapFlags['xFailed']
        elif np.isfinite(self.x_loc[y, x]) * np.logical_not(np.isfinite(self.y_loc[y, x])):
            self.flagMap[y, x] = beamMapFlags['yFailed']
        elif np.logical_not(np.isfinite(self.x_loc[y, x])) * np.logical_not(np.isfinite(self.y_loc[y, x])):
            self.flagMap[y, x] = beamMapFlags['failed']
        self.saveTemporalBeammap()
        self.updateFlagMapPlot()

    def updateFlagMapPlot(self):
        flagMap_masked = np.ma.masked_where(self.flagMap == beamMapFlags['noDacTone'], self.flagMap)
        flagMap_masked[self.goodPix[0][self.curPixInd], self.goodPix[1][self.curPixInd]] = self.curPixValue
        self.ln_flags.set_data(flagMap_masked)
        self.fig_flags.canvas.draw()

    def plotFlagMap(self):
        self.fig_flags, self.ax_flags = plt.subplots()
        flagMap_masked = np.ma.masked_where(self.flagMap == beamMapFlags['noDacTone'], self.flagMap)
        my_cmap = matplotlib.cm.get_cmap('YlOrRd')
        my_cmap.set_under('w')
        my_cmap.set_over('c')
        my_cmap.set_bad('k')

        flagMap_masked[self.goodPix[0][self.curPixInd], self.goodPix[1][self.curPixInd]] = self.curPixValue
        self.ln_flags = self.ax_flags.matshow(flagMap_masked, cmap=my_cmap, vmin=0.1,
                                              vmax=np.amax(beamMapFlags.values()) + .1)
        self.ax_flags.set_title('Flag map')
        # cbar = self.fig_flags.colorbar(flagMap_masked, extend='both', shrink=0.9, ax=self.ax_flags)
        self.fig_flags.canvas.mpl_connect('button_press_event', self.onClickFlagMap)
        self.fig_flags.canvas.mpl_connect('close_event', self.onCloseFlagMap)

    def onCloseFlagMap(self, event):
        # plt.close(self.fig_XY)
        # plt.close(self.fig_time)
        plt.show()
        self._want_to_close = True
        plt.close('all')

    def onClickFlagMap(self, event):
        if self.fig_flags.canvas.manager.toolbar._active is None and event.inaxes == self.ax_flags:
            x = int(np.floor(event.xdata + 0.5))
            y = int(np.floor(event.ydata + 0.5))
            nRows, nCols = self.x_images[0].shape
            if x >= 0 and x < nCols and y >= 0 and y < nRows:
                msg = 'Clicked Flag Map! [{}, {}] -> {} Flag: {}'
                getLogger('Sweep').info(msg.format(x, y, self.resIDsMap[y, x], self.flagMap[y, x]))
                pixInd = np.where((self.goodPix[0] == y) * (self.goodPix[1] == x))[0]
                if len(pixInd) == 1:
                    self.curPixInd = pixInd[0]
                    self.updateFlagMapPlot()
                    self.updateXYPlot(1)
                    self.updateTimestreamPlot(4)
                    # plt.draw()
                else:
                    getLogger('Sweep').info("No photons detected")

    def plotXYscatter(self):
        self.fig_XY, self.ax_XY = plt.subplots()
        self.ln_XY, = self.ax_XY.plot(self.x_loc[self.flagMap==0].flatten(), self.y_loc[self.flagMap==0].flatten(), 'b.', markersize=2)
        self.ln_XY_bad, = self.ax_XY.plot(self.x_loc[self.flagMap!=0].flatten(), self.y_loc[self.flagMap!=0].flatten(), 'b.', alpha=0.1, markersize=2)
        y = self.goodPix[0][self.curPixInd]
        x = self.goodPix[1][self.curPixInd]
        self.ln_XYcur, = self.ax_XY.plot([self.x_loc[y, x]], [self.y_loc[y, x]], 'go')
        self.goodregionon = 'Good region set'
        self.goodregionoff = 'Make a rectangle around the feedline by clicking 4 locations'
        self.ax_XY.set_title(self.goodregionon) if len(self.goodregion) == 4 else self.ax_XY.set_title(self.goodregionoff)
        self.fig_XY.canvas.mpl_connect('close_event', self.onCloseXY)
        self.fig_XY.canvas.mpl_connect('button_press_event', self.onClickXY)

    def onCloseXY(self, event):
        # self.fig_XY.show()
        if not self._want_to_close:
            self.plotXYscatter()
            plt.show()

    def onClickXY(self, event):
        if self.fig_XY.canvas.manager.toolbar._active is None:
            log.info('Clicked point (x, y) = (%i, %i)'% (event.xdata, event.ydata))
            if len(self.goodregion)==4:
                log.info('Creating new region')
                self.goodregion=[]

            self.goodregion.append([event.xdata, event.ydata])
            if len(self.goodregion) == 4:
                log.info(('Good region coordinates', ['(%d,%d)' % (x,y) for x,y in self.goodregion]))
            self.updateXYPlot(-1)

class TemporalBeammap():
    def __init__(self, config):
        """
        This class is for finding the temporal location of each pixel in units of timesteps
        INPUTS:
            configFN - config file listing the sweeps and properties
        """
        self.config = config
        self.x_locs = None
        self.y_locs = None
        self.x_images = None
        self.y_images = None
        self.xp_images = None
        self.yp_images = None

        self.beammapdirectory = config.paths.beammapdirectory
        self.initial_bmap = config.beammap.filenames.initial_bmap
        self.stage1_bmaps = config.beammap.filenames.stage1_bmaps
        self.stage2_bmaps = config.beammap.filenames.stage2_bmaps

        self.numcols, self.numrows = DEFAULT_ARRAY_SIZES[config.beammap.instrument.lower()]
        self.numfeed = eval(config.beammap.instrument.upper()+'_FEEDLINE_INFO')['num']

    def stackImages(self, sweepType, median=True):

        sweepType = sweepType.lower()
        if sweepType not in ('x','y'):
            raise ValueError('sweepType must be x or y')
        inten_sweeps = None  # all the sweeps combined
        phase_sweeps = None
        nTimes = 0
        nSweeps = 0
        for s in self.config.beammap.sweep.sweeps:
            if s.sweeptype in sweepType or s.sweeptype == 'raster':
                nSweeps += 1.
                if s.sweeptype == 'raster':
                    intensity_maps = raster2img(self.config.paths.bin, s.ditherlog, s.starttime, sweepType, 
                            self.numrows, self.numcols)
                    phase_maps = np.zeros(intensity_maps.shape) #todo: implement this
                else:
                    both_maps = self.loadSweepBins(s, get_phases=True)  # intensity and phase
                    intensity_maps = both_maps[:, 0]
                    phase_maps = both_maps[:, 1]
                direction = -1 if s.sweepdirection is '-' else 1
                if inten_sweeps is None:
                    inten_sweeps = np.asarray([intensity_maps[::direction, :, :]])
                    phase_sweeps = np.asarray([phase_maps[::direction, :, :]])
                    nTimes = len(intensity_maps)
                else:
                    if len(intensity_maps) < nTimes:
                        pad = np.empty((nTimes - len(intensity_maps), len(intensity_maps[0]), len(intensity_maps[0][0])))
                        pad[:] = np.nan
                        intensity_maps = np.concatenate((intensity_maps[::direction, :, :], pad), 0)
                        phase_maps = np.concatenate((phase_maps[::direction, :, :], pad), 0)
                    elif len(intensity_maps) > nTimes:
                        pad = np.empty((len(inten_sweeps), len(intensity_maps) - nTimes, len(intensity_maps[0]), len(intensity_maps[0][0])))
                        pad[:] = np.nan
                        inten_sweeps = np.concatenate((inten_sweeps, pad), 1)
                        phase_sweeps = np.concatenate((phase_sweeps, pad), 1)
                        intensity_maps = intensity_maps[::direction, :, :]
                        phase_maps = phase_maps[::direction, :, :]
                    inten_sweeps = np.concatenate((inten_sweeps, intensity_maps[np.newaxis, :, :, :]), 0)
                    phase_sweeps = np.concatenate((phase_sweeps, phase_maps[np.newaxis, :, :, :]), 0)

                    # nTimes = min(len(intensity_maps), nTimes)
                    # imageList=imageList[:nTimes] + (intensity_maps[::direction,:,:])[:nTimes]
                if hasattr(s,'boards'):
                    mask = self.get_boards_mask(s)
                    mask = mask * np.ones(inten_sweeps.shape[:2])[:,:, None, None]
                    inten_sweeps[~np.bool_(mask)] = 0
                    phase_sweeps[~np.bool_(mask)] = 0

        if median:
            inten_images = np.nanmedian(inten_sweeps, 0)
            phase_images = np.nanmedian(phase_sweeps, 0)
        else:
            inten_images = np.nanmean(inten_sweeps, 0)
            phase_images = np.nanmean(phase_sweeps, 0)

        if sweepType == 'x':
            self.x_images = inten_images
            self.xp_images = phase_images
        else:
            self.y_images = inten_images
            self.yp_images = phase_images

        getLogger('sweep.TemporalBeammap').info('Stacked {} {} sweeps', int(nSweeps), sweepType)
        return inten_images

    def concatImages(self, sweepType, removeBkg=True):
        """
        This won't work well if the background level or QE of the pixel changes between sweeps...
        Should remove this first
        """
        sweepType = sweepType.lower()
        assert sweepType in ('x', 'y')
        imageList = None
        for s in self.config.beammap.sweep.sweeps:
            if s.sweeptype in sweepType or s.sweeptype == 'raster':
                getLogger('Sweep').info('loading: ' + str(s))
                # phase info used by later steps so include phase data in the created cache
                if s.sweeptype == 'raster':
                    imList = raster2img(self.config.paths.bin, s.ditherlog, s.starttime, sweepType,
                            self.numrows, self.numcols)
                    # phasemaps = np.zeros(intensity_maps.shape) #todo: implement this
                else:
                    imList = self.loadSweepBins(s, get_phases=True).astype(np.float)[:, 0]
                if removeBkg:
                    bkgndList = np.median(imList, axis=0)
                    imList -= bkgndList
                direction = -1 if s.sweepdirection == '-' else 1
                if imageList is None:
                    imageList = imList[::direction, :, :]
                else:
                    imageList = np.concatenate((imageList, imList[::direction, :, :]), axis=0)

                if hasattr(s, 'boards'):
                    mask = self.get_boards_mask(s)
                    mask = mask * np.ones(len(imageList))[:, None, None]
                    imageList[~np.bool_(mask)] = 0

        if sweepType == 'x':
            self.x_images = imageList
        else:
            self.y_images = imageList
        return imageList

    def get_boards_mask(self, sweep):
        if self.initial_bmap is not None:
            initial = os.path.join(self.beammapdirectory, self.initial_bmap)
        else:
            initial = Beammap(default=self.config.beammap.instrument).file
        temporal = os.path.join(self.beammapdirectory, self.stage1_bmaps)

        allResIDs_map, _, _, _ = bmu.shapeBeammapIntoImages(initial, temporal)
        # res_ids = []
        mask = np.zeros((self.numrows, self.numcols)).flatten()
        for board in sweep.boards:
            flnum = int(board[:-1])
            rel_start = 0 if board[-1] == 'a' else 1024
            abs_start = 10000 * flnum + rel_start
            abs_end = abs_start + 1024
            fl_ind = (abs_start < allResIDs_map.flatten(order='F')) & (allResIDs_map.flatten(order='F') < abs_end)
            mask[fl_ind] = 1

        mask = mask.reshape(self.numrows, self.numcols, order='F') == 1
        return mask

    def findLocWithCrossCorrelation(self, sweepType, pixelComputationMask=None, snapToPeaks=True,
                                    correctMultiSweep=True):
        """
        This function estimates the location in time for the light peak in each pixel by cross-correlating the timestreams
        See CorrelateBeamSweep class

        Careful: We assume the sweep speed is always the same when looking at multiple sweeps!!!
        For now, we assume initial beammap is the same too.

        INPUTS:
            sweepType - either 'x', or 'y'
            pixelComputationMask - see CorrelateBeamSweep.__init__()
            snapToPeaks - If true, snap the cross-correlation to the biggest nearby peak
            correctMultiSweep - see self.cleanCrossCorrelationToWrongSweep()

        OUTPUTS:
            locs - map of locations for each pixel [units of time]
        """
        imageList = self.concatImages(sweepType)
        dur = [s.duration for s in self.config.beammap.sweep.sweeps if s.sweeptype in sweepType.lower()]
        # FLMap = getFLMap(self.config.beammap.sweep.initial_bmap)

        sweep = CorrelateBeamSweep(imageList, pixelComputationMask)
        locs = sweep.findRelativePixelLocations(locLimit=dur[0])
        if snapToPeaks:
            for row in range(len(locs)):
                for col in range(len(locs[0])):
                    locs[row, col] = bmu.snapToPeak(imageList[:, row, col], locs[row, col])
        if correctMultiSweep:
            locs = self.cleanCrossCorrelationToWrongSweep(sweepType, locs)
        if sweepType in ['x', 'X']:
            self.x_locs = locs
        else:
            self.y_locs = locs
        return locs

    def cleanCrossCorrelationToWrongSweep(self, sweepType, locs):
        """
        If you're doing a cross-correlation with multiple timestreams concatenated after one another
        there is a common failure mode where the max cross-correlation will be where the peak in the
        first sweep matches the peak in the second sweep.

        If the sweeps are matched up, this function will correct that systematic error. If the sweep start
        times aren't matched up then this won't work
        """
        dur = [s.duration for s in self.config.beammap.sweep.sweeps if s.sweeptype in sweepType.lower()]
        for i in range(len(dur) - 1):
            locs[np.where(locs > dur[i])] -= dur[i]
        return locs

    def refinePeakLocs(self, sweepType, fitType, locEstimates=None, fitWindow=20):
        """
        This function refines the peak locations given by locEstimates with either a gaussian
        fit or center of mass calculation. Can also be used as a standalone routine (set
        locEstimates to None), but currently doesn't work well in this mode.

        Careful: The sweep start times must be aligned such that the light peaks stack up.
        We assume the sweep speed is always the same when looking at multiple sweeps!!!
        For now, we assume initial beammap is the same too.

        INPUTS:
            sweepType - either 'x', or 'y'
            locEstimate - optional guesses for peak location. Should be 2D map of peak locations

        OUTPUTS:
            locs - map of locations for each pixel [units of time]
        """
        imageList = self.stackImages(sweepType)
        sweep = FitBeamSweep(imageList, locEstimates)
        if locEstimates is None:
            fitWindow = None
        locs = sweep.fitTemporalPeakLocs(fitType, fitWindow=fitWindow)
        if sweepType in ['x', 'X']:
            self.x_locs = locs
        else:
            self.y_locs = locs
        return locs

    # def computeSweeps(self, sweepType, pixelComputationMask=None):
    #    """
    #    Careful: We assume the sweep speed is always the same!!!
    #    For now, we assume initial beammap is the same too.
    #    """
    #    imageList=self.concatImages(sweepType)
    #    sweep = CorrelateBeamSweep(imageList,pixelComputationMask)
    #    locs=sweep.findRelativePixelLocations()
    #    sweepFit = BeamSweepGaussFit(imageList, locs)
    #    locs = sweepFit.fitTemporalPeakLocs()
    #
    #    if sweepType in ['x','X']: self.x_locs=locs
    #    else: self.y_locs=locs
    #    self.saveTemporalBeammap()

    def saveTemporalBeammap(self, split_feedlines=False, use_old_flags=False):
        """

        :param split_feedlines:
            Takes each feedline data and creates a new file
        :return:
        """

        getLogger('Sweep').info('Saving')

        if self.initial_bmap is not None:
            initial = os.path.join(self.beammapdirectory, self.initial_bmap)
        else:
            initial = Beammap(default=self.config.beammap.instrument).file
        temporal = os.path.join(self.beammapdirectory, self.stage1_bmaps)

        allResIDs_map, flag_map, x_map, y_map = bmu.shapeBeammapIntoImages(initial, temporal)
        otherFlagArgs = np.where((flag_map != beamMapFlags['good']) * (flag_map != beamMapFlags['failed']) * (
                    flag_map != beamMapFlags['xFailed']) * (flag_map != beamMapFlags['yFailed']))
        otherFlags = flag_map[otherFlagArgs]
        if self.y_locs is not None and self.y_locs.shape == flag_map.shape:
            y_map = self.y_locs
            getLogger('Sweep').info('added y')
        if self.x_locs is not None and self.x_locs.shape == flag_map.shape:
            x_map = self.x_locs
            getLogger('Sweep').info('added x')

        flag_map[np.where(np.logical_not(np.isfinite(x_map)) * np.logical_not(np.isfinite(y_map)))] = beamMapFlags[
            'failed']
        flag_map[np.where(np.logical_not(np.isfinite(x_map)) * np.isfinite(y_map))] = beamMapFlags['xFailed']
        flag_map[np.where(np.logical_not(np.isfinite(y_map)) * np.isfinite(x_map))] = beamMapFlags['yFailed']
        flag_map[np.where(np.isfinite(x_map) * np.isfinite(y_map))] = beamMapFlags['good']
        flag_map[otherFlagArgs] = otherFlags

        allResIDs = allResIDs_map.flatten()
        flags = flag_map.flatten()
        x = x_map.flatten()
        y = y_map.flatten()
        args = np.argsort(allResIDs)
        if use_old_flags:
            data = np.asarray([allResIDs[args], flags[args], x[args], y[args]]).T
        else:
            data = np.asarray([allResIDs[args], np.ones_like(flags)*beamMapFlags['good'], x[args], y[args]]).T
        # get all the boards covered by the sweeps
        try:
            boards = np.array([sweep.boards for sweep in self.config.beammap.sweep.sweeps]).flatten()
        except KeyError:
            boards = []

        if len(boards) > 0:
            board_inds = []
            for board in boards:
                flnum = int(board[:-1])
                rel_start = 0 if board[-1] == 'a' else 1024
                abs_start = 10000 * flnum + rel_start
                abs_end = abs_start + 1024
                board_inds.append( (abs_start < allResIDs_map.flatten(order='F'))
                                   & (allResIDs_map.flatten(order='F') < abs_end) )
            board_inds = np.any(board_inds, axis=0)
            FL_filename = self.get_FL_filename(self.stage1_bmaps, ''.join(boards))
            log.info('Saving data for boards %s in %s' % (board, FL_filename))
            board_data = data[board_inds]
            np.savetxt(FL_filename, board_data, fmt='%7d %3d %7f %7f')

        elif split_feedlines:
            for fl in range(1, self.numfeed + 1):
                FL_filename = self.get_FL_filename(self.stage1_bmaps, fl)
                log.info('Saving FL%i data in %s' % (fl, FL_filename))
                args = np.int_(data[:, 0] / 10000) != fl  # identify resonators for feedline fl
                FL_data = data * 1
                FL_data[args, 1] = beamMapFlags['noDacTone']
                np.savetxt(FL_filename, FL_data, fmt='%7d %3d %7f %7f')
        else:
            outfile = self.get_FL_filename(self.stage1_bmaps, 'all')
            np.savetxt(outfile.format('all'), data, fmt='%7d %3d %7f %7f')

    def loadTemporalBeammap(self):
        if self.initial_bmap is not None:
            initial = os.path.join(self.beammapdirectory, self.initial_bmap)
        else:
            initial = Beammap(default=self.config.beammap.instrument).file
        temporal = os.path.join(self.beammapdirectory, self.stage1_bmaps)

        _, _, self.x_locs, self.y_locs = bmu.shapeBeammapIntoImages(initial, temporal)

    def loadSweepBins(self, s, get_phases=True):
        """

        :param s: configdict!
            object containing single sweep info
        :param get_phases: bool
            Make phase images in addition to the intensity images (much slower)
        :return:
        """
        cachefile = os.path.join(self.beammapdirectory,
                                 self.config.beammap.sweep.cachename.format(s.starttime, s.duration, get_phases))

        try:
            images = np.load(cachefile)
            msg = 'Restored sweep images for {} s starting at {} from {}'
            getLogger('Sweep').info(msg.format(s.duration, s.starttime, cachefile))
            return images[images.keys()[0]]
        except IOError:
            pass

        startTime = s.starttime
        duration = s.duration
        if duration % 2 == 0:
            getLogger('Sweep').warn("Having an even number of time steps can create off by 1 errors: "
                                    "subtracting one time step to make it odd")
            duration -= 1

        arglist = [(os.path.join(self.config.paths.bin, '{}.bin'.format(start)),
                    self.numrows, self.numcols)
                   for start in range(startTime, startTime+duration)]

        pool = mp.Pool(self.config.beammap.ncpu)
        if get_phases:
            images = np.array(pool.map(bin2imgs, arglist))  # intensity and phase maps
        else:
            images = np.array(pool.map(bin2img, arglist))  # just intensity maps
        pool.close()
        pool.join()

        np.savez(cachefile, images)

        return images

    def manualSweepCleanup(self, feedline):
        if self.initial_bmap is None:
            initial_bmap = Beammap(default=self.config.beammap.instrument).file
        else:
            initial_bmap = os.path.join(self.beammapdirectory, self.initial_bmap)

        stage1_bmap = self.get_FL_filename(self.stage1_bmaps, feedline)
        stage2_bmap = self.get_FL_filename(self.stage2_bmaps, feedline)
        getLogger('Sweep').info('beammap to click: {}'.format(stage1_bmap))
        m = ManualTemporalBeammap(self.x_images, self.y_images, initial_bmap, stage1_bmap, stage2_bmap,
                                  self.config.beammap.sweep.fittype, self.xp_images, self.yp_images)

    def get_FL_filename(self, fn, FL):
        path = self.beammapdirectory
        name, extension = fn.split('.')
        name = name.format(FL)
        return os.path.join(path, '.'.join([name, extension]))

    def combineClicked(self):
        """
        Concatenate all relevant info from FL files after they've been verified by a human, and output to file
        stage3_bmap

        Currently, for unused feedlines duplicate and rename each one eg stage1_bmap_FL2.txt -> stage2_bmap_FL2.txt,
        run --combo, and then in stage3_bmap swap those feedlines for their equivalent data with the noDacTone flag
        applied from one of the stage1_bmap_FL{}.txt files.

        Return
        ------
            stage3_bmap a .txt file containing the contents of all the manual clickthroughs
        """

        stage3_bmap = os.path.join(self.beammapdirectory, self.config.beammap.filenames.stage3_bmap)

        filenames = [self.get_FL_filename(self.stage2_bmaps, fl) for fl in range(1, self.numfeed+1)]

        masterfile = open(stage3_bmap,'a')
        for fl, fname in enumerate(filenames, 1):
            FL_data = np.loadtxt(fname)
            args = np.int_(FL_data[:, 0] / 10000) == fl
            np.savetxt(masterfile, FL_data[args], fmt='%7d %3d %7f %7f')

        log.info('Combined contents of the clicked FL beammaps to {}'.format(stage3_bmap))

    def stitchAligned(self, by_id=True):
        """ This function is used when stages 1-4 have been completed on a series of boards separately and need to be
         stitched before stage 5 (clean). For example, the input file 20191211_stage4_FL7b.txt has the content

         resID | Flag | x (secs) | y (secs)
           .      .        .          .
           .      .        .          .
           .      .        .          .
         71024    1     66.66377   3.25130
         71025    0     152.65801  127.66477
           .      .        .          .
           .      .        .          .
           .      .        .          .
         72043    0     66.66377   3.25130
         80000    1     66.66377   3.25130
           .      .        .          .
           .      .        .          .
           .      .        .          .

        Returns
        -------
        20191211_stage4_stitched.txt
        """
        stage4_suffix = self.config.beammap.filenames.stage4_bmap.split('.')[0]
        stage4_stitched_bmap = os.path.join(self.beammapdirectory, stage4_suffix+'_stitched.txt')
        if os.path.exists(stage4_stitched_bmap):
            os.remove(stage4_stitched_bmap)

        filenames = ['20191211_stage4_FL7b.txt', '20191211_stage4_FL6b.txt']
        masterfile = open(stage4_stitched_bmap, 'a')
        for fname in filenames:
            board_data = np.loadtxt(os.path.join(self.beammapdirectory, fname))
            if by_id:
                board_id = fname.split('.')[0][-2:]
                flnum = int(board_id[0])
                rel_start = 0 if board_id[1] == 'a' else 1024
                abs_start = 10000 * flnum + rel_start
                abs_end = abs_start + 1024
                fl_ind = (abs_start < board_data[:, 0]) & (board_data[:, 0] < abs_end)
            else:
                fl_ind = board_data[:,1] == 0
            np.savetxt(masterfile, board_data[fl_ind], fmt='%7d %3d %7f %7f')

        log.info('Combined contents of aligned board beammaps to {}'.format(stage4_stitched_bmap))

    def plotTimestream(self):
        pass


if __name__ == '__main__':

    create_log('Sweep')
    create_log('mkidcore')
    create_log('mkidreadout', level='DEBUG')
    log = getLogger('Sweep')

    parser = argparse.ArgumentParser(description='MKID Beammap Analysis Utility')
    parser.add_argument('-c', '--config', default=mkidreadout.config.DEFAULT_SWEEP_CFGFILE, dest='config',
                        type=str, help='The config file')
    parser.add_argument('--gencfg', default=False, dest='genconfig', action='store_true',
                        help='generate config in CWD')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--xcor', default=False, action='store_true', dest='use_cc',
                       help='Run cross correlation (step 1)')
    group.add_argument('--simple-locate', default=False, action='store_true', dest='use_simple',
                       help='Run argmax to get peak location estimates (step 1 alt.)')
    group.add_argument('--manual', default=False, dest='manual_idx',
                       help='Run manual sweep cleanup (step 2)')
    group.add_argument('--combo', default=False, action='store_true', dest='use_combo',
                       help='Combines the different FLs into one master (step 3)')
    group.add_argument('--align', default=False, action='store_true', dest='align', help='Run align grid (step 4)')
    group.add_argument('--clean', default=False, action='store_true', dest='clean', help='Run clean (step 5)')

    group.add_argument('--stitch-aligned', default=False, action='store_true', dest='stitch_aligned', help='stitch board align (step 4.5)')

    args = parser.parse_args()

    if args.genconfig:
        mkidreadout.config.generate_default_configs(sweep=True)
        exit(0)

    config = load(args.config)

    b = TemporalBeammap(config)

    log.info('Starting temporal beammap')
    if args.use_cc:  # Cross correlation mode
        b.loadTemporalBeammap()
        b.concatImages('x',False)
        b.concatImages('y',False)
        b.findLocWithCrossCorrelation('x')
        b.findLocWithCrossCorrelation('y')
        b.refinePeakLocs('x', b.config.beammap.sweep.fittype, b.x_locs, fitWindow=15)
        b.refinePeakLocs('y', b.config.beammap.sweep.fittype, b.y_locs, fitWindow=15)
        b.saveTemporalBeammap()
    elif args.use_simple:  # Alternative to cross cor. Only works on single sweeps
        b.loadTemporalBeammap()
        b.concatImages('x',False)
        b.concatImages('y',False)
        b.refinePeakLocs('x', b.config.beammap.sweep.fittype, None, fitWindow=15)
        b.refinePeakLocs('y', b.config.beammap.sweep.fittype, None, fitWindow=15)
        b.saveTemporalBeammap()
    elif args.manual_idx:  # Manual mode
        log.info('Starting manual verification for feedline %s' % args.manual_idx)
        b.stackImages('x')
        b.stackImages('y')
        log.info('Cleanup')
        b.manualSweepCleanup(feedline=args.manual_idx)
    elif args.use_combo:  # Combine clicked FL beam files
        b.combineClicked()
    elif args.align:
        aligner = bmap_align.BMAligner(config.paths.beammapdirectory, config.beammap.filenames.stage3_bmap,
                                       config.beammap.align.cachename, config.beammap.instrument, config.beammap.flip)
        aligner.makeTemporalImage()
        aligner.loadFFT()
        aligner.findKvecsManual()
        aligner.findAngleAndScale()
        aligner.rotateAndScaleCoords()
        aligner.findOffset(100000)
        aligner.plotCoords()
        aligner.saveTemporalMap(os.path.join(config.paths.beammapdirectory, config.beammap.filenames.stage4_bmap))
        # if config.paths.masterdoubleslist is not None:
        #     aligner.makeDoublesTemporalMap(os.path.join(config.paths.beammapdirectory, config.paths.masterdoubleslist),
        #                                    os.path.join(config.paths.beammapdirectory, config.paths.outputdoublename))
    elif args.clean:
        bmap_clean.main(config)
    elif args.stitch_aligned:
        b.stitchAligned()
