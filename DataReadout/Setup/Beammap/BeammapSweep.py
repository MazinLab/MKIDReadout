"""
Author: Alex Walter
Date: June 5, 2018

This file contains classes and functions used to create a rough beammap
A rough beammap is the following format:
resID   flag    time_x  time_y
[int    int     float   float]

A regular beammap is:
resID   flag    loc_x   loc_y
[int    int     int     int]


Classes in this file:
BeamSweep1D(imageList, pixelComputationMask=None, minCounts=5, maxCountRate=2499)
ManualRoughBeammap(x_images, y_images, initialBeammap, roughBeammapFN)
RoughBeammap(configFN)

Usage:


"""

import sys, os, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

from numba import jit
import ConfigParser

from MkidDigitalReadout.DataReadout.Setup.Beammap.utils import crossCorrelateTimestreams,  determineSelfconsistentPixelLocs2, loadImgFiles, minimizePixelLocationVariance, getPeak, shapeBeammapIntoImages
from MkidDigitalReadout.DataReadout.Setup.Beammap.beammapFlags import beamMapFlags

#import pdb

class BeamSweep1D():
    """
    This class is for computing a rough beammap using a list of images
    
    It uses a complicated autocorrelation function to find the pixel locations
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
        self.imageList=np.asarray(imageList)

        #Use these parameters to determine what's a good pixel
        self.minCounts = minCounts       # counts during total exposure
        self.maxCountRate = maxCountRate # counts per image frame

        nPix = np.prod(self.imageList[0].shape)
        nTime=len(self.imageList)
        bkgndList = 1.0*np.median(self.imageList,axis=0)
        nCountsList = 1.0*np.sum(self.imageList,axis=0)
        maxCountsList = 1.0*np.amax(self.imageList,axis=0)
        badPix = np.where(np.logical_not((nCountsList>minCounts) * (maxCountsList<maxCountRate) * (bkgndList < nCountsList/nTime)))
        
        if pixelComputationMask is None:
            nGoodPix = nPix - len(badPix[0])
            #nGroups=np.prod(imageList.shape)*(np.prod(imageList[0].shape)-1)/(200*3000*2999)*nGoodPix/nPix     # 300 timesteps x 3000 pixels takes a lot of memory...
            nGroups = nTime*nGoodPix*(nGoodPix-1)/(600*3000*2999)
            pixelComputationMask=np.random.randint(0,int(round(nGroups)),imageList[0].shape)
            #pixelComputationMask=np.repeat(range(5),2000).reshape(imageList[0].shape)
        self.compMask = np.asarray(pixelComputationMask)
        if len(badPix[0])>0:
            self.compMask[badPix]=np.amax(self.compMask)+1    #remove bad pixels from the computation
            self.compGroups = (np.unique(self.compMask))[:-1]
        else: self.compGroups = np.unique(self.compMask)


    
    def getAbsOffset(self,shiftedTimes, auto=False):
        """
        The autocorrelation function can only calculate relative time differences
        between pixels. This function defines the absolute time reference (ie. the 
        location of the peak)

        INPUTS:
            shiftedTimes: a list of pixel time streams shifted to match up
            auto: if False then ask user to click on a plot
        """
        offset = np.argmax(np.sum(shiftedTimes,axis=0))
        if auto: return offset
        
        print "Please click the correct peak"
        fig, ax = plt.subplots()
        for p_i in range(len(shiftedTimes)):
            ax.plot(shiftedTimes[p_i])
        ax.plot(np.sum(shiftedTimes,axis=0), 'k-')
        ln =ax.axvline(offset,c='b')
        def onclick(event):
            if fig.canvas.manager.toolbar._active is None:
                offset=event.xdata
                print offset
                ln.set_xdata(offset)
                plt.draw()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        return offset
        

    def findRelativePixelLocations(self):
        """
        Use auto correllation and least squares to find best time offsets between pixels
        """
        try: locs=np.empty(self.imageList[0].shape)
        except TypeError: return []
        locs[:]=np.nan

        for g in self.compGroups:
        #for g in [0]:
            print 'Starting group ', g
            compPixels = np.where(self.compMask==g)
            
            timestreams=np.transpose(self.imageList[:,compPixels[0],compPixels[1]])     # shape [nPix, nTime]
            correlationList, goodPix = crossCorrelateTimestreams(timestreams, self.minCounts, self.maxCountRate)
            if len(goodPix) == 0: continue
            correlationLocs = np.argmax(correlationList,axis=1)
            
            correlationQaulity = 1.0*np.amax(correlationList,axis=1)/np.sum(correlationList,axis=1)
            #correlationQuality = np.sum((correlationList[:,:len(goodPix)/2] - (correlationList[:,-1:-len(goodPix)/2-1:-1]))**2.,axis=1)     #subtract the mirror, square, and sum. If symmetric then it should be near 0
            #pdb.set_trace()
            del correlationList

            print "Making Correlation matrix..."
            corrMatrix = np.zeros((len(goodPix),len(goodPix)))
            corrMatrix[np.triu_indices(len(goodPix),1)] = correlationLocs - len(self.imageList)/2
            corrMatrix[np.tril_indices(len(goodPix),-1)] = -1*np.transpose(corrMatrix)[np.tril_indices(len(goodPix),-1)]
            del correlationLocs
            corrQualityMatrix = np.ones((len(goodPix),len(goodPix)))
            corrQualityMatrix[np.triu_indices(len(goodPix),1)] = correlationQaulity
            corrQualityMatrix[np.tril_indices(len(goodPix),-1)] = -1*np.transpose(corrQualityMatrix)[np.tril_indices(len(goodPix),-1)]
            del correlationQaulity
            print "Done..."

            print "Finding Best Relative Locations..."
            a = minimizePixelLocationVariance(corrMatrix)
            bestPixelArgs, totalVar = determineSelfconsistentPixelLocs2(corrMatrix, a)
            bestPixels = goodPix[bestPixelArgs]
            bestPixels = bestPixels[: len(bestPixels)/20]
            best_a = minimizePixelLocationVariance(corrMatrix[:,np.where(np.in1d(goodPix,bestPixels))[0]])
            print "Done..."

            print "Finding Timestream Peak Locations..."
            shifts = np.rint(best_a[bestPixelArgs]).astype(np.int)
            shifts=shifts[: len(bestPixels)]
            shifts = shifts[:,None]+np.arange(len(timestreams[0]))
            shifts[np.where(shifts<0)]=-1
            shifts[np.where(shifts>=len(timestreams[0]))]=-1
            bkgndList = 1.0*np.median(timestreams[bestPixels],axis=1)
            nCountsList = 1.0*np.sum(timestreams[bestPixels],axis=1)
            shiftedTimes = np.zeros((len(bestPixels),len(timestreams[0])+1))
            shiftedTimes[:,:-1]=(timestreams[bestPixels]-bkgndList[:,None])/nCountsList[:,None]  #padded timestream with 0
            shiftedTimes=shiftedTimes[np.arange(len(bestPixels))[:,None], shifts]   #shift each timestream
            #offset = np.argmax(np.sum(shiftedTimes,axis=0))
            offset=self.getAbsOffset(shiftedTimes)
            del shiftedTimes

            best_a+=offset

            print "Done..."

            locs[compPixels[0][goodPix],compPixels[1][goodPix]] = best_a


        locs[np.where(locs<0)]=0
        locs[np.where(locs>=len(self.imageList))]=len(self.imageList)
        return locs


class ManualRoughBeammap():
    def __init__(self, x_images, y_images, initialBeammap, roughBeammapFN):
        """
        Class for manually clicking through beammap. 
        Saves a rough beammap with filename roughBeammapFN-HHMMSS.txt
        A 'rough' beammap is one that doesn't have x/y but instead the peak location in time from the swept light beam. 
        
        INPUTS:
            x_images - list of images for sweep(s) in x-direction
            y_images - 
            initialBeammap - path+filename of initial beammap used for making images
            roughBeammapFN - path+filename of the rough beammap (time at peak instead of x/y value)
                             If the roughBeammap doesn't exist then it will be instantiated with nans
                             We append a timestamp to this string as the output file
        """
        self.x_images=x_images
        self.y_images=y_images
        totalCounts_x = np.sum(self.x_images,axis=0)
        totalCounts_y = np.sum(self.y_images,axis=0)
        self.nTime_x=len(self.x_images)
        self.nTime_y=len(self.y_images)
        
        self.initialBeammapFN=initialBeammap
        self.roughBeammapFN=roughBeammapFN
        self.outputBeammapFn = roughBeammapFN.rsplit('.',1)[0]+time.strftime('-%H%M%S')+'.txt'
        self.resIDsMap, self.flagMap, self.x_loc, self.y_loc = shapeBeammapIntoImages(self.initialBeammapFN,self.roughBeammapFN)
        if self.roughBeammapFN is None or not os.path.isfile(self.roughBeammapFN):
            self.flagMap[np.where(self.flagMap!=beamMapFlags['noDacTone'])]=beamMapFlags['failed']

        
        self.goodPix = np.where((self.flagMap!=beamMapFlags['noDacTone']) * (totalCounts_x+totalCounts_y)>0)
        self.nGoodPix = len(self.goodPix[0])
        print 'Pixels with light: ',self.nGoodPix
        self.curPixInd=0
        self.curPixValue = np.amax(beamMapFlags.values())+1
        

        self._want_to_close=False
        self.plotFlagMap()
        self.plotXYscatter()
        self.plotTimestream()
        
        plt.show()
    
    def saveRoughBeammap(self):
        print 'Saving: ', self.outputBeammapFn
        allResIDs = self.resIDsMap.flatten()
        flags = self.flagMap.flatten()
        x = self.x_loc.flatten()
        y = self.y_loc.flatten()
        args = np.argsort(allResIDs)
        data=np.asarray([allResIDs[args], flags[args], x[args], y[args]]).T
        np.savetxt(self.outputBeammapFn, data, fmt='%7d %3d %7f %7f')
    
    def plotTimestream(self):
        self.fig_time, (self.ax_time_x, self.ax_time_y) = plt.subplots(2)
        y=self.goodPix[0][self.curPixInd]
        x=self.goodPix[1][self.curPixInd]
        
        
        xStream=self.x_images[:,y,x]
        ln_xStream, = self.ax_time_x.plot(xStream)
        x_loc=self.x_loc[y,x]
        if np.logical_not(np.isfinite(x_loc)): x_loc=-1
        ln_xLoc=self.ax_time_x.axvline(x_loc,c='r')
        
        yStream=self.y_images[:,y,x]
        ln_yStream,=self.ax_time_y.plot(yStream)
        y_loc=self.y_loc[y,x]
        if np.logical_not(np.isfinite(y_loc)): y_loc=-1
        ln_yLoc=self.ax_time_y.axvline(y_loc,c='r')
        
        #self.ax_time_x.set_title('Timestreams')
        self.ax_time_x.set_title('Pix '+str(self.curPixInd)+' resID'+str(int(self.resIDsMap[y,x]))+' ('+str(x)+', '+str(y)+')')
        self.ax_time_x.set_ylabel('X counts')
        self.ax_time_y.set_ylabel('Y counts')
        self.ax_time_y.set_xlabel('Timesteps')
        self.ax_time_x.set_xlim(-2, self.nTime_x+2)
        self.ax_time_y.set_xlim(-2, self.nTime_y+2)
        self.ax_time_x.set_ylim(0, None)
        self.ax_time_y.set_ylim(0, None)
        
        self.timestreamPlots=[ln_xLoc, ln_yLoc, ln_xStream, ln_yStream]
        
        self.fig_time.canvas.mpl_connect('button_press_event', self.onClickTime)
        self.fig_time.canvas.mpl_connect('close_event', self.onCloseTime)
        self.fig_time.canvas.mpl_connect('key_press_event', self.onKeyTime)
    
    def onKeyTime(self, event):
        print 'Pressed ',event.key
        #if event.key not in ('right', 'left'): return
        if event.key in ['right','c']: 
            self.curPixInd+=1
            self.curPixInd%=self.nGoodPix
            self.updateTimestreamPlot()
        elif event.key in ['left']: 
            self.curPixInd-=1
            self.curPixInd%=self.nGoodPix
            self.updateTimestreamPlot()
        elif event.key in ['b']:
            y=self.goodPix[0][self.curPixInd]
            x=self.goodPix[1][self.curPixInd]
            self.x_loc[y,x]=np.nan
            self.y_loc[y,x]=np.nan
            print 'Pix '+str(int(self.resIDsMap[y,x]))+' ('+str(x)+', '+str(y)+') Marked bad'
            self.curPixInd+=1
            self.curPixInd%=self.nGoodPix
            self.updateTimestreamPlot()
            self.updateXYPlot(2)
            self.updateFlagMap(self.curPixInd)
    
    def onCloseTime(self,event):
        if not self._want_to_close:
            self.curPixInd+=1
            self.curPixInd%=self.nGoodPix
            self.updateTimestreamPlot()
            self.plotTimestream()
            plt.show()
        #event.ignore()
        #self.fig_time.show()
    
    def updateTimestreamPlot(self, lineNum=4):
        y=self.goodPix[0][self.curPixInd]
        x=self.goodPix[1][self.curPixInd]
        
        if lineNum==0 or lineNum>=4:
            offset=self.x_loc[y,x]
            if np.logical_not(np.isfinite(offset)): offset=-1
            self.timestreamPlots[0].set_xdata(offset)
        if lineNum==1 or lineNum>=4:
            offset=self.y_loc[y,x]
            if np.logical_not(np.isfinite(offset)): offset=-1
            self.timestreamPlots[1].set_xdata(offset)
        if lineNum==2 or lineNum>=4:
            self.ax_time_x.set_title('Pix '+str(self.curPixInd)+' resID'+str(int(self.resIDsMap[y,x]))+' ('+str(x)+', '+str(y)+')')
            self.timestreamPlots[2].set_ydata(self.x_images[:,y,x])
            #self.ax_time_x.autoscale(True,'y',True)
            self.ax_time_x.set_ylim(0, 1.05*np.amax(self.x_images[:,y,x]))
            self.ax_time_x.set_xlim(-2, self.nTime_x+2)
        if lineNum==3 or lineNum>=4:
            self.ax_time_x.set_title('Pix '+str(self.curPixInd)+' resID'+str(int(self.resIDsMap[y,x]))+' ('+str(x)+', '+str(y)+')')
            self.timestreamPlots[3].set_ydata(self.y_images[:,y,x])
            #self.ax_time_y.autoscale(True,'y',True)
            self.ax_time_y.set_ylim(0, 1.05*np.amax(self.y_images[:,y,x]))
            self.ax_time_y.set_xlim(-2, self.nTime_y+2)
        
        self.fig_time.canvas.draw()
        
    
    def onClickTime(self,event):
        if self.fig_time.canvas.manager.toolbar._active is None:
            #update time plot and x/y_loc
            y=self.goodPix[0][self.curPixInd]
            x=self.goodPix[1][self.curPixInd]
            offset=event.xdata
            if offset<0: offset=np.nan
            if event.inaxes == self.ax_time_x:
                offset=getPeak(self.x_images[:,y,x],offset)
                #if offset>=self.nTime_x: offset=np.nan
                self.x_loc[y,x]=offset
                print 'x: ',offset
                self.updateTimestreamPlot(0)
            elif event.inaxes == self.ax_time_y:
                #if offset>=self.nTime_y: offset=np.nan
                offset=getPeak(self.y_images[:,y,x],offset)
                self.y_loc[y,x]=offset
                print 'y: ',offset
                self.updateTimestreamPlot(1)

            self.updateXYPlot(2)
            self.updateFlagMap(self.curPixInd)

        
    def updateXYPlot(self, lines):
        if lines==0 or lines==2:
            self.ln_XY.set_data(self.x_loc.flatten(), self.y_loc.flatten())
        if lines==1 or lines==2:
            y=self.goodPix[0][self.curPixInd]
            x=self.goodPix[1][self.curPixInd]
            self.ln_XYcur.set_data([self.x_loc[y,x]], [self.y_loc[y,x]])
        
        self.ax_XY.autoscale(True)
        self.fig_XY.canvas.draw()

    def updateFlagMap(self, curPixInd):
        y=self.goodPix[0][curPixInd]
        x=self.goodPix[1][curPixInd]
        if np.isfinite(self.x_loc[y,x]) * np.isfinite(self.y_loc[y,x]): self.flagMap[y,x]=beamMapFlags['good']
        elif np.logical_not(np.isfinite(self.x_loc[y,x])) * np.isfinite(self.y_loc[y,x]): self.flagMap[y,x]=beamMapFlags['xFailed']
        elif np.isfinite(self.x_loc[y,x]) * np.logical_not(np.isfinite(self.y_loc[y,x])): self.flagMap[y,x]=beamMapFlags['yFailed']
        elif np.logical_not(np.isfinite(self.x_loc[y,x])) * np.logical_not(np.isfinite(self.y_loc[y,x])): self.flagMap[y,x]=beamMapFlags['failed']
        self.saveRoughBeammap()
        self.updateFlagMapPlot()
        
    def updateFlagMapPlot(self):
        flagMap_masked = np.ma.masked_where(self.flagMap==beamMapFlags['noDacTone'], self.flagMap)
        flagMap_masked[self.goodPix[0][self.curPixInd],self.goodPix[1][self.curPixInd]]=self.curPixValue
        self.ln_flags.set_data(flagMap_masked)
        self.fig_flags.canvas.draw()
    
    def plotFlagMap(self):
        self.fig_flags, self.ax_flags = plt.subplots()
        flagMap_masked = np.ma.masked_where(self.flagMap==beamMapFlags['noDacTone'], self.flagMap)
        my_cmap = matplotlib.cm.get_cmap('YlOrRd')
        my_cmap.set_under('w')
        my_cmap.set_over('c')
        my_cmap.set_bad('k')
        
        flagMap_masked[self.goodPix[0][self.curPixInd],self.goodPix[1][self.curPixInd]]=self.curPixValue
        self.ln_flags = self.ax_flags.matshow(flagMap_masked, cmap=my_cmap,vmin=0.1, vmax=np.amax(self.flagMap)+.1)
        self.ax_flags.set_title('Flag map')
        #cbar = self.fig_flags.colorbar(flagMap_masked, extend='both', shrink=0.9, ax=self.ax_flags)
        self.fig_flags.canvas.mpl_connect('button_press_event', self.onClickFlagMap)
        self.fig_flags.canvas.mpl_connect('close_event', self.onCloseFlagMap)
    
    def onCloseFlagMap(self,event):
        #plt.close(self.fig_XY)
        #plt.close(self.fig_time)
        plt.show()
        self._want_to_close=True
        plt.close('all')

        
    def onClickFlagMap(self,event):
        if self.fig_flags.canvas.manager.toolbar._active is None and event.inaxes == self.ax_flags:
            x=int(np.floor(event.xdata+0.5))
            y=int(np.floor(event.ydata+0.5))
            nRows, nCols = self.x_images[0].shape
            if x>=0 and x<nCols and y>=0 and y<nRows:
                print 'Clicked Flag Map! [',x,', ',y,'] -> ',self.resIDsMap[y,x], ' Flag: ',self.flagMap[y,x]
                pixInd = np.where( (self.goodPix[0]==y) * (self.goodPix[1]==x))
                if len(pixInd[0])==1:
                    self.curPixInd = pixInd[0]
                    self.updateFlagMapPlot()
                    self.updateXYPlot(1)
                    self.updateTimestreamPlot(4)
                    #plt.draw()
                else: print "No photons detected"
                    
    
    def plotXYscatter(self):
        self.fig_XY, self.ax_XY = plt.subplots()
        self.ln_XY, = self.ax_XY.plot(self.x_loc.flatten(), self.y_loc.flatten(),'b.')
        y=self.goodPix[0][self.curPixInd]
        x=self.goodPix[1][self.curPixInd]
        self.ln_XYcur, = self.ax_XY.plot([self.x_loc[y,x]], [self.y_loc[y,x]],'go')
        self.ax_XY.set_title('Pixel Locations')
        self.fig_XY.canvas.mpl_connect('close_event', self.onCloseXY)
    
    def onCloseXY(self,event):
        #self.fig_XY.show()
        if not self._want_to_close:
            self.plotXYscatter()
            plt.show()

class RoughBeammap():
    def __init__(self, configFN):
        """
        This class is for finding the rough location of each pixel in units of timesteps
        INPUTS:
            configFN - config file listing the sweeps and properties
        """
        self.config = ConfigParser.ConfigParser()
        self.config.read(configFN)
        self.x_locs=None
        self.y_locs=None
        self.x_images=None
        self.y_images=None

    def stackImages(self, sweepType):
        """
        Should add option for median
        """
        assert sweepType.lower() in ['x','y']
        imageList = None
        nTimes=0
        nSweeps=0.
        for s in self.config.sections():
            if s.startswith("SWEEP") and self.config.get(s,'sweepType').lower() in [sweepType.lower()]:
                nSweeps+=1.
                sweepNum = int(s.rsplit('SWEEP',1)[-1])
                imList = self.loadSweepImgs(sweepNum)
                direction=1
                if self.config.get(s,'sweepDirection') in ['-']: direction=-1
                if imageList is None: 
                    imageList=imList[::direction,:,:]
                    nTimes=len(imageList)
                else: 
                    nTimes = min(len(imList), nTimes)
                    imageList=imageList[:nTimes] + (imList[::direction,:,:])[:nTimes]
        if sweepType in ['x','X']:
            self.x_images=imageList/nSweeps
        else:
            self.y_images=imageList/nSweeps
        print 'stacked',nSweeps, sweepType, 'sweeps'
        return imageList
        
    def concatImages(self, sweepType):
        """
        This won't work well if the background level or QE of the pixel changes between sweeps...
        Should remove this first
        """
        assert sweepType.lower() in ['x','y']
        imageList = None
        for s in self.config.sections():
            if s.startswith("SWEEP") and self.config.get(s,'sweepType').lower() in [sweepType.lower()]:
                print 'loading: '+str(s)
                sweepNum = int(s.rsplit('SWEEP',1)[-1])
                imList = self.loadSweepImgs(sweepNum)
                direction=1
                if self.config.get(s,'sweepDirection') in ['-']: direction=-1
                if imageList is None: imageList=imList[::direction,:,:]
                else: imageList=np.concatenate((imageList, imList[::direction,:,:]),axis=0)
        if sweepType in ['x','X']:
            self.x_images=imageList
        else:
            self.y_images=imageList
        return imageList

    def computeSweeps(self, sweepType, pixelComputationMask=None):
        """
        Careful: We assume the sweep speed is always the same!!!
        For now, we assume initial beammap is the same too.
        """
        imageList=self.concatImages(sweepType)
        sweep = BeamSweep1D(imageList,pixelComputationMask)
        locs=sweep.findRelativePixelLocations()
        if sweepType in ['x','X']: self.x_locs=locs
        else: self.y_locs=locs
        self.saveRoughBeammap()
        
    
    def saveRoughBeammap(self):

        print 'Saving'
        allResIDs_map, flag_map, x_map, y_map = shapeBeammapIntoImages(self.config.get('DEFAULT','initialBeammap'), self.config.get('DEFAULT','roughBeammap'))
        otherFlagArgs = np.where((flag_map!=beamMapFlags['good']) * (flag_map!=beamMapFlags['failed']) * (flag_map!=beamMapFlags['xFailed']) * (flag_map!=beamMapFlags['yFailed']))
        otherFlags = flag_map[otherFlagArgs]
        if self.y_locs is not None and self.y_locs.shape==flag_map.shape:
            y_map=self.y_locs
            print 'added y'
        if self.x_locs is not None and self.x_locs.shape==flag_map.shape:
            x_map=self.x_locs
            print 'added x'

        flag_map[np.where(np.logical_not(np.isfinite(x_map)) * np.logical_not(np.isfinite(y_map)))] = beamMapFlags['failed']
        flag_map[np.where(np.logical_not(np.isfinite(x_map)) * np.isfinite(y_map))] = beamMapFlags['xFailed']
        flag_map[np.where(np.logical_not(np.isfinite(y_map)) * np.isfinite(x_map))] = beamMapFlags['yFailed']
        flag_map[np.where(np.isfinite(x_map) * np.isfinite(y_map))] = beamMapFlags['good']
        flag_map[otherFlagArgs]=otherFlags

        allResIDs = allResIDs_map.flatten()
        flags = flag_map.flatten()
        x = x_map.flatten()
        y = y_map.flatten()
        args = np.argsort(allResIDs)
        data=np.asarray([allResIDs[args], flags[args], x[args], y[args]]).T
        np.savetxt(self.config.get('DEFAULT','roughBeammap'), data, fmt='%7d %3d %7f %7f')
        

    def loadSweepImgs(self, sweepNum):
        path = self.config.get('SWEEP'+str(sweepNum),'imgFileDirectory')
        startTime = self.config.getint('SWEEP'+str(sweepNum),'startTime')
        duration = self.config.getint('SWEEP'+str(sweepNum),'duration')
        if duration%2==0:
            warnings.warn("Having an even number of time steps can create off by 1 errors: subtracting one time step to make it odd")
            duration-=1
        fnList = [path+str(startTime + i)+'.img' for i in range(duration)]
        nRows = self.config.getint('SWEEP'+str(sweepNum),'numRows')
        nCols = self.config.getint('SWEEP'+str(sweepNum),'numCols')
        return loadImgFiles(fnList, nRows, nCols)
    
    def manualSweepCleanup(self):
        m=ManualRoughBeammap(self.x_images, self.y_images, self.config.get('DEFAULT','initialBeammap'), self.config.get('DEFAULT','roughBeammap'))
        
    
    def plotTimestream(self):
        pass

if __name__=='__main__':
    #configFN = 'beam3.cfg'
    try:
        configFN = sys.argv[1]
    except IndexError:
        print "Usage: $python BeammapSweep.py config.cfg"
        raise ValueError
    
    b=RoughBeammap(configFN)
    #b.computeSweeps('x')
    #b.computeSweeps('y')
    #b.concatImages('x')
    #b.concatImages('y')
    b.stackImages('x')
    b.stackImages('y')
    #print b.config.get('DEFAULT','initialBeammap')
    #print b.config.get('DEFAULT','roughBeammap')
    b.manualSweepCleanup()
    


