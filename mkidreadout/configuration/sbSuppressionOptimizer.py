"""
Code to minimize sideband amplitude in the ADC by applying offsets to the
relative phases and amplitudes of the I and Q signals for each tone.

Usage: python sbSuppressionOptimizer.py -c <templarcfg> <roachNums>
    templarcfg: High Templar config file
    roachNums: Last 3 digits of roach ip (or "Roach" section number in templarcfg)

This should be run when readout/array are in final configuration, as the optimization
is sensitive to tone power.

Author: Neelay Fruitwala

"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

import os
import sys
import threading
import time

from mkidreadout.channelizer.Roach2Controls import Roach2Controls
from mkidreadout.configuration.sweepdata import SweepMetadata


class SBOptimizer:
    """
    Class for optimizing ADC sideband suppression, by generating phase and amplitude offsets between
    DAC I and Q signals.

    Functions:
        initRoach: initializes communication with ROACH2 and ADCDAC boards
        loadLUT: loads DAC LUT
        takeAdcSnap: takes a snapshot of ADC signal
        gridSearchOptimizerFit: "main" function; implements the optimization algorithm
        saveGridSearchOptFreqList: saves a frequency file containing phase and I/Q corrections


    """
            
    def __init__(self, ip, metadata, loFreq, adcAtten=None, globalDacAtten=None):
        """
        INPUTS:
            ip: ROACH2 IP Address
            params: Roach2Controls parameter file
            loFreq: LO Frequency in Hz
            adcAtten: attenuation value (dB) of IF board ADC attenuator
            global Dac Atten: physical DAC attenuation
            freqList: list of tone frequencies in frequency comb to optimize 
                        (should be entire comb; no need to divide into separate sidebands
            resIDList: list of corresponding resonator IDs
            toneAttenList: list of corresponding resonator attenuation values
        """
        self.roach = Roach2Controls(ip, params, True, False)
        self.loFreq = loFreq
        self.toneAttenList = toneAttenList
        self.resIDList, self.freqList, toneAttenList = metadata.templar_data(loFreq)

        aboveLOInds = np.where(freqList>=loFreq)[0]
        belowLOInds = np.where(freqList<loFreq)[0]
        self.freqListLow = freqList[belowLOInds]
        self.freqListHigh = freqList[aboveLOInds]
        self.toneAttenListLow = toneAttenList[belowLOInds]
        self.toneAttenListHigh = toneAttenList[aboveLOInds]
        self.finalPhaseListLow = np.zeros(len(self.freqListLow))
        self.finalPhaseListHigh = np.zeros(len(self.freqListHigh))
        self.finalIQRatioListLow = np.ones(len(self.freqListLow))
        self.finalIQRatioListHigh = np.ones(len(self.freqListHigh))
        self._initRoach(globalDacAtten, adcAtten)

    def _initRoach(self, globalDacAtten, adcAtten):
        """
        Initializes the ROACH2: connects, initializes the UART, loads the LO and physical attenuations
        """
        self.roach.connect()
        self.roach.setLOFreq(self.loFreq)
        self.roach.initializeV7UART(waitForV7Ready=False)
        self.roach.loadLOFreq()
        self.roach.loadFullDelayCal()

        if globalDacAtten is None or adcAtten is None:
            atten = self.loadLUT(globalDacAtten=globalDacAtten)
            globalDacAtten = atten

        if adcAtten is None:
            self.adcAtten = self.roach.getOptimalADCAtten()

        else:
            self.adcAtten = adcAtten
            self.roach.changeAtten(3,np.floor(self.adcAtten*2)/4.)
            self.roach.changeAtten(4,np.ceil(self.adcAtten*2)/4.)

        #self.roach.changeAtten(1,np.floor(self.globalDacAtten/31.75)*31.75)
        #self.roach.changeAtten(2,self.globalDacAtten%31.75)

    def loadLUT(self, sideband='all', phaseList=None, iqRatioList=None, globalDacAtten=None):
        """
        Loads DAC LUT
        INPUTS:
            sideband: specifies which sideband to load:
                    all: full LUT
                    lower: frequencies below LO
                    upper: frequencies above LO
            phaseList: list of phase offsets at each frequency
            iqRatioList: list of I/Q amplitude ratios at each frequency
            No offsets are loaded if the above two parameters are 'None'
        """
        if sideband=='all':
            freqList = self.freqList
            attenList = self.toneAttenList
        elif sideband=='lower':
            freqList = self.freqListLow
            attenList = self.toneAttenListLow
        elif sideband=='upper':
            freqList = self.freqListHigh
            attenList = self.toneAttenListHigh
        else:
            raise Exception('Are you serious?')

        self.roach.generateResonatorChannels(freqList)
        self.roach.generateFftChanSelection()
        #self.roach.generateDacComb(resAttenList=attenList,globalDacAtten=9)
        print('Generating DDS Tones...')
        self.roach.generateDdsTones()
        self.roach.debug=False
        globalDacAtten = self.roach.generateDacComb(freqList=freqList, resAttenList=attenList, 
                                    iqRatioList=iqRatioList, iqPhaseOffsList=phaseList, 
                                    globalDacAtten=globalDacAtten)['dacAtten']
        self.roach.loadDacLUT()
        self.roach.fpga.write_int('run',1)#send ready signals
        time.sleep(1)
        return globalDacAtten
    
    def takeAdcSnap(self):
        """
        Takes an ADC Snap
        OUTPUTS:
            snapDict: Dictionary w/ I and Q values (see adcSnapCheck.py)
        """

        snapDict = self.roach.snapZdok()
        
        return snapDict
 
    def gridSearchOptimizerFit(self, phases=np.arange(-25, 25), iqRatios=np.arange(0.8, 1.75, 0.02), sideband='upper', 
                                threshold=38, weightDecayDist=1, nAvgs=5, saveNPZ=False):
        """
        Determines optimal phase/iq offsets for each tone in comb simultaneosly. Treats all tones as independent. For each tone,
        samples the SB suppression at a few random points in phase/iq offset search space, then fits a 2D exponential decay. Takes
        additional samples both near and far from the peak, refitting after each sample until an above-threshold point is found. Threshold
        decreases with each iteration.

        INPUTS:
            phases - list of phases in search space
            iqRatios - list of IQ ratios in search space
            threshold - starting SB suppression completion threshold

        OUTPUTS:
            self.finalPhaseList - list of optimal phases
            self.finalIQRatioList - list of optimal IQ ratios
            self.finalSBSupList - list of SB suppressions at optimal points
        """
        if sideband == 'lower':
            freqList = self.freqListLow
            attenList = self.toneAttenListLow
        elif sideband == 'upper':
            freqList = self.freqListHigh
            attenList = self.toneAttenListHigh
        else:
            raise Exception('Specify a valid sideband (either upper or lower)!')

        sampledSBSups = np.zeros((len(freqList), len(phases), len(iqRatios)))
        sampledSBSups[:] = np.nan
        
        nSamples = 4096.
        sampleRate = 2000. #MHz
        quantFreqsMHz = np.array(freqList/1.e6-self.loFreq/1.e6)
        quantFreqsMHz = np.round(quantFreqsMHz*nSamples/sampleRate)*sampleRate/nSamples
        snapDict = self.takeAdcSnap()
        specDict = adcSnap.streamSpectrum(snapDict['iVals'], snapDict['qVals'])        
        findFreq = lambda freq: np.where(specDict['freqsMHz']==freq)[0][0]
        print('quantFreqsMHz', quantFreqsMHz)
        print('spectDictFreqs', specDict['freqsMHz'])
        print('nSamples', specDict['nSamples'])
        freqLocs = np.asarray(map(findFreq, quantFreqsMHz))
        sbLocs = -1*freqLocs + len(specDict['freqsMHz'])

        #take initial spectrum
        self.loadLUT(sideband)
        initialSpectrum = np.zeros(specDict['nSamples'])
        for k in range(nAvgs):
            snapDict = self.takeAdcSnap()
            specDict = adcSnap.streamSpectrum(snapDict['iVals'], snapDict['qVals'])
            initialSpectrum += specDict['spectrumDb']
        initialSpectrum /= nAvgs

        #specifies the probability distribution from which to sample points in phase, IQ ratio, search space
        weights = np.ones((len(freqList), len(phases), len(iqRatios)))

        print('weightShape', np.shape(weights))

        #normalize weights within each frequency
        normWeights = np.transpose(np.transpose(np.reshape(weights,(len(freqList),-1)))/np.sum(weights, axis=(1,2)))
        normWeights = np.reshape(normWeights, np.shape(weights))
        
        sbSupIndList = np.zeros((len(freqList), 2), dtype=np.int) #points currently being sampled at each frequency (in I/Q ratio, phase search space
        flatInds = np.arange(len(phases)*len(iqRatios))
        curSupList = np.zeros(len(freqList)) #list of last-sampled SB suppression at each frequency, averaged over nAvgs
        curRawSupList = np.zeros((len(freqList), nAvgs)) #unaveraged list of SB suppressions
        phaseList = np.zeros(len(freqList)) #list of phase offsets being sampled at each frequency
        iqRatioList = np.ones(len(freqList)) #list of I/Q offsets being sampled at each frequency
        finalPhaseList = np.zeros(len(freqList)) #list of optimal phase offsets
        finalIQRatioList = np.ones(len(freqList)) #list of optimal I/Q amplitude offsets
        finalSBSupList = np.zeros(len(freqList)) #list of SB suppressions at optimal values
        foundMaxList = np.zeros(len(freqList)) #list of booleans indicating whether above-threshold value has been found
        counter = 0
        
        #functions used for fitting
        def gaussian(x, x0r, x0c, scale, width):
            return scale*np.exp(-((x[0]-x0r)**2+(x[1]-x0c)**2)/width**2)

        def expDecay(x, x0r, x0c, scale, width):
            return scale*np.exp(-np.sqrt((x[0]-x0r)**2+(x[1]-x0c)**2)/width)

        #sample initial points
        for i in range(7):
            for j in range(len(freqList)):
                if i==0:
                    sbSupInd = np.zeros(2, dtype=np.int)
                    sbSupInd[0] = np.where(phases<=0.01)[0][0]
                    print(iqRatios)
                    sbSupInd[1] = np.where(iqRatios-1<=0.001)[0][0]
                else:
                    flatInd = np.random.choice(flatInds, p=normWeights[j,:].flatten())
                    sbSupInd = np.unravel_index(flatInd, np.shape(sampledSBSups[j]))
                phaseList[j] = phases[sbSupInd[0]]
                iqRatioList[j] = iqRatios[sbSupInd[1]]
                sbSupIndList[j] = np.asarray(sbSupInd, dtype=np.int)
                
            self.loadLUT(sideband, phaseList, iqRatioList)

            for k in range(nAvgs):
                snapDict = self.takeAdcSnap()
                specDict = adcSnap.streamSpectrum(snapDict['iVals'], snapDict['qVals'])
                curRawSupList[:,k] = specDict['spectrumDb'][freqLocs]-specDict['spectrumDb'][sbLocs]

            curSupList = np.mean(curRawSupList, axis=1)
            
            for j in range(len(freqList)):
                print(j, sbSupIndList[j,0], sbSupIndList[j,1])
                sampledSBSups[j, sbSupIndList[j,0], sbSupIndList[j,1]] = curSupList[j]
                weights[j, sbSupIndList[j,0], sbSupIndList[j,1]] = 0
                    
            normWeights = np.transpose(np.transpose(np.reshape(weights,(len(freqList),-1)))/np.sum(weights, axis=(1,2)))
            normWeights = np.reshape(normWeights, np.shape(weights))
            


        fitParams = [10, 10, 40, 15]
        rowCoords = np.tile(np.arange(np.shape(sampledSBSups[0])[0]),(np.shape(sampledSBSups[0])[1],1))
        rowCoords = np.transpose(rowCoords)
        colCoords = np.tile(np.arange(np.shape(sampledSBSups[0])[1]),(np.shape(sampledSBSups[0])[0],1))

        while np.any(foundMaxList==0):
            nFailedFits = 0
            # choose points to sample, based on probability distribution specified by 'weights'
            for j in range(len(freqList)):
                if False: #foundMaxList[j]==1:
                    phaseList[j] = finalPhaseList[j]
                    iqRatioList[j] = finalIQRatioList[j]
                    sbSupIndList[j] = np.array([np.where(phases==finalPhaseList[j])[0][0], np.where(iqRatios==finalIQRatioList[j])[0][0]])

                else:
                    flatInd = np.random.choice(flatInds, p=normWeights[j,:].flatten())
                    sbSupInd = np.unravel_index(flatInd, np.shape(sampledSBSups[j]))
                    phaseList[j] = phases[sbSupInd[0]]
                    iqRatioList[j] = iqRatios[sbSupInd[1]]
                    sbSupIndList[j] = np.asarray(sbSupInd)

            # load in new offsets and take ADC snap 
            self.loadLUT(sideband, phaseList, iqRatioList)
            
            for k in range(nAvgs):
                snapDict = self.takeAdcSnap()
                specDict = adcSnap.streamSpectrum(snapDict['iVals'], snapDict['qVals'])
                curRawSupList[:,k] = specDict['spectrumDb'][freqLocs]-specDict['spectrumDb'][sbLocs]

            curSupList = np.mean(curRawSupList, axis=1)
            
            for j in range(len(freqList)):
                sampledSBSups[j, sbSupIndList[j,0], sbSupIndList[j,1]] = curSupList[j]
                
                # determine whether any sampled SB suppressions are above threshold
                if(np.any(sampledSBSups[j]>=threshold)):
                    foundMaxList[j]=1
                    optSBSupIndFlat = np.nanargmax(sampledSBSups[j])
                    optSBSupInd = np.unravel_index(optSBSupIndFlat, np.shape(sampledSBSups[j]))
                    finalPhaseList[j] = phases[optSBSupInd[0]]
                    finalIQRatioList[j] = iqRatios[optSBSupInd[1]]
                    finalSBSupList[j] = sampledSBSups[j, optSBSupInd[0], optSBSupInd[1]]
                
                #calculate new weights

                validSBLocs = np.where(np.isnan(sampledSBSups[j])==0) #coordinates of sampled points
                xdata = np.array([validSBLocs[0],validSBLocs[1]])
                ydata = np.array(sampledSBSups[j,validSBLocs[0],validSBLocs[1]])
                # print 'xdata', xdata
                # print 'ydata', ydata
                
                # fit exponential decay to all sampled points
                try:
                    fitParams, pcov = spo.curve_fit(expDecay, xdata, ydata, fitParams, 
                        bounds=([0, 0, 30, 2], [np.shape(sampledSBSups[0])[0], np.shape(sampledSBSups[0])[1], 50, 20]), method='trf')
                
                    # print 'fitParams', fitParams
                    # print 'fitting errors', np.sqrt(np.diag(pcov))

                    rowDist = rowCoords - fitParams[0]
                    colDist = colCoords - fitParams[1]
                    weightDecay = np.random.choice([25,weightDecayDist]) #choose whether to sample close to or far away from fit center
                    weights[j] = np.exp(-(rowDist**2+colDist**2)/weightDecay**2)

                except RuntimeError:
                    nFailedFits += 1
                    pass
                
                weights[j, validSBLocs]=0 #set weights to 0 at previously sampled points
            
            normWeights = np.transpose(np.transpose(np.reshape(weights,(len(freqList),-1)))/np.sum(weights, axis=(1,2)))
            normWeights = np.reshape(normWeights, np.shape(weights))
            
            counter += 1
            
            threshold -= 0.5

            print('Number of Failed Fits', nFailedFits)
            print('Number past threshold', sum(foundMaxList))
            print('threshold', threshold)
            print(counter, 'iterations')
            print('finalSBSupList', finalSBSupList)
            #plt.imshow(normWeights)
            #plt.colorbar()
            #plt.show()

        #take final spectrum
        self.loadLUT(sideband, finalPhaseList, finalIQRatioList)
        finalSpectrum = np.zeros(specDict['nSamples'])
        for k in range(nAvgs):
            snapDict = self.takeAdcSnap()
            specDict = adcSnap.streamSpectrum(snapDict['iVals'], snapDict['qVals'])
            finalSpectrum += specDict['spectrumDb']
        finalSpectrum /= nAvgs

        if sideband=='lower':
            self.finalPhaseListLow = finalPhaseList
            self.finalIQRatioListLow = finalIQRatioList
            self.finalSBSupListLow = finalSBSupList

        else:
            self.finalPhaseListHigh = finalPhaseList
            self.finalIQRatioListHigh = finalIQRatioList
            self.finalSBSupListHigh = finalSBSupList
            

        if saveNPZ:
            np.savez('grid_search_opt_vals_'+'r'+self.roach.ip[-3:]+'_'+str(len(freqList))+'_freqs_'+sideband+'_' \
                        +time.strftime("%Y%m%d-%H%M%S",time.localtime()), freqs=freqList, optPhases=finalPhaseList, 
                        optIQRatios=finalIQRatioList, maxSBSuppressions=finalSBSupList, toneAttenList=attenList, loFreq=self.loFreq,
                        globalDacAtten=self.globalDacAtten, adcAtten=self.adcAtten, initialSpectrum=initialSpectrum, finalSpectrum=finalSpectrum)

        
    def saveGridSearchOptFreqList(self, filename, useResID=True):
        """
        Saves new frequency list containing optimal phase and I/Q offsets.
        Compatible with HighTemplar

        INPUTS:
            filename: name (including location) of new frequency file
            useResID: should be true if resIDs were initially supplied

        """
        data = np.zeros((len(self.freqList), 5))
        data[:, 0] = self.resIDList
        data[:, 1] = self.freqList
        data[:, 2] = self.toneAttenList
        data[:, 3] = np.concatenate((self.finalPhaseListLow, self.finalPhaseListHigh))
        data[:, 4] = np.concatenate((self.finalIQRatioListLow, self.finalIQRatioListHigh))
        np.savetxt(filename, data, fmt='%4i %10.9e %4i %4i %4f')


    def ampScalePlotter(self, freq, phase, scaleRange=np.arange(0.5,1.5,0.02)):
        """
        Plot sideband suprression as a function of ADC amplitude scaling

        INPUTS:
            freq: tone frequency (Hz)
            phase: phase delay
            scaleRange: range of ADC scalings to check
        """
        self.loadLUT(np.asarray([freq]), phase)
        snapDict = self.takeAdcSnap()
        sbSuppressions = np.zeros(len(scaleRange))
        for i,scale in enumerate(scaleRange):
            iVals = scale*snapDict['iVals']
            qVals = snapDict['qVals']
            specDict = adcSnap.streamSpectrum(iVals, qVals)
            sbSuppressions[i] = specDict['peakFreqPower'] - specDict['sidebandFreqPower']

        plt.plot(scaleRange, sbSuppressions)
        plt.show()
        return sbSuppressions
    
    def checkSBSuppression(self, sideband):        
        nSamples = 4096.
        sampleRate = 2000. #MHz
        if sideband=='upper':
            freqs = self.freqListHigh
        elif sideband=='lower':
            freqs = self.freqListLow
        else:
            raise ValueError
        quantFreqsMHz = np.array(freqs/1.e6-self.loFreq/1.e6)
        quantFreqsMHz = np.round(quantFreqsMHz*nSamples/sampleRate)*sampleRate/nSamples
        snapDict = self.takeAdcSnap()
        specDict = adcSnap.streamSpectrum(snapDict['iVals'], snapDict['qVals'])        
        findFreq = lambda freq: np.where(specDict['freqsMHz']==freq)[0][0]
        print('quantFreqsMHz', quantFreqsMHz)
        print('spectDictFreqs', specDict['freqsMHz'])
        print('nSamples', specDict['nSamples'])
        freqLocs = np.asarray(map(findFreq, quantFreqsMHz))
        sbLocs = -1*freqLocs + len(specDict['freqsMHz'])
        sbSuppressions = specDict['spectrumDb'][freqLocs]-specDict['spectrumDb'][sbLocs]
        return sbSuppressions
               
def loadGridTable(fileName):
    """
    Loads exhaustive grid search data (made by makeGridPlot), finds optimal phase/iq ratio locations and loads
    them into DAC LUT. Then takes adcSnap and finds SB suppression at each frequency
    """
    data = np.load(fileName)
    sbSupFlat = np.reshape(data['sbSuppressions'],(np.shape(data['sbSuppressions'])[0],-1))
    sbMaxInds = np.argmax(sbSupFlat, axis=1)
    sbMaxLocs = np.asarray([np.unravel_index(index, (len(data['phases']), len(data['iqRatios']))) for index in sbMaxInds])
    phaseList = data['phases'][sbMaxLocs[:,0]]
    iqRatioList = data['iqRatios'][sbMaxLocs[:,1]]

    # plt.plot(data['freqs'], phaseList)
    # plt.show()
    # plt.plot(data['freqs'], iqRatioList)
    # plt.show()
     
    sbo = SBOptimizer(globalDacAtten=11, toneAtten=45, adcAtten=31.75)
    sbo.initRoach()
    # phaseList = np.zeros(len(data['freqs'])) # uncomment these lines if you don't want to load in optimal phases, just frequencies
    # iqRatioList = np.ones(len(data['freqs']))
    sbo.loadLUT(freqList=data['freqs'], phaseList=phaseList, iqRatioList=iqRatioList)

    print('phases', phaseList)
    print('iqRatios', iqRatioList)

    nSamples = 4096.
    sampleRate = 2000. #MHz
    freqs = data['freqs']
    quantFreqsMHz = np.array(freqs/1.e6-sbo.loFreq/1.e6)
    quantFreqsMHz = np.round(quantFreqsMHz*nSamples/sampleRate)*sampleRate/nSamples
    snapDict = sbo.takeAdcSnap()
    specDict = adcSnap.streamSpectrum(snapDict['iVals'], snapDict['qVals'])        
    findFreq = lambda freq: np.where(specDict['freqsMHz']==freq)[0][0]
    print('quantFreqsMHz', quantFreqsMHz)
    print('spectDictFreqs', specDict['freqsMHz'])
    print('nSamples', specDict['nSamples'])
    freqLocs = np.asarray(map(findFreq, quantFreqsMHz))
    sbLocs = -1*freqLocs + len(specDict['freqsMHz'])
    sbSuppressions = specDict['spectrumDb'][freqLocs]-specDict['spectrumDb'][sbLocs]
    print(sbSuppressions)


def optRawGridData(filename, freqInd=10, corrLen=20, threshold=40, sbSupScale=5, sbSupIncThresh=20):
    data = np.load(filename)
    sbSups = data['sbSuppressions'][freqInd]
    sampledSBSups = np.zeros(np.shape(sbSups))

    weights = np.ones(np.shape(sbSups))
    print('weightShape', np.shape(weights))
    normWeights = weights/np.sum(weights)
    flatInds = np.arange(len(weights.flatten()))
    curSup = 0
    counter = 0

    baseRowCoords = np.tile(np.arange(np.shape(sbSups)[0]),(np.shape(sbSups)[1],1))
    baseRowCoords = np.transpose(baseRowCoords)
    baseColCoords = np.tile(np.arange(np.shape(sbSups)[1]),(np.shape(sbSups)[0],1))
    while curSup<threshold:
        print(np.shape(flatInds))
        print(np.shape(normWeights.flatten()))
        print(np.sum(normWeights.flatten()))
        flatInd = np.random.choice(flatInds, p=normWeights.flatten())
        sbSupInd = np.unravel_index(flatInd, np.shape(sbSups))
        sampledSBSups[sbSupInd] = sbSups[sbSupInd]
        curSup = sbSups[sbSupInd]
        weights[sbSupInd] = 0

        #calculate addition to weights
        if(curSup > sbSupIncThresh):
            rowCoords = baseRowCoords - sbSupInd[0]
            colCoords = baseColCoords - sbSupInd[1]
            distMatrix = np.sqrt(rowCoords**2+colCoords**2) #compute the distance from each point to current point
            weightAdditions = (curSup-sbSupIncThresh)/sbSupScale*np.exp(-distMatrix**2/corrLen**2)
            weights *= weightAdditions
            normWeights = weights/np.sum(weights)

        counter += 1

        print(counter, 'iterations')
        print('sampledSBSups')
        print('curSup', curSup)
        print('position', sbSupInd)
        #plt.imshow(normWeights)
        #plt.colorbar()
        #plt.show()
    
    plt.imshow(normWeights)
    plt.colorbar()
    plt.show()

    plt.imshow(sampledSBSups)
    plt.colorbar()
    plt.show()
      

def loadOptimizedLUT(filename, ip, loadCorrections=True, plot=True):
    data = np.load(filename)
    sbo = SBOptimizer(ip=ip, freqList=data['freqs'], toneAttenList=data['toneAttenList'], globalDacAtten=data['globalDacAtten'], adcAtten=data['adcAtten'], loFreq=data['loFreq'])
    sbo.initRoach()
    print('global dac atten', sbo.globalDacAtten)
    #print 'toneAtten', sbo.toneAtten
    if loadCorrections:
        sbo.loadLUT(phaseList=data['optPhases'], iqRatioList=data['optIQRatios'])

    else:
        sbo.loadLUT()
    #sbo.loadLUT(data['freqs'])  

    nSamples = 4096.
    sampleRate = 2000. #MHz
    quantFreqsMHz = np.array(data['freqs']/1.e6-sbo.loFreq/1.e6)
    quantFreqsMHz = np.round(quantFreqsMHz*nSamples/sampleRate)*sampleRate/nSamples
    snapDict = sbo.takeAdcSnap()
    specDict = adcSnap.streamSpectrum(snapDict['iVals'], snapDict['qVals'])        
    findFreq = lambda freq: np.where(specDict['freqsMHz']==freq)[0][0]
    print('quantFreqsMHz', quantFreqsMHz)
    print('spectDictFreqs', specDict['freqsMHz'])
    print('nSamples', specDict['nSamples'])
    freqLocs = np.asarray(map(findFreq, quantFreqsMHz))
    sbLocs = -1*freqLocs + len(specDict['freqsMHz'])
    curSupList = specDict['spectrumDb'][freqLocs]-specDict['spectrumDb'][sbLocs]
    
    if plot:
        plt.plot(data['freqs'], curSupList)
        plt.plot(data['freqs'], data['maxSBSuppressions'])
        plt.show()

    return sbo
        
def optRawGridDataFit(filename, freqInd=40, threshold=32, weightDecayDist=1):
    data = np.load(filename)
    sbSups = data['sbSuppressions'][freqInd] #exhaustive grid search array of SB suppressions
    sampledSBSups = np.zeros(np.shape(sbSups))
    sampledSBSups[:] = np.nan

    weights = np.ones(np.shape(sbSups))
    print('weightShape', np.shape(weights))
    normWeights = weights/np.sum(weights)
    flatInds = np.arange(len(weights.flatten()))
    curSup = 0
    counter = 0

    def gaussian(x, x0r, x0c, scale, width):
        return scale*np.exp(-((x[0]-x0r)**2+(x[1]-x0c)**2)/width**2)

    def expDecay(x, x0r, x0c, scale, width):
        return scale*np.exp(-np.sqrt((x[0]-x0r)**2+(x[1]-x0c)**2)/width)

    #sample initial points
    for i in range(5):
        flatInd = np.random.choice(flatInds, p=normWeights.flatten())
        sbSupInd = np.unravel_index(flatInd, np.shape(sbSups))
        sampledSBSups[sbSupInd] = sbSups[sbSupInd]
        curSup = sbSups[sbSupInd]
        weights[sbSupInd] = 0
        normWeights = weights/np.sum(weights)

    fitParams = [10, 10, 40, 15]
    rowCoords = np.tile(np.arange(np.shape(sbSups)[0]),(np.shape(sbSups)[1],1))
    rowCoords = np.transpose(rowCoords)
    colCoords = np.tile(np.arange(np.shape(sbSups)[1]),(np.shape(sbSups)[0],1))

    while curSup<threshold:
        flatInd = np.random.choice(flatInds, p=normWeights.flatten())
        sbSupInd = np.unravel_index(flatInd, np.shape(sbSups))
        sampledSBSups[sbSupInd] = sbSups[sbSupInd]
        curSup = sbSups[sbSupInd]
        weights[sbSupInd] = 0

        #calculate new weights

        validSBLocs = np.where(np.isnan(sampledSBSups)==0)
        xdata = np.array([validSBLocs[0],validSBLocs[1]])
        ydata = np.array(sampledSBSups[validSBLocs])
        #print 'xdata', xdata
        #print 'ydata', ydata

        fitParams, pcov = spo.curve_fit(expDecay, xdata, ydata, fitParams, 
            bounds=([0, 0, 30, 2], [np.shape(sampledSBSups)[0], np.shape(sampledSBSups)[1], 50, 20]), method='trf')

        print('fitParams', fitParams)

        rowDist = rowCoords - fitParams[0]
        colDist = colCoords - fitParams[1]
        weights = np.exp(-(rowDist**2+colDist**2)/weightDecayDist**2)
        
        weights[validSBLocs]=0

        normWeights = weights/np.sum(weights)
        
        counter += 1

        print(counter, 'iterations')
        print('sampledSBSups')
        print('curSup', curSup)
        print('position', sbSupInd)
        #plt.imshow(normWeights)
        #plt.colorbar()
        #plt.show()
    
    plt.imshow(normWeights)
    plt.colorbar()
    plt.show()

    plt.imshow(sampledSBSups)
    plt.colorbar()
    plt.show()

class sbOptThread(threading.Thread):
    """
    Multithreads optimization code to make it easy to run with multiple boards simultaneously
    """
    def __init__(self, **kwargs):
        threading.Thread.__init__(self)
        self.ip = kwargs.pop('ip', '10.0.0.112')
        self.adcAtten = kwargs.pop('adcAtten', 31.75)
        self.globalDacAtten = kwargs.pop('globalDacAtten', 12)
        self.loFreq = kwargs.pop('loFreq')
        self.freqFileName = kwargs.pop('freqFile')
        self.mdd = os.environ['MKID_DATA_DIR']

    def run(self):
        print('Starting optimization for ROACH ' + self.ip)
        freqFile = os.path.join(self.mdd, self.freqFileName)
        resIDList, freqList, attenList = np.loadtxt(freqFile, unpack=True)
        sbo = SBOptimizer(ip=self.ip, freqList=freqList, toneAttenList=attenList, resIDList=resIDList, adcAtten=self.adcAtten, globalDacAtten=self.globalDacAtten, loFreq=self.loFreq)
        sbo.initRoach()
        sbo.gridSearchOptimizerFit(sideband='upper', saveNPZ=True)
        sbo.gridSearchOptimizerFit(sideband='lower', saveNPZ=True)
        sbo.saveGridSearchOptFreqList(freqFile.split('.')[0] + '_sbOpt_v3.txt')


if __name__=='__main__': 
    parser = argparse.ArgumentParser(description='Optimize frequency and power specific sideband suppression')
    parser.add_argument('roaches', nargs='+', help='Roach numbers')
    parser.add_argument('-c', '--config', default=mkidreadout.config.DEFAULT_ROACH_CFGFILE, help='roach.yml config file')
    args = parser.parse_args()

    for roachNum in args.roaches:
        ip = templarConf.get('Roach '+str(roachNum), 'ipaddress')
        globalDacAtten = templarConf.getfloat('Roach '+str(roachNum), 'dacatten_start')
        adcAtten = templarConf.getfloat('Roach '+str(roachNum), 'adcatten')
        loFreq = templarConf.getfloat('Roach '+str(roachNum), 'lo_freq')
        freqFile = templarConf.get('Roach '+str(roachNum), 'freqfile')
        sbThread = sbOptThread(ip=ip, globalDacAtten=globalDacAtten, adcAtten=adcAtten, loFreq=loFreq, freqFile=freqFile)
        threadpool.append(sbThread)

    for thread in threadpool:
        thread.start()


    # ip = '10.0.0.122'
    # freqFileName = 'ps_r122_FL4_b_faceless_hf_train_rm_doubles.txt'
    # 
    # freqFile = os.path.join(mdd, 'truncatedAttens', freqFileName)
    # resIDList, freqList, attenList = np.loadtxt(freqFile, unpack=True)
    # 
    # adcAtten=31.75
    # globalDacAtten=9
    # loFreq=7.2416148e9
    # sbo = SBOptimizer(ip=ip, freqList=freqList, toneAttenList=attenList, resIDList=resIDList, adcAtten=adcAtten, globalDacAtten=globalDacAtten, loFreq=loFreq)
    # sbo.initRoach()
    # sbo.gridSearchOptimizerFit(sideband='upper', saveNPZ=True)
    # sbo.gridSearchOptimizerFit(sideband='lower', saveNPZ=True)
    # sbo.saveGridSearchOptFreqList(freqFile.split('.')[0] + '_sbOpt.txt')
            
