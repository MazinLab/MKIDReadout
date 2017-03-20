'''
Code to minimize sideband amplitude in the ADC by applying offsets to the 
relative phases and amplitudes of the I and Q signals for each tone.

Author: Neelay Fruitwala

'''

import adcSnapCheck as adcSnap
from Roach2Controls import Roach2Controls
import numpy as np
import time, struct, sys, os
import matplotlib.pyplot as plt
import random
import scipy.optimize as spo

class SBOptimizer:
    def __init__(self, ip='10.0.0.112', params='/mnt/data0/neelay/MkidDigitalReadout/DataReadout/ChannelizerControls/DarknessFpga_V2.param', freqList=None,
        toneAttenList=None, resIDList=None, loFreq=5.e9, adcAtten=31.75, globalDacAtten=9):
        '''
        INPUTS:
            ip: ROACH2 IP Address
            params: Roach2Controls parameter file
            loFreq: LO Frequency in Hz
            adcAtten: attenuation value (dB) of IF board ADC attenuator
            toneAtten: total resonator attenuation
            global Dac Atten: physical DAC attenuation
            frequencty: tone frequency (optional)
        '''
        if freqList is None or toneAttenList is None:
            raise Exception('Must specify frequencies and attenuations')

        if len(freqList)!=len(toneAttenList):
            raise ValueError('Frequency and attenuation lists must be the same length!')

        self.roach = Roach2Controls(ip, params, True, False)
        self.loFreq = loFreq
        self.adcAtten = adcAtten
        self.toneAttenList = toneAttenList
        self.globalDacAtten = globalDacAtten
        self.freqList = freqList
        self.resIDList = resIDList

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

    def initRoach(self):
        '''
        Initializes the ROACH2: connects, initializes the UART, loads the LO and physical attenuations
        '''
        self.roach.connect()
        self.roach.setLOFreq(self.loFreq)
        self.roach.initializeV7UART(waitForV7Ready=False)
        self.roach.loadLOFreq()
        self.roach.changeAtten(1,np.floor(self.globalDacAtten/31.75)*31.75)
        self.roach.changeAtten(2,self.globalDacAtten%31.75)
        self.roach.changeAtten(3,self.adcAtten)

    def loadLUT(self, sideband='all', phaseList=None, iqRatioList=None):
        '''
        Loads DAC LUT
        INPUTS:
            freq: tone frequency in Hz
            phaseDelay: IQ phase offset (degrees)
            iqRatio: IAmp/QAmp
        '''
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
        print 'Generating DDS Tones...'
        self.roach.generateDdsTones()
        self.roach.debug=False
        self.roach.generateDacComb(freqList=freqList, resAttenList=attenList, globalDacAtten=self.globalDacAtten, iqRatioList=iqRatioList, 
            iqPhaseOffsList=phaseList)
        self.roach.loadDacLUT()
        self.roach.fpga.write_int('run',1)#send ready signals
        time.sleep(1)
    
    def takeAdcSnap(self):
        '''
        Takes an ADC Snap
        OUTPUTS:
            snapDict: Dictionary w/ I and Q values (see adcSnapCheck.py)
        '''
        delayLut0 = zip(np.arange(0,12),np.ones(12)*14)
        delayLut1 = zip(np.arange(14,26),np.ones(12)*18)
        delayLut2 = zip(np.arange(28,40),np.ones(12)*14)
        delayLut3 = zip(np.arange(42,54),np.ones(12)*13)
        adcSnap.loadDelayCal(self.roach.fpga,delayLut0)
        adcSnap.loadDelayCal(self.roach.fpga,delayLut1)
        adcSnap.loadDelayCal(self.roach.fpga,delayLut2)
        adcSnap.loadDelayCal(self.roach.fpga,delayLut3)

        snapDict = adcSnap.snapZdok(self.roach.fpga,nRolls=0)
        
        return snapDict
        
    def gridSearchOptimizerFit(self, phases=np.arange(-25, 10), iqRatios=np.arange(0.65, 1.35, 0.02), sideband='upper', threshold=45, weightDecayDist=1, saveNPZ=False):
        '''
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
        '''
        if sideband == 'lower':
            freqList = self.freqListLow
        elif sideband == 'upper':
            freqList = self.freqListHigh
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
        print 'quantFreqsMHz', quantFreqsMHz
        print 'spectDictFreqs', specDict['freqsMHz']
        print 'nSamples', specDict['nSamples']
        freqLocs = np.asarray(map(findFreq, quantFreqsMHz))
        sbLocs = -1*freqLocs + len(specDict['freqsMHz'])

        weights = np.ones((len(freqList), len(phases), len(iqRatios)))
        print 'weightShape', np.shape(weights)
        normWeights = np.transpose(np.transpose(np.reshape(weights,(len(freqList),-1)))/np.sum(weights, axis=(1,2)))
        normWeights = np.reshape(normWeights, np.shape(weights))
        
        sbSupIndList = np.zeros((len(freqList), 2))
        flatInds = np.arange(len(phases)*len(iqRatios))
        curSupList = np.zeros(len(freqList))
        phaseList = np.zeros(len(freqList))
        iqRatioList = np.ones(len(freqList))
        finalPhaseList = np.zeros(len(freqList))
        finalIQRatioList = np.ones(len(freqList))
        finalSBSupList = np.zeros(len(freqList))
        foundMaxList = np.zeros(len(freqList))
        counter = 0

        def gaussian(x, x0r, x0c, scale, width):
            return scale*np.exp(-((x[0]-x0r)**2+(x[1]-x0c)**2)/width**2)

        def expDecay(x, x0r, x0c, scale, width):
            return scale*np.exp(-np.sqrt((x[0]-x0r)**2+(x[1]-x0c)**2)/width)

        #sample initial points
        for i in range(7):
            for j in range(len(freqList)):
                flatInd = np.random.choice(flatInds, p=normWeights[j,:].flatten())
                sbSupInd = np.unravel_index(flatInd, np.shape(sampledSBSups[j]))
                phaseList[j] = phases[sbSupInd[0]]
                iqRatioList[j] = iqRatios[sbSupInd[1]]
                sbSupIndList[j] = np.asarray(sbSupInd)
                
            self.loadLUT(sideband, phaseList, iqRatioList)
            snapDict = self.takeAdcSnap()
            specDict = adcSnap.streamSpectrum(snapDict['iVals'], snapDict['qVals'])
            curSupList = specDict['spectrumDb'][freqLocs]-specDict['spectrumDb'][sbLocs]
            
            for j in range(len(freqList)):
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
            for j in range(len(freqList)):
                flatInd = np.random.choice(flatInds, p=normWeights[j,:].flatten())
                sbSupInd = np.unravel_index(flatInd, np.shape(sampledSBSups[j]))
                phaseList[j] = phases[sbSupInd[0]]
                iqRatioList[j] = iqRatios[sbSupInd[1]]
                sbSupIndList[j] = np.asarray(sbSupInd)

                
            self.loadLUT(sideband, phaseList, iqRatioList)
            snapDict = self.takeAdcSnap()
            specDict = adcSnap.streamSpectrum(snapDict['iVals'], snapDict['qVals'])
            curSupList = specDict['spectrumDb'][freqLocs]-specDict['spectrumDb'][sbLocs]
            
            for j in range(len(freqList)):
                sampledSBSups[j, sbSupIndList[j,0], sbSupIndList[j,1]] = curSupList[j]
                
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
                
                try:
                    fitParams, pcov = spo.curve_fit(expDecay, xdata, ydata, fitParams, 
                        bounds=([0, 0, 30, 2], [np.shape(sampledSBSups[0])[0], np.shape(sampledSBSups[0])[1], 50, 20]), method='trf')
                
                    # print 'fitParams', fitParams
                    # print 'fitting errors', np.sqrt(np.diag(pcov))

                    rowDist = rowCoords - fitParams[0]
                    colDist = colCoords - fitParams[1]
                    weightDecay = np.random.choice([25,weightDecayDist])
                    weights[j] = np.exp(-(rowDist**2+colDist**2)/weightDecay**2)

                except RuntimeError:
                    nFailedFits += 1
                    pass
                
                weights[j, validSBLocs]=0
            
            normWeights = np.transpose(np.transpose(np.reshape(weights,(len(freqList),-1)))/np.sum(weights, axis=(1,2)))
            normWeights = np.reshape(normWeights, np.shape(weights))
            
            counter += 1
            
            threshold -= 2

            print 'Number of Failed Fits', nFailedFits
            print 'Number past threshold', sum(foundMaxList)
            print 'threshold', threshold
            print counter, 'iterations'
            print 'finalSBSupList', finalSBSupList
            #plt.imshow(normWeights)
            #plt.colorbar()
            #plt.show()

        if sideband=='lower':
            self.finalPhaseListLow = finalPhaseList
            self.finalIQRatioListLow = finalIQRatioList
            self.finalSBSupListLow = finalSBSupList

        else:
            self.finalPhaseListHigh = finalPhaseList
            self.finalIQRatioListHigh = finalIQRatioList
            self.finalSBSupListHigh = finalSBSupList
            

        if saveNPZ:
            np.savez('grid_search_opt_vals_'+str(len(freqList))+'_freqs_'+time.strftime("%Y%m%d-%H%M%S",time.localtime()), freqs=freqList,
                            optPhases=finalPhaseList, optIQRatios=finalIQRatioList, maxSBSuppressions=finalSBSupList, toneAttenList=self.toneAttenList,
                                            globalDacAtten=self.globalDacAtten, adcAtten=self.adcAtten)

        
    def saveGridSearchOptFreqList(self, filename, useResID=True):
        if useResID:
           data = np.zeros((len(self.freqList), 5))
           data[:, 0] = self.resIDList
           data[:, 1] = self.freqList
           data[:, 2] = self.toneAttenList
           data[:, 3] = np.concatenate((self.finalPhaseListLow, self.finalPhaseListHigh))
           data[:, 4] = np.concatenate((self.finalIQRatioListLow, self.finalIQRatioListHigh))
           np.savetxt(filename, data)

        else:
           data = np.zeros((len(self.freqList), 4))
           data[:, 0] = self.freqList
           data[:, 1] = self.toneAttenList
           data[:, 2] = np.concatenate((self.finalPhaseListLow, self.finalPhaseListHigh))
           data[:, 3] = np.concatenate((self.finalIQRatioListLow, self.finalIQRatioListHigh))
           np.savetxt(filename, data)


    def ampScalePlotter(self, freq, phase, scaleRange=np.arange(0.5,1.5,0.02)):
        '''
        Plot sideband suprression as a function of ADC amplitude scaling
        
        INPUTS:
            freq: tone frequency (Hz)
            phase: phase delay
            scaleRange: range of ADC scalings to check
        '''
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

    def setGlobalDacAtten(self, globalDacAtten):
        self.globalDacAtten=globalDacAtten 
        self.roach.changeAtten(1,np.floor(self.globalDacAtten/31.75)*31.75)
        self.roach.changeAtten(2,self.globalDacAtten%31.75)
               
def loadGridTable(fileName):
    '''
    Loads exhaustive grid search data (made by makeGridPlot), finds optimal phase/iq ratio locations and loads
    them into DAC LUT. Then takes adcSnap and finds SB suppression at each frequency
    '''
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
    
    print 'phases', phaseList
    print 'iqRatios', iqRatioList

    nSamples = 4096.
    sampleRate = 2000. #MHz
    freqs = data['freqs']
    quantFreqsMHz = np.array(freqs/1.e6-sbo.loFreq/1.e6)
    quantFreqsMHz = np.round(quantFreqsMHz*nSamples/sampleRate)*sampleRate/nSamples
    snapDict = sbo.takeAdcSnap()
    specDict = adcSnap.streamSpectrum(snapDict['iVals'], snapDict['qVals'])        
    findFreq = lambda freq: np.where(specDict['freqsMHz']==freq)[0][0]
    print 'quantFreqsMHz', quantFreqsMHz
    print 'spectDictFreqs', specDict['freqsMHz']
    print 'nSamples', specDict['nSamples']
    freqLocs = np.asarray(map(findFreq, quantFreqsMHz))
    sbLocs = -1*freqLocs + len(specDict['freqsMHz'])
    sbSuppressions = specDict['spectrumDb'][freqLocs]-specDict['spectrumDb'][sbLocs]
    print sbSuppressions

def optRawGridData(filename, freqInd=10, corrLen=20, threshold=40, sbSupScale=5, sbSupIncThresh=20):
    data = np.load(filename)
    sbSups = data['sbSuppressions'][freqInd]
    sampledSBSups = np.zeros(np.shape(sbSups))

    weights = np.ones(np.shape(sbSups))
    print 'weightShape', np.shape(weights)
    normWeights = weights/np.sum(weights)
    flatInds = np.arange(len(weights.flatten()))
    curSup = 0
    counter = 0

    baseRowCoords = np.tile(np.arange(np.shape(sbSups)[0]),(np.shape(sbSups)[1],1))
    baseRowCoords = np.transpose(baseRowCoords)
    baseColCoords = np.tile(np.arange(np.shape(sbSups)[1]),(np.shape(sbSups)[0],1))
    while curSup<threshold:
        print np.shape(flatInds)
        print np.shape(normWeights.flatten())
        print np.sum(normWeights.flatten())
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

        print counter, 'iterations'
        print 'sampledSBSups'
        print 'curSup', curSup
        print 'position', sbSupInd
        #plt.imshow(normWeights)
        #plt.colorbar()
        #plt.show()
    
    plt.imshow(normWeights)
    plt.colorbar()
    plt.show()

    plt.imshow(sampledSBSups)
    plt.colorbar()
    plt.show()
      

def loadOptimizedLUT(filename):
    data = np.load(filename)
    sbo = SBOptimizer(ip='10.0.0.112', toneAtten=data['toneAtten'], globalDacAtten=data['globalDacAtten'], adcAtten=data['adcAtten'])
    sbo.initRoach()
    print 'global dac atten', sbo.globalDacAtten
    print 'toneAtten', sbo.toneAtten
    sbo.loadLUT(data['freqs'], phaseList=data['optPhases'], iqRatioList=data['optIQRatios'])
    #sbo.loadLUT(data['freqs'])  

    nSamples = 4096.
    sampleRate = 2000. #MHz
    quantFreqsMHz = np.array(data['freqs']/1.e6-sbo.loFreq/1.e6)
    quantFreqsMHz = np.round(quantFreqsMHz*nSamples/sampleRate)*sampleRate/nSamples
    snapDict = sbo.takeAdcSnap()
    specDict = adcSnap.streamSpectrum(snapDict['iVals'], snapDict['qVals'])        
    findFreq = lambda freq: np.where(specDict['freqsMHz']==freq)[0][0]
    print 'quantFreqsMHz', quantFreqsMHz
    print 'spectDictFreqs', specDict['freqsMHz']
    print 'nSamples', specDict['nSamples']
    freqLocs = np.asarray(map(findFreq, quantFreqsMHz))
    sbLocs = -1*freqLocs + len(specDict['freqsMHz'])
    curSupList = specDict['spectrumDb'][freqLocs]-specDict['spectrumDb'][sbLocs]

    plt.plot(data['freqs'], curSupList)
    plt.plot(data['freqs'], data['maxSBSuppressions'])
    plt.show()
        
def optRawGridDataFit(filename, freqInd=40, threshold=32, weightDecayDist=1):
    data = np.load(filename)
    sbSups = data['sbSuppressions'][freqInd] #exhaustive grid search array of SB suppressions
    sampledSBSups = np.zeros(np.shape(sbSups))
    sampledSBSups[:] = np.nan

    weights = np.ones(np.shape(sbSups))
    print 'weightShape', np.shape(weights)
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
        
        print 'fitParams', fitParams

        rowDist = rowCoords - fitParams[0]
        colDist = colCoords - fitParams[1]
        weights = np.exp(-(rowDist**2+colDist**2)/weightDecayDist**2)
        
        weights[validSBLocs]=0

        normWeights = weights/np.sum(weights)
        
        counter += 1

        print counter, 'iterations'
        print 'sampledSBSups'
        print 'curSup', curSup
        print 'position', sbSupInd
        #plt.imshow(normWeights)
        #plt.colorbar()
        #plt.show()
    
    plt.imshow(normWeights)
    plt.colorbar()
    plt.show()

    plt.imshow(sampledSBSups)
    plt.colorbar()
    plt.show()


if __name__=='__main__':
    if len(sys.argv)<3:
        raise Exception('Must specify IP address and frequency file in MKID_DATA_DIR')
    ip = '10.0.0.' + str(sys.argv[1])
    mdd = os.environ['MKID_DATA_DIR']
    freqFile = os.path.join(mdd, sys.argv[2])
    resIDList, freqList, attenList = np.loadtxt(freqFile, unpack=True)

    sbo = SBOptimizer(ip=ip, freqList=freqList, toneAttenList=attenList, resIDList=resIDList, globalDacAtten=6, loFreq=6.7354026e9)
    sbo.initRoach()
    sbo.gridSearchOptimizerFit(sideband='upper', saveNPZ=True)
    sbo.gridSearchOptimizerFit(sideband='lower', saveNPZ=True)
    sbo.saveGridSearchOptFreqList(freqFile.split('.')[0] + '_sbOpt.txt')
            