import numpy as np
import matplotlib.pyplot as plt
import os, pickle
import mkidreadout.configuration.sweepdata as sd
from wpsnn import WPSNeuralNet
from mkidcore.readdict import ReadDict
import mkidcore.instruments as inst
import findResonatorsWPS as finder
from compareResLists import retrieveManResList, retrieveMLResList
from checkWPSPerformance import matchResonators, matchAttens

def trainAndCrossValidate(mlDict, rootDir, trainFileLists, trainLabelsLists, valFileLists, valLabelsLists, satLabelsLists=None):
    modelName = mlDict['modelName']
    trainNPZ = mlDict['trainNPZ'].split('.')[0]
    valThingList = []
    if not len(trainFileLists) ==  len(trainLabelsLists) == len(valFileLists) == len(valLabelsLists):
        raise Exception('Need to provide same number of train, val file/label sets')
    if satLabelsLists is not None and len(satLabelsLists) != len(valLabelsLists):
        raise Exception('Need to provide same number of val/sat label sets')
    for i in range(len(trainFileLists)):
        mlDict.update({'modelName': modelName + '_' + str(i)})
        mlDict.update({'modelDir': os.path.join(rootDir, modelName + '_' + str(i))})
        mlDict.update({'trainNPZ': trainNPZ + '_' + str(i) + '.npz'})
        mlDict.update({'rawTrainFiles': trainFileLists[i]})
        mlDict.update({'rawTrainLabels': trainLabelsLists[i]})
        if not os.path.isfile(os.path.join(mlDict['modelDir'], mlDict['modelName'] + '.meta')):
            mlClass = WPSNeuralNet(mlDict)
            mlClass.initializeAndTrainModel()
        else:
            print 'found model, skipping train step.'

        if satLabelsLists is None:
            valThing = validateModel(mlDict['modelDir'], valFileLists[i], valLabelsLists[i])
        else:
            valThing = validateModel(mlDict['modelDir'], valFileLists[i], valLabelsLists[i], satLabelsLists[i])
        valThing.savePlots(mlDict['modelDir'], 'fullValidation')
        pickle.dump(valThing, open(os.path.join(mlDict['modelDir'], 'fullValidation.p'), 'w'))
        valThingList.append(valThing)

    fullValThing = valThingList[0]
    for valThing in valThingList[1:]:
        fullValThing += valThing
    fullValThing.savePlots(rootDir, modelName + '_full_val')
    pickle.dump(fullValThing, open(os.path.join(rootDir, modelName + '_full_val.p'), 'w'))



def validateModel(modelDir, valFiles, valMDFiles, satMDFiles=None, saveWPSMap=False):
    """
    Validates a single model (training run) w/ data given by
    valFiles and valMDFiles
    """
    mdFiles = []
    for valFile in valFiles:
        mdFile = os.path.join(modelDir, os.path.basename(valFile).split('.')[0] + '_metadata.txt')
        mdFiles.append(mdFile)
        if os.path.isfile(mdFile):
            continue

        sweep = sd.FreqSweep(valFile)
        wpsmapFile = os.path.join(modelDir, os.path.basename(valFile).split('.')[0] + '_wpsmap.npz')
        if os.path.isfile(wpsmapFile):
            wsf = np.load(wpsmapFile)
            wpsmap = wsf['wpsmap']
            freqs = wsf['freqs']
            attens = wsf['attens']
        else:
            wpsmap, freqs, attens = finder.makeWPSMap(modelDir, sweep)
            if saveWPSMap:
                np.savez(wpsmapFile, wpsmap=wpsmap, freqs=freqs, attens=attens)

        resFreqs, resAttens, scores = finder.findResonators(wpsmap, freqs, attens, 
                peakThresh=0.90, nRes=1024)

        if resFreqs[0] < 4.7e9:
            band = 'a'
        else:
            band = 'b'
        try:
            fl = inst.guessFeedline(os.path.basename(mdFile))
        except ValueError:
            fl = 1
        finder.saveMetadata(mdFile, resFreqs, resAttens, scores, fl, band)
    
    valThingList = []

    for i in range(len(mdFiles)): 
        valThing = getValThing(valMDFiles[i], mdFiles[i])
        if satMDFiles is not None:
            satValThing = getValThing(satMDFiles[i], mdFiles[i])
            manSatValThing = getValThing(satMDFiles[i], valMDFiles[i], type=('man', 'man'))
            valThing.addSatAttenDiffs(satValThing.attenDiffs)
            valThing.addManSatAttenDiffs(manSatValThing.attenDiffs)
        valThing.savePlots(modelDir, os.path.basename(valFiles[i]).split('.')[0])
        fileName = os.path.join(modelDir, os.path.basename(valFiles[i]).split('.')[0] + '_validation.p')
        print 'saving', fileName
        pickle.dump(valThing, open(fileName, 'w'))
        valThingList.append(valThing)


    fullValThing = valThingList[0]
    for valThing in valThingList[1:]:
        fullValThing += valThing

    return fullValThing

def getValThing(manMDFile, mlMDFile, type=('man', 'ml')):
    manMD = sd.SweepMetadata(file=manMDFile)
    mlMD = sd.SweepMetadata(file=mlMDFile)

    if type[1]=='ml':
        mlResID, mlFreq, mlAtten, mlGoodMask= retrieveMLResList(mlMD)
    elif type[1]=='man':
        mlResID, mlFreq, mlAtten, mlGoodMask= retrieveManResList(mlMD)
    else:
        raise Exception('type must be man or ml')

    if type[0]=='man':
        manResID, manFreq, manAtten, manGoodMask= retrieveManResList(manMD)
    elif type[0]=='ml':
        manResID, manFreq, manAtten, manGoodMask= retrieveMLResList(manMD)
    else:
        raise Exception('type must be man or ml')

    manResID = manResID[manGoodMask]
    manFreq = manFreq[manGoodMask]
    manAtten = manAtten[manGoodMask]
    mlResID = mlResID[mlGoodMask]
    mlFreq = mlFreq[mlGoodMask]
    mlAtten = mlAtten[mlGoodMask]

    manSortedInd = np.argsort(manFreq)
    manResID = manResID[manSortedInd]
    manFreq = manFreq[manSortedInd]
    manAtten = manAtten[manSortedInd]

    mlSortedInd = np.argsort(mlFreq)
    mlResID = mlResID[mlSortedInd]
    mlFreq = mlFreq[mlSortedInd]
    mlAtten = mlAtten[mlSortedInd]
    
    mantoml = matchResonators(manResID, mlResID, manFreq, mlFreq, 200.e3)
    mlNotInMan = np.empty((0, 2))
    for j, resID in enumerate(mlResID):
        if not np.any(j == mantoml[:, 0]):
            mlNotInMan = np.vstack((mlNotInMan, np.array([resID, 0])))
    
    nMatched = np.sum(~np.isnan(mantoml[:, 0]))
    nManNotInML = np.sum(np.isnan(mantoml[:, 0]))
    nMLNotInMan = len(mlNotInMan)
    
    manAttenMatched, mlAttenMatched = matchAttens(manAtten, mlAtten, mantoml)
    manFreqMatched, mlFreqMatched = matchAttens(manFreq, mlFreq, mantoml)

    attenDiff = mlAttenMatched - manAttenMatched
    freqDiff = mlFreqMatched - manFreqMatched

    attenStart = 40
    manAttenMatched -= attenStart
    mlAttenMatched -= attenStart
    manAttenMatched = np.round(manAttenMatched).astype(int)
    mlAttenMatched = np.round(mlAttenMatched).astype(int)
    confImage = np.zeros((40, 40))
    for j in range(len(manAttenMatched)):
        confImage[manAttenMatched[j], mlAttenMatched[j]] += 1

    valThing = ValidationThing(nMatched, nManNotInML, nMLNotInMan, confImage, attenDiff,
            freqDiff, None, manMDFile)

    return valThing

class ValidationThing(object):

    def __init__(self, nMatched, nManNotInML, nMLNotInMan, confImage, attenDiffs, freqDiffs, sweepFiles=None, 
            mdFiles=None, satAttenDiffs=None, manSatAttenDiffs=None):
        self.nMatched = nMatched
        self.nManNotInML = nManNotInML
        self.nMLNotInMan = nMLNotInMan
        self.confImage = confImage
        self.attenDiffs = attenDiffs
        self.freqDiffs = freqDiffs
        self.satAttenDiffs = satAttenDiffs
        self.manSatAttenDiffs = manSatAttenDiffs
        
        if sweepFiles is not None:
            self.sweepFiles = list(np.atleast_1d(sweepFiles))
        else:
            self.sweepFiles = None
        if mdFiles is not None:
            self.mdFiles = list(np.atleast_1d(mdFiles))
        else:
            self.mdFiles = None

    def __add__(self, valThing):
        if valThing.sweepFiles is None or self.sweepFiles is None:
            sweepFiles = None
        else:
            sweepFiles = self.sweepFiles + valThing.sweepFiles
        if valThing.mdFiles is None or self.mdFiles is None:
            mdFiles = None
        else:
            mdFiles = self.mdFiles + valThing.mdFiles


        nMatched = valThing.nMatched + self.nMatched
        nManNotInML = valThing.nManNotInML + self.nManNotInML
        nMLNotInMan = valThing.nMLNotInMan + self.nMLNotInMan
        confImage = valThing.confImage + self.confImage
        attenDiffs = np.append(self.attenDiffs, valThing.attenDiffs)
        freqDiffs = np.append(self.freqDiffs, valThing.freqDiffs)

        if self.satAttenDiffs is not None and valThing.satAttenDiffs is not None:
            satAttenDiffs = np.append(self.satAttenDiffs, valThing.satAttenDiffs)
        else:
            satAttenDiffs = None

        if self.manSatAttenDiffs is not None and valThing.manSatAttenDiffs is not None:
            manSatAttenDiffs = np.append(self.manSatAttenDiffs, valThing.manSatAttenDiffs)
        else:
            manSatAttenDiffs = None

        return ValidationThing(nMatched, nManNotInML, nMLNotInMan, confImage, attenDiffs, 
                freqDiffs, sweepFiles, mdFiles, satAttenDiffs, manSatAttenDiffs)

    def savePlots(self, directory, prefix=None):
        if prefix is None:
            try:
                prefix = '_'.join(os.path.basename(self.mdFiles).split('.')[0])
            except TypeError:
                prefix = 'YOU_SUCK'
        
        fig = plt.figure()
        ax00 = fig.add_subplot(221)
        ax01 = fig.add_subplot(222)
        ax10 = fig.add_subplot(223)
        ax11 = fig.add_subplot(224)

        im = ax01.imshow(self.confImage, vmax=55)
        #fig.colorbar(im, cax=ax01)
        ax01.set_title('Confusion', fontsize=7)
        ax01.set_ylabel('Manual', fontsize=7)
        ax01.set_xlabel('ML', fontsize=7)
        ax01.tick_params(axis='both', labelsize=5)

        #check attenDiff increment
        attenDiffList = np.unique(self.attenDiffs)
        if np.min(np.abs(np.diff(attenDiffList))) < 0.9:
            attenHistBins = 20
            attenHistRange = (-4.75, 5.25)
        else:
            attenHistBins = 10
            attenHistRange = (-4.5, 5.5)

        print attenDiffList
        print 'bins', attenHistBins

        if self.satAttenDiffs is None:
            ax00.hist(self.attenDiffs, bins=attenHistBins, range=attenHistRange)
            ax00.set_xlabel('ML Atten - Manual Atten', fontsize=7)
        else:
            ax00.hist(self.attenDiffs, bins=attenHistBins, range=attenHistRange, label='ML - Manual', alpha=0.3)
            ax00.hist(self.satAttenDiffs, bins=attenHistBins, range=attenHistRange, label='ML - Saturation', alpha=0.3) 
            if self.manSatAttenDiffs is not None:
                ax00.hist(self.manSatAttenDiffs, bins=attenHistBins, range=attenHistRange, label='Manual - Saturation', alpha=0.3) 
            ax00.legend(fontsize=5)
            ax00.set_xlabel('Atten Difference (dB)', fontsize=7)
        ax00.tick_params(axis='both', labelsize=5)
        ax10.hist(self.freqDiffs, bins=20, range=(-100.e3, 100.e3))
        ax10.tick_params(axis='both', labelsize=5)
        ax10.set_xlabel('ML Freq - Manual Freq (Hz)', fontsize=7)
        
        #resTable = [['Matched between manual and ML', str(self.nMatched)],
        #            ['Manual res not found in ML', str(self.nManNotInML)],
        #            ['ML res not found in manual', str(self.nMLNotInMan)]]
        resTable = [[ str(self.nMatched)],
                    [ str(self.nManNotInML)],
                    [ str(self.nMLNotInMan)]]
        labels = ['Matched between \nmanual and ML',
                    'Manual res not \nfound in ML',
                    'ML res not found \nin manual']
        ax11.axis('tight')
        ax11.axis('off')
        tab = ax11.table(cellText=resTable, rowLabels=labels, loc='center right', colWidths=[.2])
        tab.scale(1, 2)
        print 'saving', os.path.join(directory, prefix + '_summary.pdf') 
        fig.savefig(os.path.join(directory, prefix + '_summary.pdf'))

    def addSatAttenDiffs(self, satAttenDiffs):
        self.satAttenDiffs = satAttenDiffs

    def addManSatAttenDiffs(self, satAttenDiffs):
        self.manSatAttenDiffs = satAttenDiffs


