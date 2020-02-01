import numpy as np
import matplotlib.pyplot as plt
import os, pickle
import mkidreadout.configuration.sweepdata as sd
from wpsnn import WPSNeuralNet
from mkidcore.readdict import ReadDict
import findResonatorsWPS as finder
from compareResLists import retrieveManResList, retrieveMLResList
from checkWPSPerformance import matchResonators, matchAttens

def trainAndCrossValidate(mlDict, trainFileLists, trainLabelsLists, valFileLists, valLabelsLists):
    modelName = mlDict['modelName']
    rootDir = os.path.join(mlDict['modelDir'], '..')
    trainNPZ = mlDict['trainNPZ'].split('.')[0]
    if not len(trainFileLists) ==  len(treanLabelsLists) == len(valFileLists) == len(valLabelsLists):
        raise Exception('Need to provide same number of train, val file/label sets')
    for i in range(len(trainFileLists)):
        mlDict.update({'modelName': modelName + '_' + str(i)})
        mlDict.update({'modelDir': os.path.join(rootDir, modelName + '_' + str(i))})
        mlDict.update({'trainNPZ': trainNPZ + '_' + str(i) + '.npz'})
        mlDict.update({'rawTrainFiles': trainFileLists[i]})
        mlDict.update({'rawTrainLabels': trainLabelsLists[i]})
        mlClass = WPSNeuralNet(mlDict)
        mlClass.initializeAndTrainModel()

        valThing = validateModel(mlDict['modelDir'], valFileLists[i], valLabelsLists[i])


def validateModel(modelDir, valFiles, valMDFiles):
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
        wpsmap, freqs, attens = finder.makeWPSMap(modelDir, sweep)
        resFreqs, resAttens, scores = finder.findResonators(wpsmap, freqs, attens, 
                peakThresh=0.94, nRes=1024)

        if resFreqs[0] < 4.7e9:
            band = 'a'
        else:
            band = 'b'
        try:
            fl = inst.guessFeedline(os.path.basename(args.inferenceData))
        except ValueError:
            fl = 1
        finder.saveMetadata(mdFile, resFreqs, resAttens, scores, fl, band)
    
    valThingList = []

    for i in range(len(mdFiles)): 
        manMD = sd.SweepMetadata(file=valMDFiles[i])
        mlMD = sd.SweepMetadata(file=mdFiles[i])
        mlResID, mlFreq, mlAtten, mlGoodMask= retrieveMLResList(mlMD)
        manResID, manFreq, manAtten, manGoodMask= retrieveManResList(manMD)

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
                freqDiff, valFiles[i], valMDFiles[i])
        valThing.savePlots(modelDir, os.path.basename(valFiles[i]).split('.')[0])
        fileName = os.path.join(modelDir, os.path.basename(valFiles[i]).split('.')[0] + '_validation.p')
        pickle.dump(valThing, open(fileName, 'w'))
        valThingList.append(valThing)


    return valThingList

class ValidationThing(object):

    def __init__(self, nMatched, nManNotInML, nMLNotInMan, confImage, attenDiffs, freqDiffs, sweepFiles=None, mdFiles=None):
        self.nMatched = nMatched
        self.nManNotInML = nManNotInML
        self.nMLNotInMan = nMLNotInMan
        self.confImage = confImage
        self.attenDiffs = attenDiffs
        self.freqDiffs = freqDiffs
        
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
        confImage = covalThing.nfImage + self.nfImage
        attenDiffs = np.append(attenDiffs, valThing.attenDiffs)
        freqDiffs = np.append(freqDiffs, valThing.freqDiffs)

        return ValidationThing(nMatched, nManNotInML, nMLNotInMan, confImage, attenDiffs, freqDiffs, sweepFiles=None, mdFiles=None)

    def savePlots(self, directory, prefix=None):
        if prefix is None:
            try:
                prefix = '_'.join(self.mdFiles)
            except TypeError:
                prefix = 'YOU_SUCK'

        plt.imshow(self.confImage)
        plt.savefig(os.path.join(directory, prefix + '_confusion.pdf'))
        plt.hist(self.attenDiffs, bins=10, range=(-4.5, 5.5))
        plt.savefig(os.path.join(directory, prefix + '_attenDiff.pdf'))
        plt.hist(self.freqDiffs, bins=20, range=(-100.e3, 100.e3))
        plt.savefig(os.path.join(directory, prefix + '_freqDiff.pdf'))


