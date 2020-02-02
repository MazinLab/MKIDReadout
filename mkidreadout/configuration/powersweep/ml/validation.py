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

def trainAndCrossValidate(mlDict, rootDir, trainFileLists, trainLabelsLists, valFileLists, valLabelsLists):
    modelName = mlDict['modelName']
    trainNPZ = mlDict['trainNPZ'].split('.')[0]
    valThingList = []
    if not len(trainFileLists) ==  len(trainLabelsLists) == len(valFileLists) == len(valLabelsLists):
        raise Exception('Need to provide same number of train, val file/label sets')
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

        valThing = validateModel(mlDict['modelDir'], valFileLists[i], valLabelsLists[i])
        valThing.savePlots(mlDict['modelDir'], 'fullValidation')
        pickle.dump(valThing, os.path.join(mlDict['modelDir'], 'fullValidation.p'))
        valThingList.append(valThing)

    fullValThing = valThingList[0]
    for valThing in valThingList[1:]:
        fullValThing += valThing
    fullValThing.savePlots(rootDir, modelName + '_full_val')
    pickle.dump(fullValThing, os.path.join(rootDir, modelName + '_full_val.p'))



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
            fl = inst.guessFeedline(os.path.basename(mdFile))
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


    fullValThing = valThingList[0]
    for valThing in valThingList[1:]:
        fullValThing += valThing

    return fullValThing

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
        confImage = valThing.confImage + self.confImage
        attenDiffs = np.append(self.attenDiffs, valThing.attenDiffs)
        freqDiffs = np.append(self.freqDiffs, valThing.freqDiffs)

        return ValidationThing(nMatched, nManNotInML, nMLNotInMan, confImage, attenDiffs, freqDiffs, sweepFiles=None, mdFiles=None)

    def savePlots(self, directory, prefix=None):
        if prefix is None:
            try:
                prefix = '_'.join(self.mdFiles)
            except TypeError:
                prefix = 'YOU_SUCK'

        plt.imshow(self.confImage, vmax=55)
        plt.colorbar()
        plt.savefig(os.path.join(directory, prefix + '_confusion.pdf'))
        plt.close()
        plt.hist(self.attenDiffs, bins=10, range=(-4.5, 5.5))
        plt.xlabel('ML Atten - Manual Atten')
        plt.savefig(os.path.join(directory, prefix + '_attenDiff.pdf'))
        plt.close()
        plt.hist(self.freqDiffs, bins=20, range=(-100.e3, 100.e3))
        plt.xlabel('ML Freq - Manual Freq')
        plt.savefig(os.path.join(directory, prefix + '_freqDiff.pdf'))
        plt.close()


