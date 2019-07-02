N_CLASSES = 4 #good, saturated, underpowered, bad/no res


class WPSNeuralNet(object):
    
    def __init__(self, mlDict):
        self.mlDict = mlDict
        self.nClasses = N_CLASSES
        self.nColors = 2
        if mlDict['useIQV']:
            self.nColors += 1
        if mlDict['useVectIQV']:
            self.nColors += 2

        self.imageShape = (mlDict['attenWinBelow'] + mlDict['attenWinAbove'] + 1, mlDict['freqWinSize'], self.nColors)

    def makeTrainData(self):
        trainImages = np.empty((0, *self.imageShape))
        testImages = np.empty((0, *self.imageShape))
        trainLabels = np.empty((0, self.nClasses))
        testLabels = np.empty((0, self.nClasses))
        for i, rawTrainFile in enumerate(self.mlDict['rawTrainFiles']):
            rawTrainFile = os.path.join(self.mlDict['trainFileDir'], rawTrainFile)
            rawTrainMDFile = os.path.join(self.mlDict['trainFileDir'], self.mlDict['rawTrainLabels'])
            trainMD = SweepMetadata(rawTrainMDFile)
            trainSweep = FreqSweep(rawTrainFile)

            goodResMDMask = ~np.isnan(trainMD.atten)
            attenblock = np.tile(trainSweep.atten, (len(goodResMDMask),1))
            optAttenInds = np.argmin(np.abs(attenblock.T - trainMD.atten), axis=0)
            
            if self.mlDict['trimAttens']:
                goodResMask = goodResMask & ~(optAttenInds < self.mlDict['attenWinBelow'])
                goodResMask = goodResMask & ~(optAttenInds >= (len(trainSweep.attens) - self.mlDict['attenWinAbove']))
            if self.mlDict['filterMaxedAttens']:
                maxAttenInd = np.argmax(trainSweep.atten)
                goodResMask = goodResMask & ~(optAttenInds==maxAttenInd)
                print 'Filtered', np.sum(rawTrainData.opt_iAttens==maxAttenInd), 'maxed out attens.'

            images = np.zeros((self.mlDict['nImagesPerRes']*self.nClasses*np.sum(goodResMask), *self.imageShape))
            labels = np.zeros((self.mlDict['nImagesPerRes']*self.nClasses*np.sum(goodResMask), self.nClasses))

            optAttens = trainMD.atten[goodResMask]
            optFreqs = trainMD.freq[goodResMask]
            optAttenInds = optAttenInds[goodResMask]

            imgCtr = 0
            for i in range(np.sum(goodResMask)):
                for j in range(self.mlDict['nImagesPerRes']):
                    images[imgCtr] = mlt.makeWPSImage(trainSweep, optFreqs[i], optAttens[i], self.imageShape[1], 
                        self.imageShape[0], self.mlDict['useIQV'], self.mlDict['useVectIQV']) #good image
                    labels[imgCtr] = np.array([1, 0, 0, 0])
                    imgCtr += 1
                    satResMask = np.ones(len(trainSweep.attens))
                    satResMask[optAttenInds[i] - self.mlDict['trainSatThresh']:] = 0
                    if np.any(satResMask):
                        satResInds = np.where(satResMask)[0]
                        satResInd =  np.random.choice(satResInds)
                        images[imgCtr] = mlt.makeWPSImage(trainSweep, optFreqs[i], satResInd, self.imageShape[1], 
                            self.imageShape[0], self.mlDict['useIQV'], self.mlDict['useVectIQV']) #saturated image
                        labels[imgCtr] = np.array([0, 1, 0, 0])
                        imgCtr += 1

                    upResMask = np.ones(len(trainSweep.attens))
                    upResMask[:optAttenInds[i] + self.mlDict['trainUPThresh']] = 0
                    if np.any(upResMask):
                        upResInds = np.where(upResMask)[0]
                        upResInd =  np.random.choice(upResInds)
                        images[imgCtr] = mlt.makeWPSImage(trainSweep, optFreqs[i], upResInd, self.imageShape[1], 
                            self.imageShape[0], self.mlDict['useIQV'], self.mlDict['useVectIQV']) #upurated image
                        labels[imgCtr] = np.array([0, 0, 1, 0])
                        imgCtr += 1
                


