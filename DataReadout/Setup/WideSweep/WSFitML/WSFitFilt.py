'''
Implements a template filter to identify WS peaks
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from WSFitMLData import WSFitMLData
import os, sys

class WSTemplateFilt:
    def __init__(self):
        self.spacing = 12.5 #kHz
        self.winSize = int(500/self.spacing)

    def makeTemplate(self, trainFileList):
        self.trainData = WSFitMLData(trainFileList)
        self.trainData.loadPeaks()
        filtMagsDB = self.trainData.filterMags(self.trainData.magsdb)
        self.template = np.zeros(self.winSize)
        for loc in self.trainData.peakLocs:
            self.template += filtMagsDB[int(loc-self.winSize/2):int(loc+self.winSize/2)]

        self.template /= len(self.trainData.peakLocs)
        plt.plot(self.template)
        plt.show()

    def saveTemplate(self, filename):
        if self.template is None:
            raise Exception('Train template first!')
        np.savetxt(filename, self.template)

    def loadTemplate(self, filename):
        self.template = np.loadtxt(filename)

    def setWinSize(self, winSize):
        self.winSize = winSize
        if not self.template is None:
            if winSize>len(self.template):
                raise Exception('Existing template is too small')

    def inferPeaks(self, inferenceFile, sigThresh=0.75, nDerivChecks=7, nDerivSlack=0):
        if self.template is None:
            raise Exception('Train or load template first!')

        self.inferenceData = WSFitMLData([inferenceFile])
        self.inferenceFile = inferenceFile
        filtMagsDB = self.inferenceData.filterMags(self.inferenceData.magsdb)
        if not self.winSize==len(self.template):
            winSizeDiff = len(self.template) - self.winSize
            template = self.template[int(winSizeDiff/2):int(-winSizeDiff/2)]
        else:
            template = self.template

        tempFiltMagsDB = np.correlate(filtMagsDB, template, mode='same')

        threshold = sigThresh*np.std(tempFiltMagsDB)
        triggerBooleans = tempFiltMagsDB[nDerivChecks:-nDerivChecks-1] > threshold
        print 'threshold', threshold
        #plt.plot(tempFiltMagsDB[2010:4010])

        derivative = np.diff(tempFiltMagsDB)
        negDeriv = derivative <= 0
        posDeriv = np.logical_not(negDeriv)

        posDerivChecksSum = np.zeros(len(posDeriv[0:-2*nDerivChecks]))
        negDerivChecksSum = np.zeros(len(negDeriv[0:-2*nDerivChecks]))
        for i in range(nDerivChecks):
            posDerivChecksSum += posDeriv[i:i-2*nDerivChecks]
            negDerivChecksSum += negDeriv[i+nDerivChecks: i-nDerivChecks]
        peakCondition0 = negDerivChecksSum >= (nDerivChecks - nDerivSlack)
        peakCondition1 = posDerivChecksSum >= (nDerivChecks - nDerivSlack)
        peakCondition01 = np.logical_and(peakCondition0,peakCondition1)
        peakBooleans = np.logical_and(triggerBooleans,peakCondition01)
        self.peakIndices = np.where(peakBooleans)[0] + nDerivChecks
        #plt.plot(triggerBooleans[2000:4000]*tempFiltMagsDB[2010:4010])
        #plt.show()

    def markCollisions(self, resBWkHz=500):
        if self.peakIndices is None:
            raise Exception('Infer peak locations first!')
        minResSpacing = resBWkHz/self.spacing #resonators must be separated by this number of points
        peakSpacing = np.diff(self.peakIndices)
        collisionMask = peakSpacing<minResSpacing
        collisionInds = np.where(collisionMask)[0] #locations in peakIndices where there are collisions
        goodPeakInds = np.where(np.logical_not(collisionMask))[0]
        self.badPeakIndices = self.peakIndices[collisionInds]
        self.goodPeakIndices = self.peakIndices[goodPeakInds]

    def findLocalMinima(self):
        if self.peakIndices is None:
            raise Exception('Infer peak locations first!')
        foundMinima = np.zeros(len(self.peakIndices))
        # print (len(foundMinima))
        peakVals = self.inferenceData.magsdb
        while np.any(foundMinima==0):
            peakValsRight = np.roll(peakVals, -1)
            peakValsLeft = np.roll(peakVals, 1)
            peakValsRightLess = np.less_equal(peakVals[self.peakIndices], peakValsRight[self.peakIndices])
            peakValsLeftLess = np.less_equal(peakVals[self.peakIndices], peakValsLeft[self.peakIndices])
            foundMinima = np.logical_and(peakValsLeftLess, peakValsRightLess)
            
            peakValsRightGreater = np.logical_not(peakValsRightLess)
            peakValsLeftGreater = np.logical_and(peakValsRightLess, np.logical_not(foundMinima)) #not greater, but not a minimum
            peakValsRightGreaterInd = np.where(peakValsRightGreater)[0]
            peakValsLeftGreaterInd = np.where(peakValsLeftGreater)[0]

            self.peakIndices[peakValsRightGreaterInd] += 1
            self.peakIndices[peakValsLeftGreaterInd] -= 1
            # print sum(foundMinima)

    def saveInferenceFile(self):
        goodSaveFile = self.inferenceFile.split('.')[0]
        badSaveFile = self.inferenceFile.split('.')[0]
        goodSaveFile += '-ml-good.txt'
        badSaveFile += '-ml-bad.txt'

        np.savetxt(goodSaveFile, self.goodPeakIndices)
        np.savetxt(badSaveFile, self.badPeakIndices)

if __name__=='__main__':
    mdd = os.environ['MKID_DATA_DIR']
    templateDir = '.'
    if len(sys.argv)<3:
        raise Exception('Need to specify WS and template files')
    
    wsFile = os.path.join(mdd, sys.argv[1])
    templateFile = os.path.join(templateDir, sys.argv[2])
    
    wsFilt = WSTemplateFilt()
    wsFilt.loadTemplate(templateFile)
    wsFilt.inferPeaks(wsFile)
    wsFilt.findLocalMinima()
    wsFilt.markCollisions()
    wsFilt.saveInferenceFile()

