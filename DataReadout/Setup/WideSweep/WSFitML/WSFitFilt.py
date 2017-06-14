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
            self.template += filtMagsDB[loc-self.winSize/2:loc+self.winSize/2]

        self.template /= len(self.trainData.peakLocs)
        plt.plot(self.template)
        plt.show()

    def saveTemplate(self, filename):
        if self.template is None:
            raise Exception('Train template first!')
        np.savetxt(filename, self.template)

    def inferPeaks(self, inferenceFile, sigThresh=0.75, nDerivChecks=7, nDerivSlack=0):
        if self.template is None:
            raise Exception('Train or load template first!')

        self.inferenceData = WSFitMLData([inferenceFile])
        filtMagsDB = self.inferenceData.filterMags(self.inferenceData.magsdb)
        tempFiltMagsDB = np.correlate(filtMagsDB, self.template, mode='same')

        threshold = sigThresh*np.std(tempFiltMagsDB)
        triggerBooleans = tempFiltMagsDB[nDerivChecks:-nDerivChecks-1] > threshold
        print 'threshold', threshold
        plt.plot(tempFiltMagsDB[2010:4010])

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
        plt.plot(triggerBooleans[2000:4000]*tempFiltMagsDB[2010:4010])
        plt.show()
        saveFile = inferenceFile.split('.')[0]
        saveFile += '-ml.txt'
        np.savetxt(saveFile, self.peakIndices)

if __name__=='__main__':
    mdd = os.environ['MKID_DATA_DIR']

    wsFilt = WSTemplateFilt()
    wsFilt.makeTemplate([os.path.join(mdd, 'Hexis_WS_FL3.txt')])
    wsFilt.inferPeaks(os.path.join(mdd, 'Hojira_FL3.txt'))

