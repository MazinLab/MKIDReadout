'''
Implements a template filter to identify WS peaks
'''
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from mkidreadout.configuration.widesweep.wsdata import WSFitMLData
import os
import argparse
ANALOG_TEMPLATE_SPACING = 12.5


class AutoPeakFinder:
    def __init__(self, spacing):
        self.spacing = spacing #kHz
        self.winSize = int(500/self.spacing)

    def setWinSize(self, winSize):
        self.winSize = winSize
        if not self.template is None:
            if winSize>len(self.template):
                raise Exception('Existing template is too small')

    def inferPeaks(self, inferenceFile, isDigital, sigThresh=0.5):
        self.inferenceData = WSFitMLData([inferenceFile], freqStep=self.spacing/1.e6)
        self.inferenceFile = inferenceFile
        if isDigital:
            self.inferenceData.stitchDigitalData()
            #self.inferenceData.saveData(inferenceFile.split('.')[0] + '_stitched.txt')
            
        bpFilt = signal.firwin(1001, (0.7*0.005*12.5/7.6, 0.175), pass_zero=False, window=('chebwin', 100))
        firFiltMagsDB = np.convolve(self.inferenceData.magsdb, bpFilt, mode='same')
        #bpFiltLP = signal.firwin(1001, 0.175, pass_zero=True, window=('chebwin', 100))
        #bpFiltHP = signal.firwin(1001, 0.75*0.005*12.5/7.6, pass_zero=False, window=('chebwin', 100))
        #firFiltMags2 = np.convolve(self.inferenceData.mags, bpFiltLP, mode='same')
        #firFiltMagsDB2 = 20*np.log10(firFiltMags2)
        #firFiltMagsDB2 = np.convolve(firFiltMagsDB2, bpFiltHP, mode='same')

        freqs = self.inferenceData.freqs
        thresh = sigThresh*np.std(firFiltMagsDB)
        print 'threshold', thresh
        peaks, _ = signal.find_peaks(-firFiltMagsDB, prominence=thresh, width=2, wlen=35)
        morePeaks, _ = signal.find_peaks(-firFiltMagsDB, thresh, prominence=0.4*thresh, width=2, wlen=30)
        peaks = np.append(peaks, morePeaks)
        peaks = np.unique(peaks)
        peaks = np.sort(peaks)
        print 'sp signal found', len(peaks), 'peaks'
        #plt.plot(freqs, filtMagsDB, label='cheby iir mags')
        #plt.plot(freqs, tempFiltMagsDB/np.sum(template), label='temp filt cheby iir mags')
        #plt.plot(freqs, firFiltMagsDB, label = 'fir cheby window')
        #plt.plot(freqs, self.inferenceData.magsdb - np.mean(self.inferenceData.magsdb), label='raw data')
        ##plt.plot(freqs, firFiltMagsDB2, label = 'fir cheby window 2')
        #plt.plot(freqs[peaks], firFiltMagsDB[peaks], '.', label = 'signal peaks')
        #plt.legend()



        self.peakIndices = peaks

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
        goodSaveFile += '_stitched-ml-good.txt'
        badSaveFile += '_stitched-ml-bad.txt'

        np.savetxt(goodSaveFile, self.goodPeakIndices)
        np.savetxt(badSaveFile, self.badPeakIndices)


if __name__=='__main__':
    mdd = os.environ['MKID_DATA_DIR']
    
    parser = argparse.ArgumentParser(description='WS Auto Peak Finding')
    parser.add_argument('wsDataFile', nargs=1, help='Raw Widesweep data')
    parser.add_argument('-d', '--digital', action='store_true', help='Perform preprocessing step for digital data')
    args = parser.parse_args()

    wsFile = args.wsDataFile[0]
    if not os.path.isfile(wsFile):
        wsFile = os.path.join(mdd, wsFile)

    sigmaThresh = 0.5
    
    if args.digital:
        spacing = 7.629
    else:
        spacing = 12.5

    wsFilt = AutoPeakFinder(spacing)
    wsFilt.inferPeaks(wsFile, args.digital, sigmaThresh)
    wsFilt.findLocalMinima()
    wsFilt.markCollisions(200)
    print 'Found', len(wsFilt.goodPeakIndices), 'good peaks'
    wsFilt.saveInferenceFile()

