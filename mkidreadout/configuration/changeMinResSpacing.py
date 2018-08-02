import numpy as np
import os, sys

#todo make part of clickthrough result object

if __name__=='__main__':
    freqFileName = str(sys.argv[1])
    resSpacing = int(sys.argv[2]) #in kHz
    freqFile = os.path.join(os.environ['MKID_DATA_DIR'], freqFileName)
    resIDs, locs, freqs = np.loadtxt(freqFile, unpack=True)
    freqDiffs = np.diff(freqs)
    goodInds = np.where(freqDiffs>resSpacing/1.e6)
    goodIDs = resIDs[goodInds]
    goodLocs = locs[goodInds]
    goodFreqs = freqs[goodInds]

    baseFileName = freqFileName.split('.')[0]
    goodFileName = baseFileName + '-' + str(resSpacing) + 'kHz.txt'
    goodFile = os.path.join(os.environ['MKID_DATA_DIR'], goodFileName)
    np.savetxt(goodFile, np.transpose([goodIDs, goodLocs, goodFreqs]), fmt='%4i %4i %0.7f')
