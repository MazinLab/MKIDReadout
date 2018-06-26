import numpy as np
import os, sys

if __name__=='__main__':
    fn = str(sys.argv[1])
    mdd = os.environ['MKID_DATA_DIR']
    freqFile = os.path.join(mdd, fn)
    resIDs, freqs, attens = np.loadtxt(freqFile, unpack=True)
    freqDiffs = np.diff(freqs)
    doubleLocs = np.where(freqDiffs<200000)[0]
    freqs = np.delete(freqs, doubleLocs)
    attens = np.delete(attens, doubleLocs)
    resIDs = np.delete(resIDs, doubleLocs)
    newFreqFile = freqFile[:-4] + '_rm_doubles.txt'
    data = np.transpose([resIDs, freqs, attens])
    np.savetxt(newFreqFile, data, fmt="%4i %10.1f %4i")
