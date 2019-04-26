import os
import sys

import numpy as np

#todo make part of a frequency comb object

if __name__=='__main__':
    fn = sys.argv[1]
    freqShift = float(sys.argv[2])
    mdd = os.environ['MKID_DATA_DIR']
    resIDs, freqs, attens = np.loadtxt(os.path.join(mdd, fn), unpack=True)
    freqs += freqShift
    newfn = fn[:-4] + '_shifted_' + str(freqShift/1000) +'_kHz.txt'
    data = np.transpose([resIDs, freqs, attens])
    np.savetxt(os.path.join(mdd, newfn), data, fmt="%4i %10.1f %4i")
