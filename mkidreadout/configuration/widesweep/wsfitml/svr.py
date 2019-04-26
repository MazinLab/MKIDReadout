import matplotlib.pyplot as plt
import numpy as np
from mkidreadout.configuration.widesweep.wsfitml.params import datadir, rawsweepfile
from sklearn.svm import SVR

import mlutils as ws


def fitSVR(freqs, mag, C, gamma):
    x = np.reshape(freqs, (len(freqs),1))
    y = mag
    svr_rbf = SVR(kernel='rbf', C=C, gamma= gamma)
    svr_rbf.fit(x, mag)
    return svr_rbf.predict(x)

freqs, Is, Qs = ws.load_raw_wide_sweep(datadir+rawsweepfile)

# raw data before the peaks are cut out
mag = ws.calc_mag(Is, Qs)

freqs = ws.reduce_to_band(freqs,freqs, [4,6])
mag = ws.reduce_to_band(mag,freqs, [4,6])
freqs = freqs[:-1]

print np.shape(freqs), np.shape(mag)
plt.plot(freqs, mag)

for gamma in [1e2]: #1e2 ,1e3,1e4
    print gamma
    for C in [1e3]: #1, 1e6,1e7,
        print C
        fit = fitSVR(freqs, mag, C, gamma)
        plt.plot(freqs, fit, label='%1.4f %1.4f' % (C, gamma))
plt.legend()
plt.show()

np.savetxt(datadir+'SVR'+rawsweepfile, fit, fmt='%i', delimiter=',')
