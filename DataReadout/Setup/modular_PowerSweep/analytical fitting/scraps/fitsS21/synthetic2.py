import numpy as np
import matplotlib.pyplot as plt
import scraps as scr
import sys
import synthetic as syn
sys.path.append('/Data/venv/lib/python2.7/site-packages/scraps/fitsS21/')
import getxdetune as gx
import math


dfs = [2.83526508e-06]
f0 = 4.54800961e+09
qcs = [3.52091289e+04]
qis = [4.29318778e+05]

gain0s =[2.24601723e+03]
gain1s = [1.48834526e+06]
pgain0s =[1.22288109e+01]
pgain1s =[-2.59663203e+03]
a_s =   [1.72545353e+00]
# a_s = [0]

# f0 = 4.500e9
freqs = np.linspace(f0-5e5,f0+5e5,10000)
print freqs
# # df = 0.007
# dfs = [0]#,1e-6,0.5e-5,1e-5]
# qcs = [1e3,1e4]
# qis = [1e3,1e4]
# gain0s = [1]#, 10, 100]#,10,100,1000,10000]#1500
# gain1s = [0]#,10000,50000, 100000]#-2500000
# gain2s = [0]#1e9
# pgain0s = [0]#300
# pgain1s = [0]#5000,10000, 8000]#,1000000,-1000]#-6000
# a_s = [0]#,0.77,2,4,6]
# # a_s=[0]
offsets = [0]
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

import matplotlib
matplotlib.rcParams.update({'font.size': 16})
plt.rc('axes', titlesize='20')     # fontsize of the axes title
for ia, a in enumerate(a_s):
    for qi in qis:
        for qc in qcs:
            for df in dfs:
                for gain0 in gain0s:
                    for gain1 in gain1s:
                        # for gain2 in gain2s:
                        for pgain0 in pgain0s:
                            for pgain1 in pgain1s:
                                for offset in offsets:
                                    # qc = qcs# Make everything referenced to the shifted, unitless, reduced frequency
                                    print 'a', a, 'qi', qi, 'qc', qc, 'df', df, 'gain0', gain0, 'gain1', gain1, 'pgain0', pgain0, 'pgain1', pgain1

                                    fs = f0 + df
                                    ff = (freqs - fs) / fs

                                    # Except for the gain, which should reference the file midpoint
                                    # This is important because the baseline coefs shouldn't drift
                                    # around with changes in f0 due to power or temperature

                                    # Of course, this philosophy goes out the window if different sweeps have
                                    # # different ranges.
                                    fm = freqs[int(np.round((len(freqs) - 1) / 2.0))]
                                    ffm = (freqs - fm) / fm

                                    # Calculate the total Q_0
                                    q0 = 1. / (1. / qi + 1. / qc)

                                    y = q0*ff

                                    kwargs = {'ff': ff, 'f0': f0, 'q0': q0, 'qc': qc, 'a': a}
                                    # E_scale = kwargs.pop('E_scale')
                                    ff = gx.getxdetune(**kwargs)

                                    # S21 = 1 - q0/(qc*(1+2j*q0*ff))
                                    # plt.plot(ff, np.log(S21))
                                    # plt.show()

                                    gain = gain0 + gain1 * ff #+0.5 * gain2 * ff ** 2
                                    pgain = np.exp(1j * (pgain0 + pgain1 * ff))

                                    modelCmplx = gain * pgain * (1 - q0 / (qc * (1 + 2j * q0 * (ff + df))))  #+offset
                                    # Package complex data in 1D vector form
                                    modelI = np.real(modelCmplx)
                                    modelQ = np.imag(modelCmplx)


                                    # plt.plot(modelI, modelQ, label=pgain1)
                                    #
                                    # plt.figure()
                                    # alabels = ['0','$4\sqrt{3}/9$','2','4','6']
                                    # ax.text(3.9, -0.9, '$g_1$', fontsize=19)
                                    # ax.text(3.9, -1, '$\delta f$', fontsize=19)
                                    ax.plot(freqs / 1e9, 20*np.log10(np.abs(modelI + modelQ*1j)))#alabels[ia]

# leg = ax.legend(loc='lower right')
# leg.set_title('a', prop={'size': 14, 'weight': 'heavy'})
ax.set_xlabel('freqs')
ax.set_ylabel('$S_{21}$')
# plt.show()

paramsVec = [2.83526508e-06, 4.54800961e+09, 3.52091289e+04,  4.29318778e+05,
   2.24601723e+03,   1.48834526e+06,   1.22288109e+01,  -2.59663203e+03,
   1.72545353e+00]

cmplxResult = scr.cmplxIQ_fit(paramsVec, freqs)

I, Q = np.split(cmplxResult, 2)

res = syn.make_res(I, Q, freqs)
ps.plotResListData([res], plot_fits=False)