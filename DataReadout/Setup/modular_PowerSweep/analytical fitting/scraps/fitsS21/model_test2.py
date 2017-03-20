import numpy as np
import matplotlib.pyplot as plt
import getxdetune as gx

f0 = 4.5004e9
freqs = np.linspace(4.50e9,4.501e9,10000)
# df = 0.007
# dfs = [0]#1e-8,1e-7]
# qcs = [1e8]
# qis = [1e8]#,1e7]
# gain0s = [1]#,10,100,1000,10000]#1500
# gain1s = [0]#,1e4,1e6]#-2500000
# gain2s = [0]#,1e13]#1e9
# pgain0s = [0]#300
# pgain1s = [0]#,1000000,-1000]#-6000
# a_s = [0,0.1,2]
# # Escales = [1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
# # pwrs = [-40,-50,-60]

df = 0#1e-8,1e-7]
qc = 1e6
qi = 1e6#,1e7]
gain0 = 1#,10,100,1000,10000]#1500
gain1 = 0#,1e4,1e6]#-2500000
gain2 = 0#,1e13]#1e9
pgain0 = 0#300
pgain1 = 0#,1000000,-1000]#-6000
a = 0#,0.1,2]

f0ds = [4.5006e9]#[4.9e9,5.2e9,6e9]
dfd = 0#1e-8,1e-7]
qcd = 1e6
qid = 1e6#,1e7]
gain0d = 1#,10,100,1000,10000]#1500
gain1d = 0#,1e4,1e6]#-2500000
gain2d = 0#,1e13]#1e9
pgain0d = 0#300
pgain1d = 0#,1000000,-1000]#-6000
ad = 0#,0.1,2]

# for a in a_s:
# for qi in qis:
#     for qc in qcs:
#         for df in dfs:
#             for gain0 in gain0s:
#                 for gain1 in gain1s:
#                     for gain2 in gain2s:
#                         for pgain0 in pgain0s:
#                             for pgain1 in pgain1s:
#                                 for pwr in pwrs:
#                                     for Escale in Escales:
for f0d in f0ds:
    # qc = qcs# Make everything referenced to the shifted, unitless, reduced frequency
    print 'qi', qi, 'qc', qc, 'df', df, 'gain0', gain0, 'gain1', gain1, \
        'gain2', gain2, 'pgain0', pgain0, 'pgain1', pgain1, 'f0d', f0d


    fs = f0 + df
    ff = (freqs - fs) / fs

    fsd = f0d + dfd
    ffd = (freqs - fsd) / fsd

    # fm = freqs[int(np.round((len(freqs) - 1) / 2.0))]
    # ffm = (freqs - fm) / fm

    # Calculate the total Q_0
    q0 = 1. / (1. / qi + 1. / qc)

    # Calculate the total Q_0
    q0d = 1. / (1. / qid + 1. / qcd)

    ffm = ff
    kwargs = {'ff': ff, 'f0': f0, 'q0': q0, 'qc': qc, 'a': a}
    # E_scale = kwargs.pop('E_scale')
    ff = gx.getxdetune(**kwargs)

    kwargs = {'ff': ffd, 'f0': f0d, 'q0': q0d, 'qc': qcd, 'a': ad}
    # E_scale = kwargs.pop('E_scale')
    ffd  = gx.getxdetune(**kwargs)

    # S21 = 1 - q0/(qc*(1+2j*q0*ff))
    # plt.plot(ff, np.log(S21))
    # plt.show()

    # Calculate magnitude and phase gain
    gain = gain0 + gain1 * ff + 0.5 * gain2 * ff ** 2
    pgain = np.exp(1j * (pgain0 + pgain1 * ff))

    # Calculate magnitude and phase gain
    gaind = gain0d + gain1d * ffd + 0.5 * gain2d * ffd ** 2
    pgaind = np.exp(1j * (pgain0d + pgain1d * ffd))

    # modelCmplx = -gain * pgain * (1 - q0 / (qc * (1 + 2j * q0 * (ff + df))))  # +offset
    modelCmplx = -gain * pgain * (1 - q0 / (qc * (1 + 2j * q0 * (ff + df)))) + -gaind * pgaind * (
        1 - q0d / (qcd * (1 + 2j * q0d * (ffd + dfd))))
    # Package complex data in 1D vector form
    modelI = np.real(modelCmplx)
    modelQ = np.imag(modelCmplx)


    # plt.plot(modelI, modelQ, label=pgain1)
    #
    # plt.figure()
    plt.plot(freqs, modelI**2 + modelQ**2, label=f0d)
plt.legend()
plt.show()