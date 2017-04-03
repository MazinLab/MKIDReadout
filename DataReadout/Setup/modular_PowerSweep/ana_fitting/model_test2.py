import numpy as np
import matplotlib.pyplot as plt
import getxdetune as gx

f0 = 4.6e9
freqs = np.linspace(f0-1e3,f0+2e3,1000)
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
qc = 1e8
qi = 1e8#,1e7]
gain0 = 1#,10,100,1000,10000]#1500
gain1 = 0#,1e4,1e6]#-2500000
gain2 = 0#,1e13]#1e9
pgain0 = 0#300
pgain1 = 0#,1000000,-1000]#-6000
a = 0#,0.1,2]

f02s = [4.6, 4.65, 4.7, 4.8]
df2 = 0#1e-8,1e-7]
qc2 = 1e8
qi2 = 1e8#,1e7]
gain02 = 1#,10,100,1000,10000]#1500
gain12 = 0#,1e4,1e6]#-2500000
gain22 = 0#,1e13]#1e9
pgain02 = 0#300
pgain12 = 0#,1000000,-1000]#-6000
a2 = 0#,0.1,2]

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
for f02 in f02s:
    # qc = qcs# Make everything referenced to the shifted, unitless, reduced frequency
    print 'qi', qi, 'qc', qc, 'df', df, 'gain0', gain0, 'gain1', gain1, \
        'gain2', gain2, 'pgain0', pgain0, 'pgain1', pgain1, 'Escale', Escale,\
        'pwr', pwr


    fs = f0 + df
    ff = (freqs - fs) / fs

    fs2 = f02 + df2
    ff2 = (freqs - fs2) / fs2

    fm = freqs[int(np.round((len(freqs) - 1) / 2.0))]
    ffm = (freqs - fm) / fm

    # Calculate the total Q_0
    q0 = 1. / (1. / qi + 1. / qc)

    # Calculate the total Q_0
    q02 = 1. / (1. / qi2 + 1. / qc2)

    ffm = ff
    kwargs = {'ff': ff, 'f0': f0, 'q0': q0, 'qc': qc, 'pwr': pwr, 'a': a}
    # E_scale = kwargs.pop('E_scale')
    ff = gx.getxdetune(**kwargs)

    kwargs = {'ff': ff2, 'f0': f02, 'q0': q02, 'qc': qc2, 'pwr': pwr, 'a': a2}
    # E_scale = kwargs.pop('E_scale')
    ff2 = gx.getxdetune(**kwargs)

    # S21 = 1 - q0/(qc*(1+2j*q0*ff))
    # plt.plot(ff, np.log(S21))
    # plt.show()

    # Calculate magnitude and phase gain
    gain = gain0 + gain1 * ffm + 0.5 * gain2 * ffm ** 2
    pgain = np.exp(1j * (pgain0 + pgain1 * ffm))

    # Calculate magnitude and phase gain
    gain2 = gain02 + gain12 * ffm + 0.5 * gain22 * ffm ** 2
    pgain2 = np.exp(1j * (pgain02 + pgain12 * ffm))

    # modelCmplx = -gain * pgain * (1 - q0 / (qc * (1 + 2j * q0 * (ff + df))))  # +offset
    modelCmplx = -gain * pgain * (
    1 - q0 / (qc * (1 + 2j * q0 * (ff + df)))) + -gain2 * pgain2 * (
        1 - q02 / (qc2 * (1 + 2j * q02 * (ff + df2))))
    # Package complex data in 1D vector form
    modelI = np.real(modelCmplx)
    modelQ = np.imag(modelCmplx)


    # plt.plot(modelI, modelQ, label=pgain1)
    #
    # plt.figure()
    plt.plot(freqs, modelI**2 + modelQ**2, label=Escale)
plt.legend()
plt.show()