import numpy as np
import matplotlib.pyplot as plt
import getxdetune as gx

f0 = 4.6e9
freqs = np.linspace(f0-1e3,f0+1e3,1000)
# df = 0.007
dfs = [0]#1e-8,1e-7]
qcs = [1e8]
qis = [1e7,1e8]
gain0s = [1]#,10,100,1000,10000]#1500
gain1s = [0,1e4,1e6]#-2500000
gain2s = [0,1e13]#1e9
pgain0s = [0]#300
pgain1s = [0,1000000,-1000]#-6000
a_s = [0,0.1,2]

for a in a_s:
    for qi in qis:
        for qc in qcs:
            for df in dfs:
                for gain0 in gain0s:
                    for gain1 in gain1s:
                        for gain2 in gain2s:
                            for pgain0 in pgain0s:
                                for pgain1 in pgain1s:
                                    # qc = qcs# Make everything referenced to the shifted, unitless, reduced frequency
                                    print 'a', a, 'qi', qi, 'qc', qc, 'df', df, 'gain0', gain0, 'gain1', gain1, 'gain2', gain2, 'pgain0', pgain0, 'pgain1', pgain1

                                    fs = f0 + df
                                    ff = (freqs - fs) / fs

                                    # Except for the gain, which should reference the file midpoint
                                    # This is important because the baseline coefs shouldn't drift
                                    # around with changes in f0 due to power or temperature

                                    # Of course, this philosophy goes out the window if different sweeps have
                                    # different ranges.
                                    fm = freqs[int(np.round((len(freqs) - 1) / 2.0))]
                                    ffm = (freqs - fm) / fm

                                    # Calculate the total Q_0
                                    q0 = 1. / (1. / qi + 1. / qc)

                                    kwargs = {'ff': ff, 'f0': f0, 'q0': q0, 'qc': qc, 'a': a}
                                    # E_scale = kwargs.pop('E_scale')
                                    ff = gx.getxdetune(**kwargs)

                                    # S21 = 1 - q0/(qc*(1+2j*q0*ff))
                                    # plt.plot(ff, np.log(S21))
                                    # plt.show()

                                    gain = gain0 + gain1 * ffm + 0.5 * gain2 * ffm ** 2
                                    pgain = np.exp(1j * (pgain0 + pgain1 * ffm))

                                    modelCmplx = -gain * pgain * (1 - q0 / (qc * (1 + 2j * q0 * (ff + df))))  # +offset
                                    # Package complex data in 1D vector form
                                    modelI = np.real(modelCmplx)
                                    modelQ = np.imag(modelCmplx)


                                    # plt.plot(modelI, modelQ, label=pgain1)
                                    #
                                    # plt.figure()
                                    plt.plot(freqs, modelI**2 + modelQ**2, label=pgain1)
                                plt.legend()
                                plt.show()

