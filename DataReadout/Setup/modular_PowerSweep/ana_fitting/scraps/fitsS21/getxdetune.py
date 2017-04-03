import numpy as np
from matplotlib import pylab as plt


def getxdetune(**kwargs):
    # if kwargs is not None:
    #     for key, value in kwargs.iteritems():
    #         print "%s == %s" %(key,value)

    f0 = kwargs.pop('f0')
    q0 = kwargs.pop('q0')
    qc = kwargs.pop('qc')
    # pwr = kwargs.pop('pwr')

    ff0 = kwargs.pop('ff')
    # Escale = kwargs.pop('Escale')
    a = kwargs.pop('a')
    # print 'pwr', pwr
    # pwr = 1e-3 * 10 ** (pwr / 10)
    # print pwr

    # E_scale = (2 * q0s[0]**3 * pwr) /(0.8 * qcs[0] * f0s[0])
    # E_scale = 2e-8

    y0s = ff0 * q0

    # rearranged equation --> -4y**3 + 4y0*y**2 - y + y0 + a
    # a = 2 * q0 ** 3 * pwr / (qc * f0 * Escale)
    # print 'a', a, 'Escale', Escale

    ys = []
    for y0 in y0s:
        coeffs = [-4, 4 * y0, -1, a + y0]
        roots = np.roots(coeffs)
        # print roots
        real_loc = np.where(np.imag(roots) == 0.)[0]
        ys.append(roots[real_loc])

    ys = np.asarray(ys)
    y1s = []
    y2s = []

    non_mon_ind = 0
    for iy, y in enumerate(ys):
        if iy < len(ys) - 1:
            if len(ys[iy + 1]) > 1 and len(ys[iy]) == 1:
                non_mon_ind = iy + 1  # non-monotonic index

        if len(y) > 1:
            y1s.append(y[0])
            y2s.append(y[1])
            ys[iy] = y[2]

        else:
            ys[iy] = np.real(y)[0]

    yrange1 = np.real(ys[:non_mon_ind])
    if y1s == []:
        yrange2 = np.transpose([y1s])
    else:
        yrange2 = y1s
    yrange3 = np.real(ys[non_mon_ind + len(y1s):])

    # print type(yrange1)
    # print np.shape(yrange1), np.shape(yrange2), np.shape(yrange3)
    # if np.shape(yrange2)[0]==40:
    # print yrange1,yrange2, yrange3
    # print yrange1,yrange3
    # plt.plot(y0s, ys, 'o')
    # plt.plot(y0s[non_mon_ind:len(y1s)+non_mon_ind], y1s, 'o')
    # plt.plot(y0s[non_mon_ind:len(y1s)+non_mon_ind], y2s, 'o')
    # # plt.plot(y0s, ys_out)
    # plt.plot(y0s, yrange2, 'o')
    # plt.show()
    try:
        ys_out = np.concatenate([yrange1, yrange2, yrange3])
    except ValueError:
        ys_out = yrange2

    ff = np.zeros((len(ys_out)))
    for i in range(len(ff)):
        ff[i] = np.asarray(ys_out[i] / q0)

    # S21 = 1 - q0/(qc*(1+2j*q0*ff0))

    # plt.plot(ff0, np.log(S21), 'o')

    # print ff
    # S21 = 1 - q0/(qc*(1+2j*q0*ff))
    # plt.plot(ff, np.log(S21))
    # plt.show()

    return ff