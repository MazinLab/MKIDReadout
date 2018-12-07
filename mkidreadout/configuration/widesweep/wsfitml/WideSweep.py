import numpy as np
from matplotlib import pylab as plt
from math import fabs
from scipy.interpolate import UnivariateSpline
# from datetime import datetime
import scfit as Sc
import logits as ML
from params import *
np.set_printoptions(threshold=np.nan)

def load_raw_wide_sweep(WideSweepFile, span=-1):
    # if path.isfile(WideSweepFile):
    if span ==-1:
        span=[0,-1]
    print 'loading raw widesweep data from %s' % WideSweepFile
    data = np.loadtxt(WideSweepFile, skiprows=3)

    freqs = data[span[0]:span[1],0]
    Is = data[span[0]:span[1],1]
    Qs = data[span[0]:span[1],2]

    return freqs, Is, Qs

def load_man_click_locs(ManClickFile, span_found=-1):
    print 'loading peak location data from %s' % ManClickFile
    man_data = np.loadtxt(ManClickFile)
    man_peaks = man_data[:,1]
    if span_found ==-1:
        # man_peaks = data[:,1]
        man_peak_freqs = man_data[:, 2]
    else:
        start = find_nearest(man_peaks, span_found[0])
        end = find_nearest(man_peaks, span_found[1])
        man_peaks = man_data[start:end+1, 1]
        man_peak_freqs = man_data[start:end+1, 2]

    return man_peaks, man_peak_freqs

def load_peaks(peaks_fit_file):
    peaks = np.loadtxt(peaks_fit_file, delimiter=',', ndmin=1)
    return peaks

def calc_mag(Is, Qs):
    S21 = Is + 1j*Qs
    mag = np.abs(S21)
    mag = mag/np.max(mag)

    return mag

def find_local_min(mag,center,width =10):
    minima = np.argmin(mag[center-width:center+width]) + center-width
    return minima

def check_close(array, value, atol=5):
    return sum(np.isclose(value,array,atol=atol))

def fitSpline(freqs, mag, splineS=200, splineK=2):
    x = freqs
    y = mag
    spline = UnivariateSpline(x,y,s=splineS, k=splineK)
    baseline = spline(x)

    return baseline

def get_continuum2(mag):
    nodes = []
    print type(mag)
    # mag = mag[:1000]
    # plt.plot(mag)
    imax = mag.argsort()[::-1]
    print imax[:25]
    # imax.append( np.argmax(mag) )
    nodes.append(imax[0])
    nodes.append(imax[1])
    print nodes
    for i in range(2, len(imax)):
        # print imax[i], continuum
        if imax[i] < nodes[0] or imax[i] > nodes[-1]: #or nearest neighbour > 100 samples away 
            nodes.append(imax[i])
            nodes.sort()

    print nodes
    continuum=[]
    for n in range(len(nodes)-1):
        # print n, nodes[n], nodes[n+1], mag[nodes[n]], mag[nodes[n+1]], np.linspace(mag[nodes[n]], mag[nodes[n+1]], nodes[n+1]-nodes[n])
        continuum = np.concatenate((continuum, np.linspace(mag[nodes[n]],mag[nodes[n+1]],nodes[n+1]-nodes[n])))

    print np.shape(continuum)
    # plt.plot(continuum)
    # plt.show()
    return continuum

def get_continuum(mag, frac):
    imax = mag.argsort()[::-1]
    imax = imax[:int(len(mag)*frac)]
    imax.sort()
    print imax[0], np.min(imax),  imax[-1], np.min(imax), mag[np.argmin(imax)]
    imax[0]=0#mag[np.argmin(imax)] #np.min(imax)
    imax[-1]=len(mag)-1#mag[np.argmin(imax)]
    # imax[-1]=np.min(imax)
    continuum=[]
    for n in range(len(imax)-1):
        # print np.linspace(mag[imax[n]], mag[imax[n+1]], imax[n+1]-imax[n])
        # print n, imax[n], imax[n+1], mag[imax[n]], mag[imax[n+1]], np.linspace(mag[imax[n]], mag[imax[n+1]], imax[n+1]-imax[n])
        continuum = np.concatenate((continuum, np.linspace(mag[imax[n]],mag[imax[n+1]],imax[n+1]-imax[n])))

    continuum = np.concatenate((continuum,[continuum[-1]]), axis=0)
    plt.plot(mag)
    plt.plot(continuum)
    plt.show()
    return continuum


def check_continuum(freqs, mag, continuum):
    plt.plot(freqs, continuum)
    # plt.plot(freqs, mag-continuum)
    plt.plot(freqs, mag)
    # plt.plot(freqs, continuum2)
    plt.show()

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or fabs(value - array[idx-1]) < fabs(value - array[idx])):
        # return array[idx-1]
        return idx-1
    else:
        # return array[idx]
        return idx

def reduce_to_band(var, freqs, fspan):
    if fspan == [0,-1]:
        start = 0
        end = -1
    else:
        start = find_nearest(freqs, fspan[0])
        end = find_nearest(freqs, fspan[1])

    var = var[start:end]

    return var

def reduce_to_band_2d(data, freqs, fspan):
    if fspan == [0,-1]:
        start = 0
        end = -1
    else:
        start = find_nearest(freqs, fspan[0])
        end = find_nearest(freqs, fspan[1])
        # print freqs[:100], freqs[-100:], start, end

    data = data[:, start:end]

    return data

def load_data():
    freqs, Is, Qs = load_raw_wide_sweep(datadir+rawsweepfile)

    # raw data before the peaks are cut out
    mag = calc_mag(Is, Qs)

    # spline tends to overfit for large spans
    splineS = len(freqs)*splineS_factor
    continuum = fitSpline(freqs, mag, splineS)
    print np.shape(freqs), np.shape(mag)
    # continuum2 = fitSVR(freqs, mag)

    check_continuum(freqs, mag, continuum)

    return np.asarray([freqs, Is, Qs, mag, continuum])

def compute_agreement(fit_peaks, man_peaks, ml_peaks):
    m = 0
    for ip,p in enumerate(fit_peaks):
        if check_close(p, man_peaks) > 0:
            m += 1.
    print '# from man: %i\t# from fit: %i\t# in both: %i\t ratio: %f' % \
          (len(man_peaks), len(fit_peaks), m, m/len(man_peaks))

    m = 0
    for ip,p in enumerate(ml_peaks):
        if check_close(p, man_peaks) > 0:
            m += 1.
    print '# from man: %i\t# from ML: %i\t# in both: %i\t ratio: %f' % \
          (len(man_peaks), len(ml_peaks), m, m/len(man_peaks))

    m = 0
    for ip,p in enumerate(fit_peaks):
        if check_close(p, ml_peaks) > 0:
            m += 1.
    print '# from ML: %i\t# from fit: %i\t# in both: %i\t ratio: %f' % \
          (len(ml_peaks), len(fit_peaks), m, m/len(ml_peaks))

    # combined = list(set(np.concatenate([ml_peaks, fit_peaks])))
    # m = 0
    # for ip,p in enumerate(combined):
    #     if check_close(p, man_peaks) > 0:
    #         m += 1.
    # print '# from man: %i\t# from fit: %i\t# in both: %i\t ratio: %f' % \
    #       (len(man_peaks), len(combined), m, m/len(man_peaks))

def plot_peaks(data, inds, fit_peaks=None, man_peaks=None, ml_peaks=None):
    freqs = data[0]
    mag = 20*np.log10(data[3])
    # print np.shape(freqs), np.shape(mag)
    # start = find_nearest(freqs, fspan[0])
    # end = find_nearest(freqs, fspan[1]+1)
    # print start, end
    x = np.arange(inds[0], inds[1])
    print np.shape(x), np.shape(mag)

    plt.figure()
    plt.plot(x, mag, alpha=0.5)
    if fit_peaks != None:
        plt.plot(fit_peaks, -10*np.ones((len(fit_peaks))), 'ro', label='fit')
        # for i in range(len(fit_peaks)):
            # plt.axvline(fit_peaks[i], color='r', ls='dashed')
    if man_peaks != None:
        plt.plot(man_peaks, -15*np.ones((len(man_peaks))), 'go', label='man')
        # for i in range(len(man_peaks)):
        #     plt.axvline(man_peaks[i], color='g', ls='-.')
    if ml_peaks != None:
        plt.plot(ml_peaks, -20*np.ones((len(ml_peaks))), 'mo', label='ML')
        # for i in range(len(ml_peaks)):
        #     plt.axvline(ml_peaks[i], color='m', ls='-.')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data = load_data()

    fit_peaks = Sc.get_peaks(data)

    ml_peaks = ML.get_peaks()

    span_found = [min([min(fit_peaks),min(ml_peaks)]), max([max(fit_peaks),max(ml_peaks)])]
    man_peaks = load_man_click_locs(datadir+manpeakfile, span_found)[0]

    print np.shape(fit_peaks), np.shape(man_peaks), np.shape(ml_peaks)
    print fit_peaks, man_peaks, ml_peaks

    mag = reduce_to_band(data[3], data[0], fspan)
    if fspan != [0,-1]:
        start_ind = find_nearest(data[0], fspan[0])
        last_ind = find_nearest(data[0], fspan[1])
    else:
        start_ind = 0
        last_ind = len(data[0])-1
    data = reduce_to_band_2d(data, data[0], fspan)

    compute_agreement(fit_peaks, man_peaks, ml_peaks)
    plot_peaks(data, [start_ind, last_ind], fit_peaks, man_peaks, ml_peaks)

