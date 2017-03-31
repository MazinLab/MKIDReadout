from datetime import datetime
startTime = datetime.now()
import sys
import numpy as np
from os import path
import scraps as scr
from matplotlib import pylab as plt

import WideSweep as ws
# from params import fspan, wind_width, end_score, datadir, rawsweepfile, manpeakfile, fitpeakfile, splineS_factor
from params import *

np.set_printoptions(threshold=np.nan)

print 'time to run:\t',
print datetime.now() - startTime

def fitres(data):
    fileDataDicts=[]

    pwr = -60
    resName = 'RES'
    temp = 100
    freqData = data[0]
    IData = data[1]
    QData = data[2]
    # plt.plot(log(IData**2 + QData**2))

    dataDict = {'name':resName+'_%i'%1,'temp':temp,'pwr':pwr*-1,'freq':freqData,'I':IData,'Q':QData}
    fileDataDicts.append(dataDict)

    resList = [scr.makeResFromData(fileDataDict) for fileDataDict in fileDataDicts]

    for res in resList:
        res.load_params(scr.cmplxIQ_params)
        kwargs={'maxfev':1000}
        res.do_lmfit(scr.cmplxIQ_fit, **kwargs)
        residual = sum(res.residualI**2+res.residualQ**2)
        
        if residual > 10000:
            'It looks like theres two peaks'
            res.load_params(scr.cmplxIQ_params_cols)
            # kwargs={'maxfev':2000}
            res.do_lmfit(scr.cmplxIQ_fit_cols, **kwargs)
        
        # if residual > 5000:
        #     res.load_params(scr.cmplxIQ_params)
        #     kwargs={'maxfev':5000}
        #     res.do_lmfit(scr.cmplxIQ_fit, **kwargs)
        #     residual = sum(res.residualI**2+res.residualQ**2)
    # return resList

    return resList#, res.resultI, res.resultQ

def plotResList(resList):
    figA = scr.plotResListData(resList,
                                    plot_types=['I', #Real vs Imaginary part of S21
                                                'Q', #residual of fits in IQ plane
                                                'IQ',
                                                'LogMag', #Magnitude of S21 vs frequency
                                                'LinMag'], #Phase of S21 vs frequency
                                    color_by='pwrs',
                                    fig_size=4,
                                    num_cols = 3,
                                    plot_fits = [True] * np.ones((5))) #<-- change this to true to overplot the best fits

    plt.show()

def getWindowVals(freqs,Is,Qs,center,width):
    freqs = freqs[center-width:center+width]
    Is = Is[center-width:center+width]
    Qs = Qs[center-width:center+width]

    return freqs, Is, Qs

def killpeak(Is,Qs,Isfit, Qsfit, center,width):
    Is[center-width:center+width] = Is[center-width:center+width] + Isfit
    Qs[center-width:center+width] = Qs[center-width:center+width] + Qsfit

    return Is, Qs

def flattenpeak(Is, Qs, center, width):
    width = 3*width
    
    # plt.plot(Is[center-width/2:center+width/2])
    # plt.plot(np.linspace(Is[center-width/2], Is[center+width/2], width))
    # plt.plot(Qs[center-width/2:center+width/2])
    # plt.plot(np.linspace(Qs[center-width/2], Qs[center+width/2], width))
    # plt.show()
    # Is[center-width/2:center+width/2] = np.linspace(Is[center-width/2], Is[center+width/2], width)
    # Qs[center-width/2:center+width/2] = np.linspace(Qs[center-width/2], Qs[center+width/2], width)**(1./2) 

    # (Is[center-width/2]+Qs[center-width/2])/2
    Is[center-width/2:center+width/2] = np.linspace(Is[center-width/2], Is[center+width/2], width)
    Qs[center-width/2:center+width/2] = Is[center-width/2:center+width/2]

    plt.plot(Is[center-width/2:center+width/2])
    plt.plot(Is[center-width/2:center+width/2]**2+Qs[center-width/2:center+width/2]**2)
    plt.show()

    return Is, Qs

def remove_slice(freqs,continuum, Is,Qs,center,width, extra_reach=20):
    radius = width/2 + extra_reach
    Is = np.concatenate([Is[:center-radius],Is[center+radius:]])
    Qs = np.concatenate([Qs[:center-radius],Qs[center+radius:]])
    freqs = np.concatenate([freqs[:center-radius],freqs[center+radius:]])
    continuum = np.concatenate([continuum[:center-radius],continuum[center+radius:]])

    return freqs,continuum, Is,Qs

def getPeakIndx(resList,freqs_orig, mag_orig):
    res = resList[0]
    freqs_orig = np.around(freqs_orig, decimals=5)

    f0_ind = res.lmfit_labels.index('f0')
    center1 = np.around(res.lmfit_vals[f0_ind], decimals=5)
    center1 = ws.find_nearest(freqs_orig,center1)
    center1 = ws.find_local_min(mag_orig,center1)

    if len(res.lmfit_vals) > 11:
        f02_ind = res.lmfit_labels.index('f02')
        center2 = np.around(res.lmfit_vals[f02_ind], decimals=5)
        center2 = ws.find_nearest(freqs_orig,center2)
        center2 = ws.find_local_min(mag_orig,center2)

        return np.asarray([center1, center2])
    else:
        return center1

# def reduce_to_span(data, span=-1):
#     if span ==-1:
#         span=[0,-1]
#
#     data = data[:, span[0]:span[1]]
#
#     return data

def find_peaks(data, savefile):
    startTime = datetime.now()

    freqs, Is, Qs, mag, continuum = data
    freqs_orig, Is_orig, Qs_orig, mag_orig = data[:4]     # save these values for later

    ws.check_continuum(freqs, mag, continuum)

    mag_adj = mag-continuum

    peaks = []
    end_criteria=False
    while not end_criteria:
        if0 = np.argmin(mag_adj)

        newfreqs, newIs, newQs = getWindowVals(freqs,Is,Qs,if0,wind_width)

        # plt.plot(mag_adj)
        # plt.show()

        print 'if0', if0, freqs[if0]
        resList = fitres([newfreqs, newIs, newQs])
        # plotResList(resList)

        peak_locs = getPeakIndx(resList, freqs_orig, mag_orig)
        print peak_locs, start_ind

        peak_locs = peak_locs + start_ind
        print peak_locs, type(peak_locs)
        if type(peak_locs)!=np.int64:
            for p in range(2):
                peaks.append(peak_locs[p])
        else:
            peaks.append(peak_locs)

        print peaks
        np.savetxt(savefile, peaks, fmt='%i', delimiter=',')

        # plt.plot(mag_adj)
        # plt.show()
        freqs, continuum, Is, Qs = remove_slice(freqs,continuum, Is,Qs,if0,wind_width)

        # Is, Qs = smooth_local(continuum, Is, Qs, if0)

        mag = ws.calc_mag(Is, Qs)

        mag_adj = mag-continuum
        # plt.plot(mag_adj)
        # plt.show()

        # av_mag_adj = -1*np.median(np.sqrt(abs(mag_adj)))

        sys.stdout.write("peaks found: %i\nadjusted depth: %f\t end point: %f\n" % (len(peaks), min(mag_adj), end_score))
        sys.stdout.flush()

        end_criteria = min(mag_adj) > end_score

    # remove duplicates
    peaks =list(set(peaks))

    # plt.plot(mag_adj)

    print 'time to run:\t',
    print datetime.now() - startTime

    return peaks

def get_peaks(data):

    print np.shape(data)

    if path.isfile(datadir+fitpeakfile):
        peaks = ws.load_peaks(datadir+fitpeakfile)
    else:
        peaks = find_peaks(data, savefile=datadir+fitpeakfile)

    return peaks

if __name__ == "__main__":
    data = ws.load_data()

    start_ind = ws.find_nearest(data[0], fspan[0])
    last_ind = ws.find_nearest(data[0], fspan[1])
    # data = reduce_to_span(data, span)
    data = ws.reduce_to_band_2d(data, data[0], fspan)

    peaks = get_peaks(data)

    span_found = [min(peaks), max(peaks)]
    man_peaks, man_peak_freqs = ws.load_man_click_locs(datadir+manpeakfile, span_found)

    if len(peaks) < 100:
        print peaks, man_peaks
    ws.compute_agreement(peaks, man_peaks)

    # mag_orig = reduce_to_span(load_data(), span)[3]

    # data = ws.reduce_to_band(data, data[0], fspan)
    # mag_orig = data[3]
    # logmag = 20*np.log10(mag_orig)
    ws.plot_peaks(data, [start_ind, last_ind], peaks, man_peaks)
