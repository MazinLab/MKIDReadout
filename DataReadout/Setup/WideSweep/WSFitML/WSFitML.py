from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
import time
import os
# from params import datadir, fspan, rawsweepfile, manpeakfile, train_raw_sweep_file, train_man_peak_file, delta_left, delta_right, splineS_factor
from params import *
import WideSweep as ws

def plot_peaks(freqs, S11, prediction, peaks):
    # Plot data
    plt.figure()
    print len(freqs), len(prediction[:])
    plt.scatter(freqs, prediction[:], color='m')
    for i in range(len(peaks)):
        plt.axvline(freqs[peaks[i]], color='r', ls='dashed')
    # plt.scatter(freqs, prediction_new[:], color='c')
    # plt.scatter(freqs, prob[:, 1], color='g')
    plt.plot(freqs, S11, '.-b')

    plt.xlabel('Frequency (GHz)')
    plt.ylabel('S11 (dB)')
    # plt.xlim([np.min(X_new[:, 0]), np.max(X_new[:, 0])])
    # plt.xlim([5.05, 5.1])
    plt.show()
    return

def find_centers(prediction, S11):
    peak_locs = np.where(prediction==1)
    # print peaks[:10]
    peak_locs = np.split(peak_locs[0], np.where(np.diff(peak_locs[0]) > 5)[0]+1)
    # print peaks[:10]

    peaks = []
    for p in peak_locs:
        if len(p)> 2:
            p = np.array(p)
            min_location = np.argmin(S11[p, 1])
            peaks.append(p[min_location])

    return peaks

def find_peaks(S11, man_peaks, S11_test):

    # Create a good/bad resonator array using indices from the solution file
    resonator_mask = np.zeros((len(S11))) # Array with same shape as raw I/Q data

    for x in man_peaks:
        # if x > 20:
        # gb_resonators[int(x)-delta_left:int(x)+delta_right, 0] = S11[int(x)-delta_left:int(x)+delta_right, 0]  # Frequencies
        resonator_mask[int(x)-delta_left:int(x)+delta_right] = 1  # Good resonators = 1

    # for idx, data in enumerate(solutionset[:-1]):
    #     if solutionset[idx+1, 2] - solutionset[idx, 2] < 0.015:

    # Create classifier - LogisticRegression
    clf2 = LogisticRegression().fit(S11, resonator_mask)
    prediction = clf2.predict(S11_test)  # New function below
    # prob = clf2.predict_proba(S11_test)

    # New prediction function
    # prediction_new = np.zeros_like(X_new[:, 0])
    # prediction_new[np.where(prob[:, 1] > 0.2)] = 1

    peaks = find_centers(prediction, S11_test)

    return peaks, prediction

def get_peaks():
    # Import data
    trainset = np.asarray(ws.load_raw_wide_sweep(datadir+train_raw_sweep_file))
    testset = np.asarray(ws.load_raw_wide_sweep(datadir+rawsweepfile))

    start_ind = ws.find_nearest(testset[0], fspan[0])
    print np.shape(testset)
    testset = ws.reduce_to_band_2d(testset, testset[0], fspan)
    print np.shape(testset)

    # Create amplitude and S11 from I/Q data
    mag = ws.calc_mag(trainset[1], trainset[2])
    mag_test = ws.calc_mag(testset[1], testset[2])

    # splineS = len(mag)*splineS_factor
    # continuum_train = ws.fitSpline(trainset[0], mag, splineS)
    # ws.check_continuum(trainset[0], mag, continuum_train)

    # splineS = len(mag_test)*splineS_factor * 0.5
    # continuum_test = ws.fitSpline(testset[0], mag_test, splineS)
    # ws.check_continuum(testset[0], mag_test, continuum_test)

    # mag = mag - continuum_train
    # mag_test = mag_test - continuum_test

    S11 = np.array([trainset[0, :], 20*np.log10(mag)]).T  # Normalise
    S11_test = np.array([testset[0, :], 20*np.log10(mag_test)]).T  # Normalise

    man_peaks, _ = ws.load_man_click_locs(datadir+train_man_peak_file)
    # test_man_peaks, _ = ws.load_man_click_locs(datadir+manpeakfile)

    peaks, prediction = find_peaks(S11, man_peaks, S11_test)
    plot_peaks(testset[0], S11_test, prediction, peaks)

    peaks = peaks + start_ind
    return peaks, start_ind

def make_peak_file(peaks, start_ind, mlFile):
    if os.path.isfile(mlFile):
        mlFile = mlFile+time.strftime("-%Y-%m-%d-%H-%M-%S")
        #shutil.copy(self.mlFile, self.mlFile+time.strftime("-%Y-%m-%d-%H-%M-%S"))
    mlf = open(mlFile,'wb') #mlf machine learning file is temporary

    for pl in peaks:
        line = "%12d\n" % (pl+start_ind) # just peak locations
        mlf.write(line)
    mlf.close()


if __name__ == "__main__":
    peaks, start_ind = get_peaks()
    make_peak_file(peaks, start_ind, mlFile)