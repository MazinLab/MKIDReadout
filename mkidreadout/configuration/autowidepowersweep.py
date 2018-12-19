import sys
# from mkidreadout.configuration.powersweep.ml.PSFitMLData import PSFitMLData

import matplotlib.pylab as plt

from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np
from scipy.optimize import curve_fit
import scipy.signal as signal
from mkidcore.corelog import getLogger
import mkidreadout.configuration.sweepdata as sweepdata
# from mkidreadout.configuration.widesweep.autowidesweep import Finder
import re
import argparse


import sys, os
sys.path.append(os.environ['MEDIS_DIR'])
from Utils.misc import dprint

def plot_left():
    figManager = plt.get_current_fig_manager()
    figManager.window.move(-1920, 0)
    figManager.window.setFocus()
    plt.show()

def gaussian(x, sig,mu):
    pdf = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    pdf = pdf/sum(pdf)
    return pdf

def poly(x, a, b, c):
    pdf = a*x**2 + b*x + c
    pdf = pdf/sum(pdf)
    return pdf

def expon(x, A, b, c):
    pdf = A * np.exp(-b * x) + c
    pdf = pdf / sum(pdf)
    return pdf

def smooth(y, box_pts=10):
    box = np.ones((box_pts))/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def create_ranges(start, stop, N, endpoint=True, noise=100):
    if endpoint==1:
        divisor = N-1
    else:
        divisor = N
    steps = (1.0/divisor) * (stop - start)
    return steps[:,None]*np.arange(N) + start[:,None] + np.random.random(N)*noise#500

class widePowerSweep():
    def __init__(self, npzFile=None, use_autowidesweep=False):



        self.npzFile = npzFile
        fn = 'psData_222.npz'
        data = np.load(npzFile)

        print(data['atten'].shape)  # 1d [nAttens] dB
        print(data['freqs'].shape)  # 2d [nTones, nLOsteps] Hz
        print(data['I'].shape ) # 3d [nAttens, nTones, nLOsteps] ADC units
        print(data['Q'].shape)

        self.atten = data['atten']
        self.widesweep = data['freqs'].flatten()

        self.num_atten = data['I'].shape[0]

        self.wide_mags = data['I'].reshape(self.num_atten,-1)**2 + data['Q'].reshape(self.num_atten,-1)**2
        self.wide_mags = self.wide_mags[::-1]

        self.step = self.widesweep[2] - self.widesweep[1]

        self.freqStep=self.step/1.e6

    def stitch(self, wide_mags, next_mags, nOverlapPoints):

        lfMags = wide_mags[:, -nOverlapPoints:]
        hfMags = next_mags[:, :nOverlapPoints]

        hfWeights = np.linspace(0, 1, num=nOverlapPoints, endpoint=False)
        lfWeights = 1 - hfWeights

        mid_mags = lfWeights * lfMags + hfWeights * hfMags

        wide_mags = np.hstack((wide_mags[:,:-nOverlapPoints], mid_mags, next_mags[:, nOverlapPoints:]))

        return wide_mags

    def makeWidePowerSweep(self):
        for r in range(self.num_res-1):#range(100):#
            # dprint(r)
            if self.widesweep[-1] < self.freqs[r+1,0] - self.step:
                trans = np.arange(self.widesweep[-1], self.freqs[r+1,0], self.step)

                self.widesweep = np.hstack((self.widesweep,
                                       trans,
                                       self.freqs[r+1]))
                self.wide_mags = np.hstack((self.wide_mags,
                                    create_ranges(self.wide_mags[:,-1],self.mags[r+1, :, 0], len(trans)),
                                       self.mags[r+1]))

            elif self.widesweep[-1] == self.freqs[r+1,0] or self.widesweep[-1] == self.freqs[r+1,0] - self.step:
                self.widesweep = np.hstack((self.widesweep,
                                       self.freqs[r+1]))
                self.wide_mags = np.hstack((self.wide_mags,
                                       self.mags[r+1]))

            else:
                nOverlapPoints = np.where(self.freqs[r+1] == self.widesweep[-1])[0][0]
                self.widesweep = np.hstack((self.widesweep, self.freqs[r+1][nOverlapPoints:]))
                self.wide_mags = self.stitch(self.wide_mags, self.mags[r+1], nOverlapPoints)

        self.wide_magsdb = 20 * np.log10(self.wide_mags)

    def plotPeaks(self, scores=[None]):
        plt.figure()
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)#, sharex=ax1)
        ax1b = ax1.twiny()

        for a in range(self.num_atten):
            ax1.plot(self.widesweep, self.wide_magsdb[a])



        # ax1b.plot(self.widesweep, np.ones((len(self.widesweep))))
        # ax1b.cla

        ax2.imshow(self.peak_scores, aspect='auto')  # extent=[self.widesweep[-1],self.widesweep[0], 31,0]

        # print(scores[0], scores[0] != None)

        if scores[0] != None:
            for score, ind in zip(scores, self.peakIndices):
                ax1.text(self.widesweep[ind], 90, '%.2f' % score)
                if score >= 0.3:
                    ax1.axvline(self.widesweep[ind], linestyle='--', color='g')
                else:
                    ax1.axvline(self.widesweep[ind], linestyle='--', color='r')
        else:
            for ind in self.peakIndices:
                ax1.axvline(self.widesweep[ind], linestyle='--')

        plt.show()

    def findPeaks(self, ideal_power=[4,10], plot_sweep=False):
        bpFilt = signal.firwin(1000, (0.7 * 0.005 * 12.5 / 7.6, 0.175), pass_zero=False, window=('chebwin', 1000))

        self.peak_scores = np.zeros_like((self.wide_mags))

        for a in range(self.num_atten):
            firFiltMagsDB = np.convolve(self.wide_mags[a], bpFilt, mode='same')
            self.wide_magsdb = 20 * np.log10(self.wide_mags)

            sigThresh=0.5
            thresh = sigThresh * np.std(firFiltMagsDB)
            print('threshold', thresh)
            peaks, _ = signal.find_peaks(-firFiltMagsDB, prominence=thresh, width=2, wlen=int(267 / self.freqStep))
            morePeaks, _ = signal.find_peaks(-firFiltMagsDB, thresh, prominence=0.4 * thresh, width=2, wlen=int(229 / self.freqStep))

            peaks = np.append(peaks, morePeaks)
            peaks = np.unique(peaks)
            peaks = np.sort(peaks)
            print('sp signal found', len(peaks), 'peaks')
            if plot_sweep:
                plt.figure()
                plt.plot(self.widesweep, np.log10(firFiltMagsDB), label = 'fir cheby window')
                plt.plot(self.widesweep, np.log10(self.wide_magsdb[10]), label='raw data')
                plt.plot(self.widesweep[peaks], np.log10(firFiltMagsDB[peaks]), '.', label = 'signal peaks')
                plt.legend()
                plt.show()
            # dprint(len(peaks))

            self.peak_scores[a, peaks] = 1

        if type(ideal_power) == int:
            ideal_power = [ideal_power,ideal_power+1]

        self.deep_mags = np.sum(self.wide_mags[ideal_power[0]:ideal_power[1]], axis=0)
        firFiltMagsDB = np.convolve(self.deep_mags, bpFilt, mode='same')
        self.wide_magsdb = 20 * np.log10(self.wide_mags)

        thresh = sigThresh * np.std(firFiltMagsDB)
        print('threshold', thresh)
        peaks, _ = signal.find_peaks(-firFiltMagsDB, prominence=thresh, width=2, wlen=int(267 / self.freqStep))
        morePeaks, _ = signal.find_peaks(-firFiltMagsDB, thresh, prominence=0.4 * thresh, width=2,
                                         wlen=int(229 / self.freqStep))

        peaks = np.append(peaks, morePeaks)
        peaks = np.unique(peaks)
        peaks = np.sort(peaks)
        if plot_sweep:
            plt.figure()
            plt.plot(self.widesweep, np.log10(firFiltMagsDB), label = 'fir cheby window')
            plt.plot(self.widesweep, np.log10(self.deep_mags), label='raw data')
            plt.plot(self.widesweep[peaks], np.log10(firFiltMagsDB[peaks]), '.', label = 'signal peaks')
            plt.legend()
            plt.show()

        # self.deep_peak_scores = np.sum(self.peak_scores[-ideal_power:], axis=0)
        # self.goodPeakIndices = np.where(self.av_peak_scores >= 1)[0]

        # self.deep_peak_scores = np.zeros_like((self.wide_mags[0]))
        # self.deep_peak_scores[peaks] = 1
        self.peakIndices = peaks
        # else:
        #     self.peakIndices = np.where(self.peak_scores[-1] == 1)[0]



    def determinePeakGoodness(self, noisebuff = 4, running_av= 3, relative_score_scale=False, plot_res=False):
        good_ind = 4
        self.scores = np.zeros((len(self.peakIndices)))
        for r, orig_peakIndx in enumerate(self.peakIndices):
            score = 0
            nextPeakIndx = orig_peakIndx
            peakIndices = np.ones((self.num_atten))*np.nan
            peakIndices[0] = orig_peakIndx
            run_av = orig_peakIndx
            for a in range(self.num_atten-2,0 ,-1):

                lfbuff = np.int_((orig_peakIndx - nextPeakIndx)*1.5)
                lfbuff = lfbuff.clip(0)
                if a<self.num_atten-running_av:
                    if not np.isnan(np.nanmean(peakIndices[a:a+running_av])):
                        run_av = np.nanmean(peakIndices[a:a+running_av])
                else:
                    if not np.isnan(np.nanmean(peakIndices[-a:-1])):
                        run_av = np.nanmean(peakIndices[-a:-1])

                run_av = int(run_av)
                rec_field = self.peak_scores[a, run_av-noisebuff - lfbuff : run_av+noisebuff]
                # print((a, orig_peakIndx, nextPeakIndx, run_av, peakIndices[a:a+3], lfbuff, rec_field))
                try:
                    nextPeakIndx = np.where(rec_field == 1)[0][-1] - noisebuff - lfbuff+ nextPeakIndx
                    peakIndices[a] = nextPeakIndx
                    # print(np.where(rec_field == 1)[0][-1], - noisebuff, -lfbuff, orig_peakIndx, nextPeakIndx)
                    score += 1
                except IndexError:
                    pass

            self.scores[r] = score
            # print(score)
            if plot_res:
                plt.imshow(self.peak_scores[:, (orig_peakIndx - noisebuff) - 50: orig_peakIndx + noisebuff])
                plot_left()

        if relative_score_scale:
            final_pow = np.where(np.isnan(self.peakIndices) == False)[0][-1]
            self.scores /= final_pow
        else:
            self.scores /= self.num_atten


    def removeBad(self, score_cut = 0.3):
        self.ws_good_inds = self.peakIndices[self.scores >= score_cut]
        self.ws_bad_inds = self.peakIndices[self.scores < score_cut]
        print(self.peakIndices[:10], self.scores, self.scores >= score_cut)
        self.mlfreq = self.widesweep[self.peakIndices]#[self.scores >= score_cut]]
        self.score_cut = score_cut
        # self.peakIndices = self.peakIndices[self.scores >= score_cut]
        # self.scores = self.scores[self.scores >= score_cut]

    def saveInferenceFile(self):
        metadatafile = self.npzFile.split('.')[0] + '_metadata.txt'
        print(metadatafile)
        try:
            flNum = int(re.search('fl\d', self.h5FileName, re.IGNORECASE).group()[-1])
        except AttributeError:
            getLogger(__name__).warning('Could not guess feedline from filename.')
            flNum = 0

        print(self.ws_good_inds)
        freqs = self.widesweep[self.peakIndices]
        resIds = np.arange(len(self.peakIndices)) + flNum * 10000
        print(resIds)

        flag = np.full(len(self.peakIndices), sweepdata.ISBAD)
        print(flag.shape, self.ws_good_inds.shape)
        flag[self.scores >= self.score_cut] = sweepdata.ISGOOD
        smd = sweepdata.SweepMetadata(resid=resIds, flag=flag, wsfreq=freqs, file=metadatafile,
                                      mlfreq = self.mlfreq, atten = None,
                                      ml_isgood_score =None, ml_isbad_score=None)
        smd.save()

    # def findLocalMinima(self):
    #     if self.peakIndices is None:
    #         raise Exception('Infer peak locations first!')
    #     foundMinima = np.zeros(len(self.peakIndices))
    #     # print (len(foundMinima))
    #     peakVals = self.wide_magsdb
    #     while np.any(foundMinima == 0):
    #         peakValsRight = np.roll(peakVals, -1)
    #         peakValsLeft = np.roll(peakVals, 1)
    #         peakValsRightLess = np.less_equal(peakVals[self.peakIndices], peakValsRight[self.peakIndices])
    #         peakValsLeftLess = np.less_equal(peakVals[self.peakIndices], peakValsLeft[self.peakIndices])
    #         foundMinima = np.logical_and(peakValsLeftLess, peakValsRightLess)
    #
    #         peakValsRightGreater = np.logical_not(peakValsRightLess)
    #         peakValsLeftGreater = np.logical_and(peakValsRightLess,
    #                                              np.logical_not(foundMinima))  # not greater, but not a minimum
    #         peakValsRightGreaterInd = np.where(peakValsRightGreater)[0]
    #         peakValsLeftGreaterInd = np.where(peakValsLeftGreater)[0]
    #
    #         self.peakIndices[peakValsRightGreaterInd] += 1
    #         self.peakIndices[peakValsLeftGreaterInd] -= 1
    #         # print sum(foundMinima)
    #
    def markCollisions(self, resBWkHz=500):
        if self.peakIndices is None:
            raise Exception('Infer peak locations first!')
        minResSpacing = resBWkHz/self.step #resonators must be separated by this number of points
        peakSpacing = np.diff(self.peakIndices)
        collisionMask = peakSpacing<minResSpacing
        collisionInds = np.where(collisionMask)[0] #locations in peakIndices where there are collisions
        goodPeakInds = np.where(np.logical_not(collisionMask))[0]
        self.badPeakIndices = self.peakIndices[collisionInds]
        self.goodPeakIndices = self.peakIndices[goodPeakInds]
        dprint((len(self.badPeakIndices), len(self.goodPeakIndices)))

if __name__ == '__main__':
    mdd = os.environ['MKID_DATA_DIR']

    parser = argparse.ArgumentParser(description='WS Auto Peak Finding')
    parser.add_argument('wsPowerFile', nargs=1, help='Widesweep power data')
    # parser.add_argument('-d', '--digital', action='store_true', help='Perform preprocessing step for digital data')
    # parser.add_argument('-s', '--sigma', dest='sigma', type=float, default=.5, help='Peak inference threshold')
    # parser.add_argument('use_autowidesweep', help='Bool')
    args = parser.parse_args()
    #
    wsFile = args.wsPowerFile[0]
    if not os.path.isfile(wsFile):
        wsFile = os.path.join(mdd, wsFile)
    #
    # use_autowidesweep = args[3]
    #
    # if use_autowidesweep:
    #     ANALOG_SPACING = 12.5
    #     DIGITAL_SPACING = 7.629
    #     spacing = DIGITAL_SPACING if args.digital else ANALOG_SPACING
    #     wsFilt = Finder(spacing)
    #     wsFilt.inferPeaks(wsFile, isDigital=args.digital, sigThresh=args.sigma)
    #     wsFilt.findLocalMinima()
    #     wsFilt.markCollisions(resBWkHz=200)
    #     getLogger(__name__).info('Found {} good peaks.'.format(len(wsFilt.goodPeakIndices)))
    #     wsFilt.saveInferenceFile()
    #
    # else:
    wps = widePowerSweep(npzFile=wsFile)#'/Users/dodkins/Scratch/MEC/20181212/psData2_222.npz'
    # wps.makeWidePowerSweep()
    # wps.findPeaks(ideal_power=[4,10])
    wps.findPeaks(ideal_power=[2,16])

    wps.determinePeakGoodness()
    wps.removeBad()
    # wps.findLocalMinima()
    wps.markCollisions(resBWkHz=200)
    wps.plotPeaks()
    # wps.plotPeaks(scores=wps.scores)
    # getLogger(__name__).info('Found {} good peaks.'.format(len(wsFilt.goodPeakIndices)))
    wps.saveInferenceFile()



