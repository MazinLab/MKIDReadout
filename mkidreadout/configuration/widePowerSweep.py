import sys
from mkidreadout.configuration.powersweep.PowerSweepML.PSFitMLData import PSFitMLData
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np
from scipy.optimize import curve_fit
import scipy.signal as signal
from mkidcore.corelog import getLogger
import mkidreadout.configuration.sweepdata as sweepdata
import re

# import sys, os
# sys.path.append(os.environ['MEDIS_DIR'])
# from Utils.misc import dprint

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
    def __init__(self, h5File=None, PSFile=None, useAllAttens=True, useResID=False):
        self.h5FileName = '/Users/dodkins/Scratch/MEC/20180822/mlTrainingData/ps_r220_FL_6_a_20180618-160242.h5'
        PSFileName = '/Users/dodkins/Scratch/MEC/20180822/mlTrainingData/ps_freq_FL6a.txt'
        inferenceData = PSFitMLData(h5File=self.h5FileName, PSFile=PSFileName, useAllAttens=True, useResID=False)
        inferenceData.loadTrainData()

        inferenceData.mags = np.sqrt(inferenceData.Is ** 2 + inferenceData.Qs ** 2)
        inferenceData.magsdb = 20 * np.log10(inferenceData.mags)

        iq_vels = inferenceData.iq_vels
        self.mags = inferenceData.mags
        self.freqs = inferenceData.freqs

        freq_range = (self.freqs[0] - self.freqs[0,0])/1e3

        self.num_res = iq_vels.shape[0]
        self.num_atten = iq_vels.shape[1]
        self.num_freqs = iq_vels.shape[2]

        self.widesweep = inferenceData.freqs[0]
        self.wide_mags = self.mags[0]

        self.step = self.freqs[0,2] - self.freqs[0,1]

        print(inferenceData.mags.shape)
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
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2, sharex=ax1)

        for a in range(self.num_atten):
            ax1.plot(self.wide_magsdb[a])

        ax2.imshow(self.peak_scores, aspect='auto')  # extent=[self.widesweep[-1],self.widesweep[0], 31,0]
        for ind in self.goodPeakIndices:
            ax1.axvline(ind, linestyle='--')

        # print(scores[0], scores[0] != None)

        if scores[0] != None:
            for score, ind in zip(scores, self.goodPeakIndices):
                ax1.text(ind, 90, '%.2f' % score)
        plt.show()

    def findPeaks(self, ideal_power=8, plot_sweep=False):
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
                plt.plot(self.widesweep, firFiltMagsDB, label = 'fir cheby window')
                plt.plot(self.widesweep, self.wide_magsdb[10], label='raw data')
                plt.plot(self.widesweep[peaks], firFiltMagsDB[peaks], '.', label = 'signal peaks')
                plt.legend()
                plt.show()
            # dprint(len(peaks))

            self.peak_scores[a, peaks] = 1

        if ideal_power != 0:
            self.av_peak_scores = np.sum(self.peak_scores[-ideal_power:], axis=0)
            self.goodPeakIndices = np.where(self.av_peak_scores >= 1)[0]
        else:
            self.goodPeakIndices = np.where(self.peak_scores[-1] == 1)[0]



    def determinePeakGoodness(self, noisebuff = 4, running_av= 3, relative_score_scale=False, plot_res=False):

        self.scores = np.zeros((len(self.goodPeakIndices)))
        for r, orig_peakIndx in enumerate(self.goodPeakIndices):
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
                    # thisPeakIndx = nextPeakIndx
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
            final_pow = np.where(np.isnan(self.goodPeakIndices) == False)[0][-1]
            self.scores /= final_pow
        else:
            self.scores /= self.num_atten


    def removeBad(self, score_cut = 0.3):
        self.ws_good_inds = self.goodPeakIndices[self.scores >= score_cut]
        self.ws_bad_inds = self.goodPeakIndices[self.scores < score_cut]
        print(self.goodPeakIndices[:10], self.scores, self.scores >= score_cut)
        self.goodPeakIndices = self.goodPeakIndices[self.scores >= score_cut]
        self.scores = self.scores[self.scores >= score_cut]

    def saveInferenceFile(self):
        metadatafile = self.h5FileName.split('.')[0] + '_metadata.txt'
        print(metadatafile)
        try:
            flNum = int(re.search('fl\d', self.h5FileName, re.IGNORECASE).group()[-1])
        except AttributeError:
            getLogger(__name__).warning('Could not guess feedline from filename.')
            flNum = 0

        print(self.ws_good_inds)
        resIds = np.arange(self.widesweep.size) + flNum * 10000
        print(resIds)

        flag = np.full(self.widesweep.size, sweepdata.ISBAD)
        print(flag, flag.shape, self.ws_good_inds.shape)
        flag[self.ws_good_inds] = sweepdata.ISGOOD
        smd = sweepdata.SweepMetadata(resid=resIds, flag=flag, wsfreq=self.widesweep, file=metadatafile)
        smd.save()

    def plot_array_stats(self):
        # Argmax IQV histogram
        hist_2d = np.zeros((num_freqs, self.num_atten))
        for r in range(num_res):
            # plt.plot(np.argmax(iq_vels[r], axis=1))
            # plt.ylabel('KHz')
            # # plt.show()
            # plot_left()
            # print(np.argmax(iq_vels[r], axis=1),range(self.num_atten))
            hist_2d[np.argmax(iq_vels[r], axis=1), range(self.num_atten)] += 1
            # plt.imshow(hist_2d, aspect=0.05, extent=[0,31,1000,0])
            # plt.show()


        all_moves = np.zeros((num_res,self.num_atten))


        disp = np.zeros(num_res)
        disp_freq = np.zeros(num_res)
        for r in range(num_res):
            opt_iFreq = np.where(freqs[r] == inferenceData.opt_freqs[r])[0][0]
            print(opt_iFreq, num_freqs//2, opt_iFreq-num_freqs//2)
            disp[r] = opt_iFreq-num_freqs//2
            disp_freq[r] = (freqs[r, len(freqs[r])//2] - inferenceData.opt_freqs[r])/1e3


        plt.plot(disp)
        plot_left()

        plt.plot(disp_freq)
        plt.ylabel('Freq displacement (KKz)')
        plt.xlabel('Num res')
        plot_left()

        plt.hist(disp_freq, bins=25)
        plt.xlabel('Freq displacement (KKz)')
        plot_left()
        #
        # Investigating res IQV
        for r in range(num_res):
            cax = plt.imshow(iq_vels[r], aspect=2.5, norm=LogNorm())# vmax=250)
            cb = plt.colorbar(cax)
            plt.xlabel('Freq index (KHz)')
            plt.ylabel('Atten index (dB)')
            cb.ax.set_title('IQV')
            # print(inferenceData.opt_freqs[r], freqs[r])
            opt_iFreq = np.where(freqs[r] == inferenceData.opt_freqs[r])[0][0]
            # print(opt_iFreq,inferenceData.opt_iAttens[r])
            plt.plot(opt_iFreq,inferenceData.opt_iAttens[r], 'ko')
            plt.axvline(opt_iFreq, linestyle = '--', color='k')
            plt.axhline(inferenceData.opt_iAttens[r], linestyle = '--', color='k')
            plot_left()


        #
        #
        #     plot_left()
        # #
        # for r in range(num_res):
        #     all_moves[r] = freq_range[np.argmax(iq_vels[r], axis=1)]
        #     print(np.argmax(iq_vels[r], axis=1))
        #     hist_2d[r, np.argmax(iq_vels[r], axis=1)]+=1
        # # plt.imshow(all_moves, aspect=0.01)
        # # plt.show()

        # cax = plt.imshow(hist_2d, aspect=0.05,norm=LogNorm(), extent=[0,31,1000,0])
        # cb = plt.colorbar(cax)
        # plt.ylabel('Freq index (KHz)')
        # plt.xlabel('Atten index (dB)')
        # cb.ax.set_title('Num Res')
        # plot_left()



        av_freqs = np.zeros((self.num_atten))

        for a in range(1,self.num_atten):
            # print(a)
            tot = sum(hist_2d[:,-a])
            # plt.plot(range(num_freqs), hist_2d[:,-a])#/tot)

            hist_2d[:, -a] = smooth(hist_2d[:,-a])
            # plt.plot(range(num_freqs), hist_2d[:,-a])
            # plot_left()
            # try:
            popt, pcov = curve_fit(gaussian, np.arange(num_freqs), hist_2d[:,-a]/tot)
            # except RuntimeError:
            #     print('maxfev')
            # popt = [10,100]

            # plt.plot(gaussian(np.arange(num_freqs), 10, 100))
            # plot_left()
            # print(popt)
            av_freqs[a-1] = popt[1]
            # plt.plot(range(num_freqs), gaussian(np.arange(num_freqs), *popt)*tot)
            # plot_left()

        lim = 23
        plt.plot(av_freqs[:lim])

        tot = sum(av_freqs[:lim])
        #
        popt, pcov = curve_fit(poly, np.arange(num_atten)[:lim], av_freqs[:lim]/tot)
        # print(popt)
        plt.plot(range(num_atten)[:lim], poly(np.arange(num_atten)[:lim], *popt)*tot)
        trans = np.int_(num_freqs//2 - np.ceil(poly(np.arange(num_atten), *popt)*tot))
        print(trans, len(trans), num_atten)
        iq_vels_crop = np.zeros_like(iq_vels[:,:,:40])



        for r in range(num_res):
            for a in range(num_atten):
                print(r,a, trans[a] + (num_freqs//2) - 20, trans[a] + (num_freqs//2) + 20, trans[a], (num_freqs//2))
                print(iq_vels[r, a, (trans[a] + (num_freqs//2) - 20):(trans[a] + (num_freqs//2) + 20)].shape)
                print((trans[a] + (num_freqs//2) + 20) - (trans[a] + (num_freqs//2) - 20))
                iq_vels_crop[r, a] = iq_vels[r, a, (trans[a] + (num_freqs//2) - 20):(trans[a] + (num_freqs//2) + 20)]

        # popt, pcov = curve_fit(expon, np.arange(num_atten)[:lim], av_freqs[:lim]/tot)
        # print(popt)
        # plt.plot(range(num_atten)[:lim], expon(np.arange(num_atten)[:lim], *popt)*tot)
        # print(expon(np.arange(num_atten), *popt)*tot)
        plot_left()


        # plt.show()

if __name__ == '__main__':
    wps = widePowerSweep()
    wps.makeWidePowerSweep()
    wps.findPeaks(ideal_power=0)
    # wps.plotPeaks()
    wps.determinePeakGoodness()
    print(wps.scores,)
    wps.removeBad()
    # wps.plotPeaks(scores=wps.scores)

    wps.saveInferenceFile()



