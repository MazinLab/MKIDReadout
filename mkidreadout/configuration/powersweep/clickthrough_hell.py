#!/usr/bin/env python
# clickthrough hell
from __future__ import print_function
from numpy import *
import numpy
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from mkidreadout.utils.iqsweep import *
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
from mkidreadout.configuration.powersweep.autopeak import Finder
import mkidreadout.instruments as instruments

from pkg_resources import resource_filename

import mkidreadout.configuration.powersweep.gui as gui
import mkidreadout.configuration.sweepdata as sweepdata
from mkidreadout.configuration.powersweep import psmldata
from mkidcore.corelog import getLogger, create_log
import argparse
import scipy.integrate


class StartQt4(QMainWindow):
    def __init__(self, psfile, psmetafile, Ui, parent=None, startndx=0, goodcut=np.inf, badcut=-np.inf, useml=False):
        QWidget.__init__(self, parent)

        self.ui = Ui()
        self.ui.setupUi(self)

        self.atten = -1
        self.ui.atten.setValue(self.atten)
        self.resnum = startndx
        self.indx = 0

        self.badcut, self.goodcut = badcut, goodcut
        self.useml = useml

        QObject.connect(self.ui.atten, SIGNAL("valueChanged(int)"), self.setnewatten)
        QObject.connect(self.ui.savevalues, SIGNAL("clicked()"), self.savevalues)
        QObject.connect(self.ui.jumptores, SIGNAL("clicked()"), self.jumptores)

        QShortcut(QKeySequence(Qt.Key_Space), self, self.savevalues)
        QShortcut(QKeySequence('B'), self, self.goback)

        self.metadata = mdata = sweepdata.SweepMetadata(file=psmetafile)
        self.metadata_out = sweepdata.SweepMetadata(file=psmetafile)
        self.metadata_out.file = os.path.splitext(psmetafile)[0] + '_out.txt'
        try:
            self.metadata_out._load()
        except IOError:
            pass

        self.ui.save_filename.setText(self.metadata_out.file)

        self.fsweepdata = None
        self.mlResIDs = None
        self.mlFreqs = None
        self.mlAttens = None

        self.widesweep_goodFreqs = mdata.wsfreq[mdata.flag & sweepdata.ISGOOD]
        self.widesweep_allResIDs = mdata.resIDs
        self.widesweep_allFreqs = mdata.wsfreq

        self.widesweep = None

        self.navi_toolbar = NavigationToolbar(self.ui.plot_3.canvas, self)
        self.ui.plot_3.canvas.setFocusPolicy(Qt.ClickFocus)
        cid = self.ui.plot_3.canvas.mpl_connect('key_press_event', self.zoom_plot_3)

        self.openfile = psfile
        self.ui.open_filename.setText(str(self.openfile))
        self.loadps()

    # def open_dialog(self):
    #     self.openfile = QFileDialog.getOpenFileName(parent=None, caption=QString(str("Choose PS File")),
    #                                                 directory=".", filter=QString(str("H5 (*.h5)")))
    #     self.ui.open_filename.setText(str(self.openfile))
    #     self.loadps()

    def goback(self):
        if self.resnum == 0:
            return
        self.resnum -= 1
        self.atten = self.attenList[self.resnum]
        self.loadres()

    def loadres(self):
        self.Res1 = IQsweep()
        self.Res1.loadpowers_from_freqsweep(self.fsweepdata, self.resnum)

        self.resfreq = self.fsweepdata.opt_freqs[self.resnum]

        if not (self.Res1.freq[0] < self.resfreq < self.Res1.freq[-1]):
            raise RuntimeError('Out of order resfreq flag={}'.format(self.fsweepdata.mdResMask[
                                                                         self.fsweepdata.resIDs == self.Res1.resID]))

        self.resID = self.Res1.resID
        self.NAttens = self.Res1.atten1s.size

        self.ui.res_num.setText(str(self.resnum))
        self.ui.jumptonum.setValue(self.resnum)
        self.ui.frequency.setText(str(self.resfreq / 1e9))

        getLogger(__name__).info("Res: {} --> ID: {}".format(self.resnum, self.resID))

        self.res1_iq_vels = np.zeros((self.NAttens, self.Res1.fsteps - 1))
        self.res1_iq_amps = np.zeros((self.NAttens, self.Res1.fsteps))

        foo = np.sqrt(np.diff(self.Res1.Qs, axis=1) ** 2 + np.diff(self.Res1.Is, axis=1) ** 2)
        self.res1_iq_vels[:, 1:] = foo[:, :-1]
        self.res1_iq_amps = np.sqrt(self.Res1.Qs ** 2 + self.Res1.Is ** 2)

        # Sort the IQ velocities for each attenuation, to pick out the maximums
        sorted_vels = numpy.sort(self.res1_iq_vels, axis=1)
        # Last column is maximum values for each atten (row)
        self.res1_max_vels = sorted_vels[:, -1]
        # Second to last column has second highest value
        self.res1_max2_vels = sorted_vels[:, -2]
        # Also get indices for maximum of each atten, and second highest
        sort_indices = numpy.argsort(self.res1_iq_vels, axis=1)
        max_indices = sort_indices[:, -1]
        max2_indices = sort_indices[:, -2]
        max_neighbor = max_indices.copy()

        # for each attenuation find the ratio of the maximum velocity to the second highest velocity
        self.res1_max_ratio = self.res1_max_vels.copy()
        max_neighbors = zeros(self.NAttens)
        max2_neighbors = zeros(self.NAttens)
        self.res1_max2_ratio = self.res1_max2_vels.copy()
        for iAtt in range(self.NAttens):
            if max_indices[iAtt] == 0:
                max_neighbor = self.res1_iq_vels[iAtt, max_indices[iAtt] + 1]
            elif max_indices[iAtt] == len(self.res1_iq_vels[iAtt, :]) - 1:
                max_neighbor = self.res1_iq_vels[iAtt, max_indices[iAtt] - 1]
            else:
                max_neighbor = maximum(self.res1_iq_vels[iAtt, max_indices[iAtt] - 1],
                                       self.res1_iq_vels[iAtt, max_indices[iAtt] + 1])
            max_neighbors[iAtt] = max_neighbor
            self.res1_max_ratio[iAtt] = self.res1_max_vels[iAtt] / max_neighbor
            if max2_indices[iAtt] == 0:
                max2_neighbor = self.res1_iq_vels[iAtt, max2_indices[iAtt] + 1]
            elif max2_indices[iAtt] == len(self.res1_iq_vels[iAtt, :]) - 1:
                max2_neighbor = self.res1_iq_vels[iAtt, max2_indices[iAtt] - 1]
            else:
                max2_neighbor = maximum(self.res1_iq_vels[iAtt, max2_indices[iAtt] - 1],
                                        self.res1_iq_vels[iAtt, max2_indices[iAtt] + 1])
            max2_neighbors[iAtt] = max2_neighbor
            self.res1_max2_ratio[iAtt] = self.res1_max2_vels[iAtt] / max2_neighbor
        # normalize the new arrays
        self.res1_max_vels /= numpy.max(self.res1_max_vels)
        self.res1_max_vels *= numpy.max(self.res1_max_ratio)
        self.res1_max2_vels /= numpy.max(self.res1_max2_vels)
        # self.res1_relative_max_vels /= numpy.max(self.res1_relative_max_vels)

        if self.useml:
            self.guess_atten(('ML',))
        else:
            self.guess_atten()

    def guess_atten(self, method=('dratio', 'ROT', 'mid', 'ML')):

        guesses = {'mid': self.Res1.atten1s[self.NAttens / 2]}

        max_ratio_threshold = 1.5
        rule_of_thumb_offset = 2

        try:
            dratio = np.diff(self.res1_max_ratio) / np.diff(self.Res1.atten1s)
            nratiosp1 = np.where(dratio >= np.median(dratio))[0]
            nratiosp1 = nratiosp1[nratiosp1 > self.res1_max_vels.argmax()][0]
            guesses['dratio'] = self.Res1.atten1s[:-1][nratiosp1] + 2  # added +2 fudge factor b/c saturating on average
        except IndexError:
            guesses['dratio'] = None

        try:
            guesses['ML'] = self.mlAttens[self.resID == self.mlResIDs][0]
        except IndexError:
            getLogger(__name__).critical('No ML atten for resID{}. Not possible'.format(self.resID))
            guesses['ML'] = None

        # require ROTO adjacent elements to be all below the MRT
        bool_remove = np.ones(self.res1_max_ratio.size)
        for ri in range(self.res1_max_ratio.size - rule_of_thumb_offset - 2):
            bool_remove[ri] = (self.res1_max_ratio[ri:ri + rule_of_thumb_offset + 1] < max_ratio_threshold).all()

        # require the attenuation value to be past the initial peak in MRT
        guess_atten_idx = np.extract(bool_remove, np.arange(self.res1_max_ratio.size))
        guess_atten_idx = guess_atten_idx[guess_atten_idx > self.res1_max_ratio.argmax()]
        if guess_atten_idx.size >= 1:
            guess_atten_idx += rule_of_thumb_offset
            guesses['ROT'] = self.Res1.atten1s[guess_atten_idx.clip(max=self.Res1.atten1s.size)[0]]
        else:
            guesses['ROT'] = None

        for m in method:
            guess = guesses[m]
            if guess is not None:
                break

        getLogger(__name__).info('Select atten using {} method: {}'.format(m, guess))
        self.select_atten(guess)
        self.ui.atten.setValue(round(guess))

    def guess_res_freq(self):
        if self.useml and np.any(self.resID == self.mlResIDs):
            resInd = np.where(self.resID == self.mlResIDs)[0]
            self.select_freq(self.mlFreqs[resInd])
        else:
            use = np.abs(self.Res1.freq[:-1] - self.fsweepdata.initfreqs[self.resnum]) < .25e6
            com_guess = (scipy.integrate.trapz(self.res1_iq_vel[use] * self.Res1.freq[:-1][use]) /
                         scipy.integrate.trapz(self.res1_iq_vel[use]))

            self.select_freq(com_guess)
            getLogger(__name__).info('Guessing resonant freq at {} for self.iAtten={}'.format(com_guess, self.iAtten))

    def loadps(self):
        self.fsweepdata = psmldata.MLData(fsweep=self.openfile, mdata=self.metadata_out)
        self.widesweep = self.fsweepdata.freqSweep.oldwsformat(60, 66)
        getLogger(__name__).info('Loaded ' + self.openfile)

        self.stop_ndx = self.fsweepdata.prioritize_and_cut(self.badcut, self.goodcut, plot=True)

        self.mlResIDs = self.fsweepdata.resIDs
        self.mlFreqs = self.fsweepdata.opt_freqs
        self.mlAttens = self.fsweepdata.opt_attens
        self.freq = self.fsweepdata.opt_freqs

        self.freqList = np.zeros(2000)
        self.attenList = np.full_like(self.freqList, -1)
        self.loadres()

    def on_press(self, event):
        self.select_freq(event.xdata)

    def zoom_plot_3(self, event):
        if str(event.key) == str('z'):
            getLogger(__name__).info('zoooom')
            self.navi_toolbar.zoom()

    def click_plot_1(self, event):
        self.ui.atten.setValue(round(event.xdata))

    def select_freq(self, freq):
        freq = freq[0] if isinstance(freq, np.ndarray) else freq
        if freq is None:
            return

        self.resfreq = freq
        self.ui.frequency.setText(str(self.resfreq / 1e9))
        self.ui.plot_2.canvas.ax.plot(self.Res1.freq[self.indx], self.res1_iq_vel[self.indx], 'bo')
        self.ui.plot_3.canvas.ax.plot(self.Res1.I[self.indx], self.Res1.Q[self.indx], 'bo')

        self.indx = np.abs(self.Res1.freq - self.resfreq).argmin()

        self.ui.plot_2.canvas.ax.plot(self.Res1.freq[self.indx], self.res1_iq_vel[self.indx], 'ro')
        self.ui.plot_2.canvas.draw()
        self.ui.plot_3.canvas.ax.plot(self.Res1.I[self.indx], self.Res1.Q[self.indx], 'ro')
        self.ui.plot_3.canvas.draw()

    def select_atten(self, attenuation):

        if self.atten != -1:
            self.lastatten = self.atten
            self.lastiAtten = np.abs(self.Res1.atten1s - self.atten).argmin()
        else:
            self.lastatten = attenuation
            self.lastiAtten = np.abs(self.Res1.atten1s - attenuation).argmin()

        self.iAtten = np.abs(self.Res1.atten1s - attenuation).argmin()

        self.atten = np.round(self.Res1.atten1s[self.iAtten])
        self.res1_iq_vel = self.res1_iq_vels[self.iAtten, :]
        self.Res1.I = self.Res1.Is[self.iAtten]
        self.Res1.Q = self.Res1.Qs[self.iAtten]
        self.Res1.Icen = self.Res1.Icens[self.iAtten]
        self.Res1.Qcen = self.Res1.Qcens[self.iAtten]
        self.makeplots()
        self.guess_res_freq()

    def makeplots(self):
        try:
            # plot 1
            self.ui.plot_1.canvas.ax.clear()
            # self.ui.plot_1.canvas.ax.plot(self.Res1.atten1s, self.res1_max_vels, 'b.-', label='Max IQ velocity')
            self.ui.plot_1.canvas.ax.plot(self.Res1.atten1s, self.res1_max_ratio, 'k.-', label='Ratio (Max Vel)/(2nd '
                                                                                               'Max Vel)')
            dratio = np.diff(self.res1_max_ratio) / np.diff(self.Res1.atten1s)

            self.ui.plot_1.canvas.ax.plot(self.Res1.atten1s[:-1], dratio, 'g.-', label='D(Ratio)')
            self.ui.plot_1.canvas.ax.legend()

            self.ui.plot_1.canvas.ax.set_ylim(-1, 2.5)
            self.ui.plot_1.canvas.ax.axhline(np.median(dratio) + dratio.std() / 3, linestyle=':', linewidth=.5,
                                             color='grey')
            self.ui.plot_1.canvas.ax.axhline(0, linestyle='-', linewidth=.5, color='black')
            self.ui.plot_1.canvas.ax.axhline(np.median(dratio), linestyle='-', linewidth=.5, color='grey')
            self.ui.plot_1.canvas.ax.axhline(np.median(dratio) - dratio.std() / 3,
                                             linestyle=':', linewidth=.5, color='grey')
            self.ui.plot_1.canvas.ax.set_xlabel('Attenuation')
            self.ui.plot_1.canvas.ax.set_ylabel('IQVel Ratio')

            cid = self.ui.plot_1.canvas.mpl_connect('button_press_event', self.click_plot_1)

            self.ui.plot_1.canvas.ax.plot(self.lastatten, self.res1_max_ratio[self.lastiAtten], 'ko')
            # self.ui.plot_1.canvas.ax.plot(self.lastatten, self.res1_max_vels[self.lastiAtten], 'bo')
            self.ui.plot_1.canvas.ax.plot(self.atten, self.res1_max_ratio[self.iAtten], 'ro')
            # self.ui.plot_1.canvas.ax.plot(self.atten, self.res1_max_vels[self.iAtten], 'ro')
            self.ui.plot_1.canvas.draw()

            # Plot 2
            self.ui.plot_2.canvas.ax.clear()
            self.ui.plot_2.canvas.ax.set_xlabel('Frequency (Hz)')
            self.ui.plot_2.canvas.ax.set_ylabel('IQ velocity')
            self.ui.plot_2.canvas.ax.plot(self.Res1.freq[:-1], self.res1_iq_vel, 'b.-')
            if self.iAtten > 0:
                self.ui.plot_2.canvas.ax.plot(self.Res1.freq[:-1], self.res1_iq_vels[self.iAtten - 1], 'g.-')
                self.ui.plot_2.canvas.ax.lines[-1].set_alpha(.7)
            if self.iAtten > 1:
                self.ui.plot_2.canvas.ax.plot(self.Res1.freq[:-1], self.res1_iq_vels[self.iAtten - 2], 'g.-')
                self.ui.plot_2.canvas.ax.lines[-1].set_alpha(.3)

            self.ui.plot_2.canvas.ax.set_xlim(self.resfreq - 1e6, self.resfreq + 1e6)
            cid = self.ui.plot_2.canvas.mpl_connect('button_press_event', self.on_press)

            # test integration for center
            # import scipy.integrate
            # use = np.abs(self.Res1.freq[:-1]-self.fsweepdata.initfreqs[self.resnum])<.25e6
            # foo=scipy.integrate.trapz(self.res1_iq_vel[use]*self.Res1.freq[:-1][use],
            #                           dx=self.Res1.freq[1]-self.Res1.freq[0])
            # foo/=scipy.integrate.trapz(self.res1_iq_vel[use], dx=self.Res1.freq[1]-self.Res1.freq[0])
            # self.ui.plot_2.canvas.ax.axvline(foo,c='orange')
            self.ui.plot_2.canvas.ax.axvline(self.fsweepdata.initfreqs[self.resnum] - .25e6, c='orange', linestyle=':')
            self.ui.plot_2.canvas.ax.axvline(self.fsweepdata.initfreqs[self.resnum] + .25e6, c='orange', linestyle=':')

            # self.widesweep=None
            freq_start = self.Res1.freq[0]
            freq_stop = self.Res1.freq[-1]
            if self.widesweep is not None:
                wsmask = (self.widesweep[:, 0] >= freq_start) & (self.widesweep[:, 0] <= freq_stop)
                iqVel_med = np.median(self.res1_iq_vel)
                ws_amp = (self.widesweep[wsmask, 1] ** 2. + self.widesweep[wsmask, 2] ** 2.) ** 0.5

                ws_amp_med = np.median(ws_amp)
                ws_amp *= 1.0 * iqVel_med / ws_amp_med

                self.ui.plot_2.canvas.ax.plot(self.widesweep[wsmask, 0], ws_amp, 'k.-')
                self.ui.plot_2.canvas.ax.lines[-1].set_alpha(.5)

            ws_allFreqs = self.widesweep_allFreqs[(self.widesweep_allFreqs >= freq_start) &
                                                  (self.widesweep_allFreqs <= freq_stop)]
            ws_allResIDs = self.widesweep_allResIDs[(self.widesweep_allFreqs >= freq_start) &
                                                    (self.widesweep_allFreqs <= freq_stop)]

            ws_ymax = self.ui.plot_2.canvas.ax.yaxis.get_data_interval()[1]
            for ws_resID_i, ws_freq_i in zip(ws_allResIDs, ws_allFreqs):
                ws_color = 'r' if ws_freq_i in self.widesweep_goodFreqs else 'k'
                self.ui.plot_2.canvas.ax.axvline(ws_freq_i, c=ws_color, alpha=.5)
                self.ui.plot_2.canvas.ax.text(x=ws_freq_i, y=ws_ymax, s=str(int(ws_resID_i)), color=ws_color, alpha=.5)

            self.ui.plot_2.canvas.ax.axvline(self.fsweepdata.initfreqs[self.resnum], c='r', linewidth=1.5)
            self.ui.plot_2.canvas.ax.axvline(self.fsweepdata.opt_freqs[self.resnum], linestyle=':', c='g',
                                             linewidth=1.5)
            self.ui.plot_2.canvas.ax.set_xlim(self.fsweepdata.initfreqs[self.resnum] - 1e6,
                                              self.fsweepdata.initfreqs[self.resnum] + 1e6)
            self.ui.plot_2.canvas.draw()

            # Plot 3
            self.ui.plot_3.canvas.ax.clear()

            use = ((self.Res1.freq > self.fsweepdata.initfreqs[self.resnum] - 1e6) &
                   (self.Res1.freq < self.fsweepdata.initfreqs[self.resnum] + 1e6))
            if self.iAtten > 0:
                self.ui.plot_3.canvas.ax.plot(self.Res1.Is[self.iAtten - 1][use], self.Res1.Qs[self.iAtten - 1][use],
                                              'g.-')
                self.ui.plot_3.canvas.ax.lines[0].set_alpha(.6)
            if self.iAtten > 1:
                self.ui.plot_3.canvas.ax.plot(self.Res1.Is[self.iAtten - 2][use], self.Res1.Qs[self.iAtten - 2][use],
                                              'g.-')
                self.ui.plot_3.canvas.ax.lines[-1].set_alpha(.3)
            self.ui.plot_3.canvas.ax.plot(self.Res1.I[use], self.Res1.Q[use], '.-')

            self.ui.plot_3.canvas.draw()

        except IndexError:
            getLogger(__name__).info("reached end of resonator list, closing GUI")
            sys.exit()

    def jumptores(self):
        try:
            self.atten = -1
            self.resnum = self.ui.jumptonum.value()
            self.loadres()
        except IndexError:
            getLogger(__name__).info("Res value out of bounds.")
            self.ui.plot_1.canvas.ax.clear()
            self.ui.plot_2.canvas.ax.clear()
            self.ui.plot_3.canvas.ax.clear()
            self.ui.plot_1.canvas.draw()
            self.ui.plot_2.canvas.draw()
            self.ui.plot_3.canvas.draw()

    def setnewatten(self):
        self.select_atten(self.ui.atten.value())

    def savevalues(self):
        self.freqList[self.resnum] = self.resfreq
        self.attenList[self.resnum] = self.atten
        self.metadata_out.set(self.resID, atten=self.atten, freq=self.resfreq,
                              reviewed=True, save=True)

        msg = " ....... Saved to file:  resnum={} resID={} resfreq={} atten={}"
        getLogger(__name__).info(msg.format(self.resnum, self.resID, self.resfreq, self.atten))
        self.resnum += 1
        self.atten = -1

        if self.resnum >= self.stop_ndx:
            getLogger(__name__).info("reached end of resonator list, closing GUI")
            sys.exit()
        self.loadres()


if __name__ == "__main__":

    import mkidreadout.configuration.powersweep.ml.findPowers as findpowers
    from mkidcore.readdict import ReadDict
    from mkidreadout.configuration.powersweep.psmldata import MLData

    create_log('__main__', console=True, mpsafe=True, fmt='%(asctime)s PSClickthrough: %(levelname)s %(message)s')
    create_log('mkidreadout', console=True, mpsafe=True, fmt='%(asctime)s mkidreadout: %(levelname)s %(message)s')
    create_log('mkidcore', console=True, mpsafe=True, fmt='%(asctime)s mkidcore: %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description='MKID Powersweep GUI')
    parser.add_argument('psweep', type=str, help='A sweep npz file.')
    parser.add_argument('--meta', dest='metafile', default='', type=str, help='The matching metadata.txt file to load')

    subparsers = parser.add_subparsers(help='find->[applyml]->click', dest='mode')
    find_parser = subparsers.add_parser('find', help='Find resonators')
    find_parser.add_argument('psweep2', type=str, default='', help='The other sweep file for the FL (both needed).')
    find_parser.add_argument('-s', '--sigma', dest='sigma', type=float, default=.5, help='Peak inference threshold')

    ml_parser = subparsers.add_parser('applyml', help='Apply a ML model')
    ml_parser.add_argument('mlconfig', help='Machine learning model config file')
    ml_parser.add_argument('model', help='Machine learning model (good)')
    ml_parser.add_argument('--bmodel', dest='bmodel', default='', help='A bad model file')

    click_parser = subparsers.add_parser('click', help='Clickthrough the resonators')
    click_parser.add_argument('--small', action='store_true', dest='smallui', default=False, help='Use small GUI')
    click_parser.add_argument('--use-ml', action='store_true', dest='useml', default=False,
                              help='Use ML as initial guess')
    click_parser.add_argument('-i', dest='start_ndx', type=int, default=0, help='Starting resonator index')
    click_parser.add_argument('-gc', dest='gcut', default=1, type=float, help='Assume good if net ML score > (EXACT)')
    click_parser.add_argument('-bc', dest='bcut', default=-1, type=float, help='Assume bad if net ML score < (EXACT)')

    args = parser.parse_args()
    metafile = args.metafile if args.metafile else os.path.splitext(args.psweep)[0] + '_metadata.txt'

    if args.mode == 'find':
        finder = Finder([args.psweep, args.psweep2])
        finder.inferPeaks(sigThresh=args.sigma)
        finder.findLocalMinima()
        finder.markCollisions(resBWkHz=200)
        getLogger(__name__).info('Found {} for clickthrough peaks.'.format(finder.num_good))
        smd = finder.getSweepMetadata(instruments.guessFeedline(args.psweep))
        smd.save(file=metafile)
    elif args.mode == 'applyml':
        smd = sweepdata.SweepMetadata(metafile)
        mlDict = ReadDict(file=args.mlconfig)
        inferenceData = MLData(args.psweep, smd)
        mlArgs = dict(xWidth=mlDict['xWidth'], resWidth=mlDict['resWidth'], pad_res_win=mlDict['padResWin'],
                      useIQV=mlDict['useIQV'], useMag=mlDict['useMag'], mlDictnAttens=mlDict['nAttens'])
        findpowers.apply_ml_model(inferenceData, smd.wsatten, mlDict['nAttens'], mlArgs=mlArgs,
                                  goodModel=args.model, badModel=args.bmodel, center_loop=mlDict['centerLoop'])
        inferenceData.updatemetadata()
        smd.save()
    else:
        Ui = gui.Ui_MainWindow_Small if args.smallui else gui.Ui_MainWindow
        app = QApplication(sys.argv)
        myapp = StartQt4(args.psweep, metafile, Ui, startndx=args.start_ndx,
                         goodcut=args.gcut, badcut=args.bcut, useml=args.useml)
        myapp.show()
        app.exec_()
