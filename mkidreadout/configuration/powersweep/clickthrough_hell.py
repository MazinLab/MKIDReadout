#!/usr/bin/env python
#clickthrough hell
from __future__ import print_function
from numpy import *
import numpy
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from mkidreadout.utils.iqsweep import *
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar

from pkg_resources import resource_filename

import mkidreadout.configuration.powersweep.gui as gui
import mkidreadout.configuration.sweepdata as sweepdata
from mkidreadout.configuration.powersweep import psmldata
from mkidcore.corelog import getLogger, create_log
import argparse


class StartQt4(QMainWindow):
    def __init__(self, psfile, psmetafile, Ui, parent=None, startndx=0, goodcut=np.inf, badcut=-np.inf):
        QWidget.__init__(self, parent)

        self.ui = Ui()
        self.ui.setupUi(self)

        self.atten = -1
        self.ui.atten.setValue(self.atten)
        self.resnum = startndx
        self.indx = 0

        self.badcut, self.goodcut = badcut, goodcut

        QObject.connect(self.ui.atten, SIGNAL("valueChanged(int)"), self.setnewatten)
        QObject.connect(self.ui.savevalues, SIGNAL("clicked()"), self.savevalues)
        QObject.connect(self.ui.jumptores, SIGNAL("clicked()"), self.jumptores)

        QShortcut(QKeySequence(Qt.Key_Space), self, self.savevalues)

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

        self.widesweep_goodFreqs = mdata.wsfreq[mdata.flag == sweepdata.ISGOOD]
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

    def loadres(self):
        self.Res1 = IQsweep()
        self.Res1.loadpowers_from_freqsweep(self.fsweepdata, self.resnum)

        self.resfreq = self.fsweepdata.opt_freqs[self.resnum]

        if not (self.Res1.freq[0] < self.resfreq < self.Res1.freq[-1]):
            raise RuntimeError('Out of order resfreq flag={}'.format(self.fsweepdata.mdResMask[
                                                                     self.fsweepdata.resIDs==self.Res1.resID]))

        self.resID = self.Res1.resID
        self.NAttens = self.Res1.atten1s.size

        self.ui.res_num.setText(str(self.resnum))
        self.ui.jumptonum.setValue(self.resnum)
        self.ui.frequency.setText(str(self.resfreq/1e9))

        getLogger(__name__).info("Res: {} --> ID: {}".format(self.resnum, self.resID))

        self.res1_iq_vels = np.zeros((self.NAttens, self.Res1.fsteps - 1))
        self.res1_iq_amps = np.zeros((self.NAttens, self.Res1.fsteps))

        foo = np.sqrt(np.diff(self.Res1.Qs, axis=1)**2 + np.diff(self.Res1.Is, axis=1)**2)
        self.res1_iq_vels[:, 1:] = foo[:, :-1]
        self.res1_iq_amps = np.sqrt(self.Res1.Qs**2 + self.Res1.Is**2)

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
        self.ui.plot_1.canvas.ax.clear()
        self.ui.plot_1.canvas.ax.plot(self.Res1.atten1s, self.res1_max_vels, 'b.-', label='Max IQ velocity')
        # self.ui.plot_1.canvas.ax.plot(self.Res1.atten1s,max_neighbors,'r.-')
        self.ui.plot_1.canvas.ax.plot(self.Res1.atten1s, self.res1_max_ratio, 'k.-',
                                      label='Ratio (Max Vel)/(2nd Max Vel)')
        self.ui.plot_1.canvas.ax.legend()
        self.ui.plot_1.canvas.ax.set_xlabel('Attenuation')
        self.ui.plot_1.canvas.ax.set_ylabel('IQVel')

        cid = self.ui.plot_1.canvas.mpl_connect('button_press_event', self.click_plot_1)
        self.ui.plot_1.canvas.draw()

        max_ratio_threshold = 1.5
        rule_of_thumb_offset = 2

        # require ROTO adjacent elements to be all below the MRT
        bool_remove = np.ones(len(self.res1_max_ratio))
        for ri in range(len(self.res1_max_ratio) - rule_of_thumb_offset - 2):
            bool_remove[ri] = bool((self.res1_max_ratio[ri:ri + rule_of_thumb_offset + 1] < max_ratio_threshold).all())

        use = self.resID == self.mlResIDs
        if np.any(use):
            getLogger(__name__).debug('Select atten: {}'.format(self.mlAttens[use][0]))
            self.select_atten(self.mlAttens[use][0])
            self.ui.atten.setValue(int(np.round(self.mlAttens[use][0])))
        else:
            guess_atten_idx = np.extract(bool_remove, np.arange(len(self.res1_max_ratio)))

            # require the attenuation value to be past the initial peak in MRT
            guess_atten_idx = guess_atten_idx[where(guess_atten_idx > argmax(self.res1_max_ratio))[0]]

            if size(guess_atten_idx) >= 1:
                if guess_atten_idx[0] + rule_of_thumb_offset < len(self.Res1.atten1s):
                    guess_atten_idx[0] += rule_of_thumb_offset
                guess_atten = self.Res1.atten1s[guess_atten_idx[0]]
                getLogger(__name__).debug('Select atten (guess): {}'.format(guess_atten))
                self.select_atten(guess_atten)
                self.ui.atten.setValue(round(guess_atten))
            else:
                getLogger(__name__).debug('Select atten (mid): {}'.format(self.Res1.atten1s[self.NAttens / 2]))
                self.select_atten(self.Res1.atten1s[self.NAttens / 2])
                self.ui.atten.setValue(round(self.Res1.atten1s[self.NAttens / 2]))

    def guess_res_freq(self):
        if np.any(self.resID == self.mlResIDs):
            resInd = np.where(self.resID == self.mlResIDs)[0]
            self.select_freq(self.mlFreqs[resInd])
        else:
            guess_idx = argmax(self.res1_iq_vels[self.iAtten])
            # The longest edge is identified, choose which vertex of the edge
            # is the resonant frequency by checking the neighboring edges
            # len(IQ_vels[ch]) == len(f_span)-1, so guess_idx is the index
            # of the lower frequency vertex of the longest edge
            if guess_idx - 1 < 0 or self.res1_iq_vel[guess_idx - 1] < self.res1_iq_vel[guess_idx + 1]:
                iNewResFreq = guess_idx
            else:
                iNewResFreq = guess_idx - 1
            guess = self.Res1.freq[iNewResFreq]
            getLogger(__name__).info('Guessing resonant freq at {} for self.iAtten={}'.format(guess, self.iAtten))
            self.select_freq(guess)

    def loadps(self):
        self.fsweepdata = psmldata.MLData(fsweep=self.openfile, mdata=self.metadata_out)
        self.widesweep = self.fsweepdata.freqSweep.oldwsformat(65)
        getLogger(__name__).info('Loaded '+self.openfile)

        self.stop_ndx = self.fsweepdata.prioritize_and_cut(self.badcut, self.goodcut, plot=False)

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
        self.resfreq = freq[0]
        self.ui.frequency.setText(str(self.resfreq/1e9))
        self.ui.plot_2.canvas.ax.plot(self.Res1.freq[self.indx], self.res1_iq_vel[self.indx], 'bo')
        self.ui.plot_3.canvas.ax.plot(self.Res1.I[self.indx], self.Res1.Q[self.indx], 'bo')
        self.indx = min(where(self.Res1.freq >= self.resfreq)[0][0], self.Res1.freq.size - 1)
        self.ui.plot_2.canvas.ax.plot(self.Res1.freq[self.indx], self.res1_iq_vel[self.indx], 'ro')
        self.ui.plot_2.canvas.draw()
        self.ui.plot_3.canvas.ax.plot(self.Res1.I[self.indx], self.Res1.Q[self.indx], 'ro')
        self.ui.plot_3.canvas.draw()

    def select_atten(self, attenuation):
        if self.atten != -1:
            self.iAtten = np.abs(self.Res1.atten1s - self.atten).argmin()
            self.ui.plot_1.canvas.ax.plot(self.atten, self.res1_max_ratio[self.iAtten], 'ko')
            self.ui.plot_1.canvas.ax.plot(self.atten, self.res1_max_vels[self.iAtten], 'bo')

        self.iAtten = np.abs(self.Res1.atten1s - attenuation).argmin()
        self.atten = np.round(self.Res1.atten1s[self.iAtten])

        self.res1_iq_vel = self.res1_iq_vels[self.iAtten, :]
        self.Res1.I = self.Res1.Is[self.iAtten]
        self.Res1.Q = self.Res1.Qs[self.iAtten]
        self.Res1.Icen = self.Res1.Icens[self.iAtten]
        self.Res1.Qcen = self.Res1.Qcens[self.iAtten]
        self.ui.plot_1.canvas.ax.plot(self.atten, self.res1_max_ratio[self.iAtten], 'ro')
        self.ui.plot_1.canvas.ax.plot(self.atten, self.res1_max_vels[self.iAtten], 'ro')
        self.ui.plot_1.canvas.draw()
        self.makeplots()
        self.guess_res_freq()

    def makeplots(self):
        try:
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

            self.ui.plot_2.canvas.ax.set_xlim(self.resfreq-1e6, self.resfreq+1e6)
            cid = self.ui.plot_2.canvas.mpl_connect('button_press_event', self.on_press)

            #self.widesweep=None
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

            for ws_resID_i, ws_freq_i in zip(ws_allResIDs, ws_allFreqs):
                ws_color = 'r' if ws_freq_i in self.widesweep_goodFreqs else 'k'
                alpha = 0.5 if ws_freq_i != self.resfreq else 1.0
                self.ui.plot_2.canvas.ax.axvline(ws_freq_i, c=ws_color, alpha=alpha)
                ws_ymax = self.ui.plot_2.canvas.ax.yaxis.get_data_interval()[1]
                self.ui.plot_2.canvas.ax.text(x=ws_freq_i, y=ws_ymax, s=str(int(ws_resID_i)),
                                              color=ws_color, alpha=alpha)
            self.ui.plot_2.canvas.draw()

            #Plot 3
            self.ui.plot_3.canvas.ax.clear()
            if self.iAtten > 0:
                self.ui.plot_3.canvas.ax.plot(self.Res1.Is[self.iAtten - 1], self.Res1.Qs[self.iAtten - 1], 'g.-')
                self.ui.plot_3.canvas.ax.lines[0].set_alpha(.6)
            if self.iAtten > 1:
                self.ui.plot_3.canvas.ax.plot(self.Res1.Is[self.iAtten - 2], self.Res1.Qs[self.iAtten - 2], 'g.-')
                self.ui.plot_3.canvas.ax.lines[-1].set_alpha(.3)
            self.ui.plot_3.canvas.ax.plot(self.Res1.I, self.Res1.Q, '.-')

            if self.widesweep is not None:
                ws_I = self.widesweep[wsmask, 1]
                ws_Q = self.widesweep[wsmask, 2]
                ws_dataRange_I = self.ui.plot_3.canvas.ax.xaxis.get_data_interval()
                ws_dataRange_Q = self.ui.plot_3.canvas.ax.yaxis.get_data_interval()
                ws_I -= numpy.median(ws_I) - numpy.median(ws_dataRange_I)
                ws_Q -= numpy.median(ws_Q) - numpy.median(ws_dataRange_Q)

            getLogger(__name__).debug('makeplots')
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
        self.metadata_out.atten[self.resID == self.metadata_out.resIDs] = self.atten
        self.metadata_out.save()

        msg = " ....... Saved to file:  resnum={} resID={} resfreq={} atten={}"
        getLogger(__name__).info(msg.format(self.resnum,self.resID,self.resfreq,self.atten))
        self.resnum += 1
        self.atten = -1

        if self.resnum >= self.stop_ndx:
            getLogger(__name__).info("reached end of resonator list, closing GUI")
            sys.exit()
        self.loadres()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MKID Powersweep GUI')
    parser.add_argument('psweep', default='', type=str, help='A frequency sweep npz file to load')
    parser.add_argument('metafile', default='', type=str, help='The matching metadata.txt file to load')
    parser.add_argument('--small', action='store_true', dest='smallui', default=False, help='Use small GUI')
    parser.add_argument('-i', dest='start_ndx', type=int, default=0, help='Starting resonator index')
    parser.add_argument('-gc', dest='gcut', default=1, type=float, help='Assume good if net ML score > (EXACT)')
    parser.add_argument('-bc', dest='bcut', default=-1, type=float, help='Assume bad if net ML score < (EXACT)')
    args = parser.parse_args()

    create_log('__main__', console=True, mpsafe=True, fmt='%(asctime)s PSClickthrough: %(levelname)s %(message)s')
    create_log('mkidreadout', console=True, mpsafe=True, fmt='%(asctime)s mkidreadout: %(levelname)s %(message)s')
    create_log('mkidcore', console=True, mpsafe=True, fmt='%(asctime)s mkidcore: %(levelname)s %(message)s')

    Ui = gui.Ui_MainWindow_Small if args.smallui else gui.Ui_MainWindow

    app = QApplication(sys.argv)
    myapp = StartQt4(args.psweep, args.metafile, Ui, startndx=args.start_ndx,
                     goodcut=args.gcut, badcut=args.bcut)
    myapp.show()
    app.exec_()


