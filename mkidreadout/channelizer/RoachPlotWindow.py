"""
Author: Alex Walter
Date: May 24, 2016

Classes for the plot windows for HighTemplar.py. Sweep or phase timestream


Classes:
    RoachSweepWindow - class for plotting IQ sweep
    RoachPhaseStreamWindow - class for plotting phase timestream


Note: see http://bastibe.de/2013-05-30-speeding-up-matplotlib.html for making matplotlib faster
"""

import numpy as np
from functools import partial
from PyQt4.QtGui import *
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from mkidcore.corelog import getLogger
import mkidreadout.configuration.sweepdata as sweepdata
import scipy.signal, skimage.feature, scipy.integrate

try:
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
except ImportError:  # Named changed in some newer matplotlib versions
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from mkidreadout.channelizer.RoachStateMachine import RoachStateMachine


class RoachPhaseStreamWindow(QMainWindow):
    thresholdClicked = QtCore.pyqtSignal()
    phaseSnapClicked = QtCore.pyqtSignal()
    phaseTimestreamClicked = QtCore.pyqtSignal()
    resetRoach = QtCore.pyqtSignal(int)

    def __init__(self, roach, config, parent=None):
        """
        Window for showing snapshot of phase timestream of resonators
        
        INPUTS:
            roach - A RoachStateMachine Object. We need this to access all the relevent settings
            config - ConfigParser object
            parent - Leave as default
        """
        QMainWindow.__init__(self, parent=parent)
        self._want_to_close = False
        self.roach = roach
        self.roachNum = self.roach.num
        self.config = config
        self.setWindowTitle('r' + str(self.roachNum) + ': Phase Timestream')

        self.create_main_frame()

        self.snapDataList = []
        self.phaseNoiseDataList = []
        self.oldPhaseNoiseDataList = []
        self.thresholdDataList = []

        self.roach.snapPhase.connect(self.plotSnap)
        self.roach.timestreamPhase.connect(self.plotPhaseNoise)

    def initFreqs(self):
        """
        After we've loaded the frequency file in RoachStateMachine object then we can initialize some GUI elements
        This also gets called everytime we switch channels and when new phase data comes in
        """
        freqs = self.roach.roachController.freqList
        ch = self.spinbox_channel.value()
        resID = self.roach.roachController.resIDs[ch]

        self.spinbox_channel.setRange(0, len(freqs) - 1)
        self.label_resID.setText('ResID: ' + str(int(resID)))
        self.label_freq.setText('Freq: ' + str(freqs[ch] / 1.e9) + ' GHz')

        resDistance = np.asarray(freqs) - freqs[ch]
        resDistance[ch] = freqs[ch]
        nearestResCh = np.argmin(np.abs(resDistance))
        self.label_nearestRes.setText(
            'Nearest Res: ch ' + str(nearestResCh) + ' --> ' + str(resDistance[nearestResCh] / 1.e3) + ' kHz')
        collisionThreshold = 500000  # Collision if less than 500kHz apart
        palette = self.label_nearestRes.palette()
        if np.abs(resDistance[nearestResCh]) < collisionThreshold:
            palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.red)
            self.label_nearestRes.setPalette(palette)
        else:
            palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.black)
            self.label_nearestRes.setPalette(palette)
        try:
            lofreq = self.roach.roachController.LOFreq
            aliasFreqs = (np.asarray(freqs) - lofreq) * -1 + lofreq
            aliasDist = aliasFreqs - freqs[ch]
            nearestAliasCh = np.argmin(np.abs(aliasDist))
            self.label_nearestSideband.setText(
                'Nearest Alias: ch ' + str(nearestAliasCh) + ' --> ' + str(aliasDist[nearestAliasCh] / 1.e3) + ' kHz')
            palette = self.label_nearestRes.palette()
            if np.abs(aliasDist[nearestAliasCh]) < collisionThreshold:
                palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.red)
                self.label_nearestSideband.setPalette(palette)
            else:
                palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.black)
                self.label_nearestSideband.setPalette(palette)
        except AttributeError:
            pass

        if len(self.snapDataList) != len(freqs):
            self.snapDataList = [None] * len(freqs)
        if len(self.phaseNoiseDataList) != len(freqs):
            self.phaseNoiseDataList = [None] * len(freqs)
        if len(self.oldPhaseNoiseDataList) != len(freqs):
            self.oldPhaseNoiseDataList = [None] * len(freqs)

    def initThresh(self):
        """
        After we've loaded the thresholds we can show them
        Also everytime we change the channel we need to replot
        """
        try:
            ch = self.spinbox_channel.value()
            # thresh = self.roach.roachController.thresholds[ch]
            thresh = np.round(self.thresholdDataList[ch], 4)
            self.label_thresh.setText('Threshold: ' + str(thresh) + ' deg')
            self.plotSnap()
        except:
            pass

    def appendThresh(self, thresholds):
        """
        This function is called whenever the roach finishes a load threshold command

        INPUTS:
            thresholds - [nFreqs, 1] list of thresholds for each channel in radians
        """
        self.thresholdDataList = thresholds
        self.initThresh()

    def plotPhaseNoise(self, ch=None, data=None, **kwargs):
        # self.spinbox_channel.setEnabled(False)
        currentCh = self.spinbox_channel.value()
        if ch is None: ch = currentCh
        if data is not None:
            self.appendPhaseNoiseData(ch, data)
        if self.isVisible() and ch == currentCh:
            self.makePhaseNoisePlot(**kwargs)
            self.draw()
        # self.spinbox_channel.setEnabled(True)

    def appendPhaseNoiseData(self, ch, data):
        fftlen = self.config.get('r{}.nLongsnapFftSamples'.format(self.roachNum))
        nFftAvg = int(np.floor(len(data) / fftlen))
        dt = 1. * (self.roach.roachController.params['nChannelsPerStream'] /
                   self.roach.roachController.params['fpgaClockRate'])

        data = np.reshape(data[:nFftAvg * fftlen], (nFftAvg, fftlen))
        noiseData = np.fft.rfft(data)
        noiseData = np.abs(noiseData) ** 2  # power spectrum
        noiseData = dt * np.average(noiseData, axis=0) / fftlen  # normalize
        noiseData = 10. * np.log10(noiseData)  # convert to dBc/Hz

        if not np.all(noiseData == 0):
            getLogger(__name__).info('Adding noise data')
            if self.phaseNoiseDataList[ch] is not None:
                getLogger(__name__).info('adding old noise data')
                self.oldPhaseNoiseDataList[ch] = np.copy(self.phaseNoiseDataList[ch])
            self.phaseNoiseDataList[ch] = noiseData
        else:
            getLogger(__name__).info("Phase noise was all zeros!")

    def makePhaseNoisePlot(self, **kwargs):
        ch = self.spinbox_channel.value()

        fftlen = self.config.get('r{}.nLongsnapFftSamples'.format(self.roachNum))
        if self.oldPhaseNoiseDataList[ch] is not None:
            ydata = np.copy(self.oldPhaseNoiseDataList[ch])
            dt = 1. * (self.roach.roachController.params['nChannelsPerStream'] /
                       self.roach.roachController.params['fpgaClockRate'])
            x = np.fft.rfftfreq(fftlen, dt)
            self.line2.set_data(x, ydata)
            self.ax1.relim()
            self.ax1.autoscale_view(True, True, True)
        else:
            self.line2.set_data([], [])

        if self.phaseNoiseDataList[ch] is not None:
            ydata = np.copy(self.phaseNoiseDataList[ch])
            dt = 1. * (self.roach.roachController.params['nChannelsPerStream'] /
                       self.roach.roachController.params['fpgaClockRate'])
            x = np.fft.rfftfreq(fftlen, dt)
            self.line1.set_data(x, ydata)
            self.ax1.relim()
            self.ax1.autoscale_view(True, True, True)
        else:
            self.line1.set_data([], [])

    def phaseTimeStreamAllChannels(self):
        """ Loops through """
        timelen = self.config.get('r{}.longsnaptime'.format(self.roachNum))
        for ch in range(len(self.roach.roachController.freqList)):
            QtCore.QMetaObject.invokeMethod(self.roach, 'getPhaseStream', Qt.QueuedConnection,
                                            QtCore.Q_ARG(int, ch), QtCore.Q_ARG(float, timelen))
        self.phaseTimestreamClicked.emit()

    def phaseTimeStream(self):
        """
        This function executes when you press the collect phase timestream button.
        After running any commands that need running, it gets a new phase timestream and plots the noise
        
        Works similiarly to phaseSnapShot()
        """
        ch = self.spinbox_channel.value()
        timelen = self.config.get('r{}.longsnaptime'.format(self.roachNum))
        QtCore.QMetaObject.invokeMethod(self.roach, 'getPhaseStream', Qt.QueuedConnection,
                                        QtCore.Q_ARG(int, ch), QtCore.Q_ARG(float, timelen))
        self.phaseTimestreamClicked.emit()

    def plotSnap(self, ch=None, data=None, **kwargs):
        """
        This function executes when the roachStateMachine object emits a snapPhase signal with new data.
        We also execute without new data when switching channels so we can replot the correct snap (if it exists)
        """
        self.spinbox_channel.setEnabled(False)
        currentCh = self.spinbox_channel.value()
        if ch is None: ch = currentCh
        if data is not None:
            self.appendSnapData(ch, data)
        if self.isVisible() and ch == currentCh:
            self.makeSnapPlot(**kwargs)
            self.draw()
        self.spinbox_channel.setEnabled(True)

    def appendSnapData(self, ch, data):
        """
        This function is called when plotSnap() is executed by the RoachStateMachine signal with new snap data
        """
        self.snapDataList[ch] = data

    def makeSnapPlot(self, **kwargs):
        """
        This function actually changes the plot.
        You can pass plot arguments through kwargs but we don't at the moment
        
        NOTE: snapDict['trig'] only contains firmware trigger data from stream 0
        """
        ch = self.spinbox_channel.value()
        self.ax2.clear()
        if self.snapDataList[ch] is not None:
            snapDict = self.snapDataList[ch]
            t = np.asarray(snapDict['time']) * 1.e6
            data = np.asarray(snapDict['phase'])
            # print snapDict['trig']
            # print np.where(np.asarray(snapDict['time']))
            # print np.where(np.asarray(snapDict['time'])>0)
            trig = np.asarray(snapDict['trig'])
            swTrig = np.asarray(snapDict['swTrig'])
            data *= 180. / np.pi
            fmt = 'b.-'
            self.ax2.plot(t, data, fmt, **kwargs)
            _, stream = self.roach.roachController.getStreamChannelFromFreqChannel(ch)
            if stream[0] == 0:
                getLogger(__name__).info('nPhotons: %s', np.sum(trig))
                self.ax2.plot(t[np.where(trig)], data[np.where(trig)], 'ro')
            self.ax2.plot(t[np.where(swTrig)], data[np.where(swTrig)], 'go')
            median = np.round(np.median(data), 4)
            self.label_median.setText('Median: ' + str(median) + ' deg')
            self.ax2.axhline(y=median, color='k')
            try:
                # thresh = self.roach.roachController.thresholds[ch]*180./np.pi
                thresh = self.thresholdDataList[ch]
                thresh *= 180. / np.pi
                # self.ax2.axhline(y=median-thresh,color='r')
                self.ax2.axhline(y=median + thresh, color='r')
            except:
                pass
        self.ax2.set_xlabel('Time [us]')
        self.ax2.set_ylabel('Phase [deg]')

    def phaseSnapShot(self):
        """
        This function executes when you press the phase snap button.
        After running any commands that need running, it gets a new phase snap shot and plots it
        
        The way it works is: 
            The RoachStateMachine object is told that we want to run the function getPhaseFromSnap()
            But it won't run unless the roach thread is in it's event loop. 
            We emit a signal to the HighTemplar GUI telling it we want the phase snap shot
            
            --That's the end of this function-- (but next...)
            
            HighTemplar tells the RoachStateMachine object to add all commands (as if you clicked a button 1 below load thresholds)
            Then HighTemplar starts the roach thread's event loop
            The roach thread automatically starts executing any needed commands in it's queue (ie. loadFreq, DefineLUT, etc..)
            When the commands are done executing it executes anything waiting for the thread event loop (ie. getPhaseFromSnap())
            getPhaseFromSnap() runs on the RoachStateMachine object and when done emits a snapPhase signal with the data
            This window object sees that signal and updates the plot with plotSnap()
        """
        ch = self.spinbox_channel.value()
        QtCore.QMetaObject.invokeMethod(self.roach, 'getPhaseFromSnap', Qt.QueuedConnection,
                                        QtCore.Q_ARG(int, ch))
        self.phaseSnapClicked.emit()

    def create_main_frame(self):
        """
        Makes GUI elements on the window
        
        
        """
        self.main_frame = QWidget()

        self.dpi = 100
        self.fig = Figure((9.0, 5.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_ylabel('Phase Noise PSD [dBc/Hz]')
        self.ax1.set_xlabel('f [Hz]')
        self.line2, = self.ax1.semilogx([], [], color='cyan')
        self.line1, = self.ax1.semilogx([], [], color='blue')  # line 1 on top of line 2
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_xlabel('Time [us]')
        self.ax2.set_ylabel('Phase [Deg]')

        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        label_channel = QLabel('Channel:')
        self.spinbox_channel = QSpinBox()  # initializes to 0
        self.spinbox_channel.setRange(0, 0)  # set the range after we read the freq file
        self.spinbox_channel.setWrapping(True)
        # self.spinbox_channel.valueChanged.connect(lambda x: self.plotSnap())
        # self.spinbox_channel.valueChanged.connect(lambda x: self.plotPhaseNoise())
        # self.spinbox_channel.valueChanged.connect(lambda x: self.initFreqs())
        # self.spinbox_channel.valueChanged.connect(lambda x: self.initThresh())
        self.spinbox_channel.editingFinished.connect(self.initFreqs)
        self.spinbox_channel.editingFinished.connect(self.initThresh)
        self.spinbox_channel.editingFinished.connect(self.plotSnap)
        self.spinbox_channel.editingFinished.connect(self.plotPhaseNoise)

        self.label_resID = QLabel('ResID: ')
        self.label_resID.setMinimumWidth(80)
        self.label_resID.setMaximumWidth(80)

        self.label_freq = QLabel('Freq: 0 GHz')
        self.label_freq.setMinimumWidth(145)
        self.label_freq.setMaximumWidth(145)
        self.label_thresh = QLabel('Thresh: 0 deg')
        self.label_thresh.setMinimumWidth(120)
        self.label_thresh.setMaximumWidth(120)
        self.label_median = QLabel('Median: 0 deg')
        self.label_median.setMinimumWidth(120)
        self.label_median.setMaximumWidth(120)

        button_snapPhase = QPushButton("Phase Snapshot")
        button_snapPhase.setEnabled(True)
        button_snapPhase.clicked.connect(self.phaseSnapShot)

        self.label_nearestRes = QLabel('Nearest Res: ')
        self.label_nearestRes.setMinimumWidth(240)
        self.label_nearestRes.setMaximumWidth(240)
        self.label_nearestSideband = QLabel('Nearest Alias: ')
        self.label_nearestSideband.setMinimumWidth(240)
        self.label_nearestSideband.setMaximumWidth(240)

        numSnapsThresh = self.config.get('r{}.numsnaps_thresh'.format(self.roachNum))
        spinbox_numSnapsThresh = QSpinBox()
        spinbox_numSnapsThresh.setValue(numSnapsThresh)
        spinbox_numSnapsThresh.setRange(1, 100)
        spinbox_numSnapsThresh.setSuffix(" *2 msec")
        spinbox_numSnapsThresh.setWrapping(False)
        spinbox_numSnapsThresh.setButtonSymbols(QAbstractSpinBox.NoButtons)
        spinbox_numSnapsThresh.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_numSnapsThresh.valueChanged.connect(partial(self.changedSetting, 'numsnaps_thresh'))

        threshSigs = self.config.get('r{}.numsigs_thresh'.format(self.roachNum))
        spinbox_threshSigs = QDoubleSpinBox()
        spinbox_threshSigs.setValue(threshSigs)
        spinbox_threshSigs.setRange(0, 100)
        spinbox_threshSigs.setSuffix(" sigmas")
        spinbox_threshSigs.setWrapping(False)
        spinbox_threshSigs.setButtonSymbols(QAbstractSpinBox.NoButtons)
        spinbox_threshSigs.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_threshSigs.valueChanged.connect(partial(self.changedSetting, 'numsigs_thresh'))
        # reset state of roach
        spinbox_threshSigs.valueChanged.connect(lambda x: self.resetRoach.emit(RoachStateMachine.LOADTHRESHOLD))

        button_loadThresh = QPushButton("Load Thresholds")
        button_loadThresh.setEnabled(True)
        button_loadThresh.clicked.connect(self.thresholdClicked)

        longSnapTime = self.config.get('r{}.longsnaptime'.format(self.roachNum))
        spinbox_longSnapTime = QDoubleSpinBox()
        spinbox_longSnapTime.setValue(longSnapTime)
        spinbox_longSnapTime.setRange(0, 1000)
        spinbox_longSnapTime.setSuffix(" seconds")
        spinbox_longSnapTime.setWrapping(False)
        spinbox_longSnapTime.setButtonSymbols(QAbstractSpinBox.NoButtons)
        spinbox_longSnapTime.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_longSnapTime.valueChanged.connect(partial(self.changedSetting, 'longsnaptime'))

        button_longSnap = QPushButton("Collect Phase Timestream")
        button_longSnap.setEnabled(True)
        button_longSnap.clicked.connect(self.phaseTimeStream)

        button_allLongSnap = QPushButton("Collect Phase Timestream for all Channels")

        def allChPhaseStreamClicked():
            ret = QMessageBox.warning(self, 'Collect Phase Timestream for all Channels',
                                      "Are you sure you want to collect phase time stream "
                                      "data on all channels? This may take a while. Also, don't "
                                      "try to collect photon data while streaming phase streams "
                                      "over the ethernet.",
                                      buttons=QMessageBox.Ok | QMessageBox.Cancel,
                                      defaultButton=QMessageBox.Cancel)
            if ret == QMessageBox.Ok:
                self.phaseTimeStreamAllChannels()

        button_allLongSnap.clicked.connect(allChPhaseStreamClicked)

        #TODO fix changedsetting in all locations. it assumes it is a roachspecific setting
        longSnapFile = self.config.get('r{}.longsnaproot'.format(self.roachNum))
        label_longSnapFile = QLabel('Long Snap Filename:')
        textbox_longSnapFile = QLineEdit(longSnapFile)
        textbox_longSnapFile.setMaximumWidth(400)
        textbox_longSnapFile.setMinimumWidth(400)
        textbox_longSnapFile.textChanged.connect(partial(self.changedSetting, 'longsnaproot'))

        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        vbox_plot.addWidget(self.mpl_toolbar)

        hbox_ch = QHBoxLayout()
        hbox_ch.addWidget(label_channel)
        hbox_ch.addWidget(self.spinbox_channel)
        hbox_ch.addWidget(self.label_resID)
        hbox_ch.addSpacing(5)
        hbox_ch.addWidget(self.label_freq)
        hbox_ch.addWidget(self.label_thresh)
        hbox_ch.addWidget(self.label_median)
        hbox_ch.addWidget(button_snapPhase)
        hbox_ch.addStretch()

        hbox_nearestRes = QHBoxLayout()
        hbox_nearestRes.addWidget(self.label_nearestRes)
        hbox_nearestRes.addSpacing(7)
        hbox_nearestRes.addWidget(self.label_nearestSideband)
        hbox_nearestRes.addStretch()

        hbox_thresh = QHBoxLayout()
        hbox_thresh.addWidget(spinbox_numSnapsThresh)
        hbox_thresh.addWidget(spinbox_threshSigs)
        hbox_thresh.addWidget(button_loadThresh)
        hbox_thresh.addStretch()

        hbox_phaseTimestream = QHBoxLayout()
        hbox_phaseTimestream.addWidget(spinbox_longSnapTime)
        hbox_phaseTimestream.addWidget(button_longSnap)
        hbox_phaseTimestream.addWidget(button_allLongSnap)
        hbox_phaseTimestream.addStretch()

        hbox_longSnapFile = QHBoxLayout()
        hbox_longSnapFile.addWidget(label_longSnapFile)
        hbox_longSnapFile.addWidget(textbox_longSnapFile)
        hbox_longSnapFile.addStretch()

        vbox1 = QVBoxLayout()
        vbox1.addLayout(vbox_plot)
        vbox1.addLayout(hbox_ch)
        vbox1.addLayout(hbox_nearestRes)
        vbox1.addLayout(hbox_thresh)
        vbox1.addLayout(hbox_phaseTimestream)
        vbox1.addLayout(hbox_longSnapFile)

        self.main_frame.setLayout(vbox1)
        self.setCentralWidget(self.main_frame)

    def draw(self):
        """
        The plot window calls this function
        """
        # print 'r'+str(self.roachNum)+' drawing data - '+str(self.counter)
        self.canvas.draw()
        self.canvas.flush_events()

    def changedSetting(self, settingID, setting):
        """
        When a setting is changed, reflect the change in the config object which is shared across all GUI elements.
        
        INPUTS:
            settingID - the key in the configparser
            setting - the value
        """
        old = self.config.get('r{}.{}'.format(self.roachNum, settingID))
        self.config.update('r{}.{}'.format(self.roachNum, settingID), setting)
        # If we don't force the setting value to be a string then the configparser has trouble grabbing the value later on for some unknown reason
        newSetting = self.config.get('r{}.{}'.format(self.roachNum, settingID))
        getLogger(__name__).info('setting {} from {} to {}'.format(settingID, old, newSetting))

    def closeEvent(self, event):
        """
        When you try to close the window it just hides it instead so that all the internal variables are saved
        """
        if self._want_to_close:
            event.accept()
            self.close()
        else:
            event.ignore()
            self.hide()


class RoachSweepWindow(QMainWindow):
    sweepClicked = QtCore.pyqtSignal()
    # fitClicked = QtCore.pyqtSignal()
    rotateClicked = QtCore.pyqtSignal()
    translateClicked = QtCore.pyqtSignal()
    adcAttenChanged = QtCore.pyqtSignal()
    dacAttenChanged = QtCore.pyqtSignal()
    resetRoach = QtCore.pyqtSignal(int)

    def __init__(self, roach, config, parent=None):
        """
        Window for showing IQ plot of resonators
        
        INPUTS:
            roach - A RoachStateMachine Object. We need this to access all the relevent settings
            config - ConfigParser object
            parent - Leave as default
        """
        QMainWindow.__init__(self, parent=parent)
        self._want_to_close = False
        self.roach = roach
        self.roachNum = self.roach.num
        self.config = config
        self.setWindowTitle('r' + str(self.roachNum) + ': IQ Plot')

        self.channelsModified = set()  # Remember which channels we've already modified but haven't reloaded into roach
        self.dataList = []  # Save data from sweeps and translates in memory
        self.rotatedList = []  # Save data from rotate corresponding to data in dataList
        self.numData2Show = 4  # number of previous sweeps to show
        self.maxDataListLength = 10  # maximum number of sweeps to save in memory

        self.create_main_frame()
        # self.counter = 0

    def plotData(self, data=None, rotated=False, **kwargs):
        # print 'Plotting Data: ',data
        if data is not None and not rotated:
            self.appendData(data.copy())
        if rotated:
            self.appendRotated(data.copy())
        if self.isVisible():
            self.makePlot(**kwargs)
            self.draw()

    def appendData(self, data):
        if len(self.dataList) > 0:
            if len(data['I']) != len(self.dataList[0]['I']):  # Changed the frequncy list
                self.dataList = []
                self.rotatedList = []
        self.dataList.append(data)
        self.rotatedList.append(None)
        if len(self.dataList) > self.maxDataListLength:
            self.dataList = self.dataList[-1 * self.maxDataListLength:]
            self.rotatedList = self.rotatedList[-1 * self.maxDataListLength:]

    def appendRotated(self, data):
        self.rotatedList[-1] = data

    def makePlot(self, **kwargs):
        getLogger(__name__).info("Making sweep plot")
        self.ax.clear()
        self.ax2.clear()
        numData2Show = min(self.numData2Show, len(self.dataList))
        ch = self.spinbox_channel.value()
        c, s = self.roach.roachController.getStreamChannelFromFreqChannel(ch)
        getLogger(__name__).info('ch: {}   freq[ch]: {}'.format(ch, self.roach.roachController.freqList[ch]))
        getLogger(__name__).info('ch/stream: {}/{}   freq[ch,stream]: {}'.format(c, s,
                                self.roach.roachController.freqChannels[c, s]))

        for i in range(len(self.dataList) - numData2Show, len(self.dataList)):

            data = self.dataList[i]
            I = data['I']
            Q = data['Q']
            # kwargs['alpha']=1. if i==0 else .6 - 0.5*(i-1)/(numData2Show-1)
            kwargs['alpha'] = 1. if i == len(self.dataList) - 1 else .6 - .5 * (
                        len(self.dataList) - i - 1) / numData2Show
            fmt = 'b.-' if i == len(self.dataList) - 1 else 'c.-'
            self.ax.plot(I[ch], Q[ch], fmt, **kwargs)
            center = data['centers'][ch]
            # print 'center1 ',center
            self.ax.plot(center[0], center[1], 'gx', alpha=kwargs['alpha'])
            iOnRes = data['IonRes']
            qOnRes = data['QonRes']
            self.ax.plot(iOnRes[ch], qOnRes[ch], 'g.', alpha=kwargs['alpha'])

            # resFreq = self.roach.roachController.freqList[ch]
            # loSpan = self.config.getfloat('Roach '+str(self.roachNum),'sweeplospan')
            # nSteps = len(I[ch])
            # loStep = self.config.getfloat('Roach '+str(self.roachNum),'sweeplostep')
            # freqs = np.linspace(resFreq-loSpan/2., resFreq+loSpan/2., nSteps)
            try:
                # freqs = data['freqOffsets'] + self.roach.roachController.freqList[ch]
                freqs = data['freqOffsets'] + data['freqList'][ch]
                vel = np.sqrt((I[ch][1:] - I[ch][:-1]) ** 2 + (Q[ch][1:] - Q[ch][:-1]) ** 2)
                freqs = freqs[:-1] + (freqs[1] - freqs[0]) / 2.
                self.ax2.plot(freqs, vel, fmt, alpha=kwargs['alpha'])
                # self.ax2.semilogy(freqs, np.sqrt(I[ch]**2 + Q[ch]**2),fmt,alpha=kwargs['alpha'])
            except:
                getLogger(__name__).info("Couldn't make IQ velocity plot", exc_info=True)
            self.ax2.axvline(x=self.roach.roachController.freqList[ch], color='r')

            if self.rotatedList[i] is not None:
                iOnRes2 = self.rotatedList[i]['IonRes'][ch]
                qOnRes2 = self.rotatedList[i]['QonRes'][ch]
                center2 = np.copy(center)  # needs local copy, not pointer
                # print 'center2 ',center2
                avgI = np.average(iOnRes2)
                avgQ = np.average(qOnRes2)
                rotation = self.rotatedList[i]['rotation']
                getLogger(__name__).info('Rotated ch {} by {} deg'.format(ch, -180*rotation[ch]/np.pi))

                self.ax.plot(iOnRes2, qOnRes2, 'r.', alpha=kwargs['alpha'])
                self.ax.plot([center2[0], avgI], [center2[1], avgQ], 'r--', alpha=kwargs['alpha'])
                self.ax.plot([center2[0], avgI], [center2[1], center2[1]], 'r--', alpha=kwargs['alpha'])

                # self.ax.plot(centers[ch][0],centers[ch][1],'rx',**kwargs)
                # self.ax.plot(iqOnRes[ch][0]+centers[ch][0],iqOnRes[ch][1]+centers[ch][1],'ro',**kwargs)
                # print 'channel center plot',ch,centers[ch][0],centers[ch][1]

        self.ax.set_xlabel('I')
        self.ax.set_ylabel('Q')
        self.ax2.set_xlabel('Freqs [Hz]')
        self.ax2.set_ylabel('IQ velocity')

    def draw(self):
        #getLogger(__name__).debug('r{} drawing data - {}'.format(self.roachNum, self.counter))
        self.canvas.draw()
        self.canvas.flush_events()

    def initFreqs(self, fromRoach=True):
        """
        After we've loaded the frequency file in RoachStateMachine object then we can initialize some GUI elements
        Also called whenever we change the current channel
        
        INPUTS:
            fromRoach - True if we've just finished a LoadFreq command
                      - False if we call it otherwise. Like from switching channels
        """
        ch = self.spinbox_channel.value()
        if fromRoach:
            self.channelsModified = set()
        if ch in self.channelsModified:
            self.label_modifyFlag.show()
        else:
            self.label_modifyFlag.hide()

        resIDs = self.roach.roachController.resIDs
        freqs = self.roach.roachController.freqList
        attens = self.roach.roachController.attenList
        try:
            lofreq = self.roach.roachController.LOFreq
        except AttributeError:
            lofreq = 0

        try:
            fList = np.copy(self.roach.roachController.dacQuantizedFreqList)
            fList[np.where(fList > (self.roach.roachController.params['dacSampleRate'] / 2.))] -= \
            self.roach.roachController.params['dacSampleRate']
            fList += self.roach.roachController.LOFreq
            freqs = np.copy(fList)
            self.label_freq.setText('Quantized Freq: ' + str(freqs[ch] / 1.e9) + ' GHz')
        except AttributeError:
            self.label_freq.setText('Freq: ' + str(freqs[ch] / 1.e9) + ' GHz')

        self.spinbox_channel.setRange(0, len(freqs) - 1)
        self.label_resID.setText('resID: ' + str(int(resIDs[ch])))
        self.label_atten.setText('Atten: ' + str(attens[ch]) + ' dB')
        self.label_lofreq.setText('LO Freq: ' + str(lofreq / 1.e9) + ' GHz')

        self.textbox_modifyFreq.setText("%.9e" % freqs[ch])
        self.spinbox_modifyAtten.setValue(int(attens[ch]))

    def getBestFreqs(self, channels=None):
        """
        Get the freq with the highest IQ velocity for channel or list of channels

        INPUTS:
            ch - single value or list of freq channel
                 if negative then do all channels
                 if None then do the current channel
        OUTPUTS:
            newFreqs - list of new freqs
                       The freq is the old freq or the freq with highest iqVel if that channel was specified
        """
        plot = False
        if channels is None:
            channels = self.spinbox_channel.value()
        elif (np.atleast_1d(channels) < 0).any():
            plot = True
            channels = range(len(self.roach.roachController.freqList))

        data = self.dataList[-1]  #[sweep index, channel] i|q|freqOffsets|freqList

        filt_wid = 3
        thresh_rel = .25
        peaksep = 1

        newFreqs = np.copy(data['freqList'])

        getLogger(__name__).debug("   channels: {} {}\n".format(type(channels),channels) +
                                  "   data['freqOffsets'] {}\n   {}\n".format(type(data['I']), data['I']) +
                                  "   data['freqOffsets'] {}\n   {}\n".format(type(data['freqOffsets']), data['freqOffsets']) +
                                  "   data['freqList'] {}\n{}".format(type(data['freqList']), data['freqList']))

        for ch in np.atleast_1d(channels):
            iVals = data['I'][ch]
            qVals = data['Q'][ch]
            iqVel = np.sqrt(np.diff(iVals) ** 2 + np.diff(qVals) ** 2)
            freq = data['freqList'][ch] + data['freqOffsets'][:-1]

            getLogger(__name__).debug('   ivals: {}\n'.format(iVals.shape) +
                                      '   qvals: {}\n'.format(qVals.shape) +
                                      '   iqvel: {}\n'.format(iqVel.shape) +
                                      "   data['freqList'][ch]: {}\n".format(data['freqList'][ch]) +
                                      '   freq:  {}'.format(freq.shape))
            try:
                filt_vel = scipy.signal.medfilt(iqVel, kernel_size=filt_wid)
                peakloc = skimage.feature.peak_local_max(filt_vel, min_distance=peaksep, threshold_rel=thresh_rel,
                                                         exclude_border=True, indices=True, num_peaks=np.inf)
                com = scipy.integrate.trapz(filt_vel * freq) / scipy.integrate.trapz(filt_vel)

                ndx = np.abs(freq[peakloc] - com).argmin()
                old = data['freqList'][ch]
                newFreqs[ch] += data['freqOffsets'][peakloc[ndx]]
                msg = 'Snapped channel {} (rid={}) {} kHz from {} to {} Hz'
                getLogger(__name__).info(msg.format(ch, self.roach.roachController.resIDs[ch],
                                                    (newFreqs[ch]-old)/1024, old, newFreqs[ch]))
            except Exception:
                getLogger(__name__).error('Unable to snap channel {}'.format(ch), exc_info=True)

        if plot:
            cv = FigureCanvas(Figure(figsize=(5, 3)))
            ax = cv.figure.subplots()
            ax.plot(data['freqList']/1024/1024, (data['freqList'] - newFreqs)/1024, '.')
            ax.set_xlabel('Old Freq (MHz)')
            ax.set_ylabel('Freq Shift (kHz)')
            cv.show()

        return newFreqs if np.atleast_1d(channels).size > 1 else newFreqs[channels]

    def snapFreq(self, ch=None):

        if isinstance(ch, bool):  #This is a hack to deal with the button state being passed
            ch = None

        if ch is None:
            ch = self.spinbox_channel.value()

        newFreq = self.getBestFreqs(ch)

        if ch == self.spinbox_channel.value():
            self.textbox_modifyFreq.setText("%.9e" % newFreq)
            try:
                self.clickLine.remove()
            except:
                pass
            self.clickLine = self.ax2.axvline(x=newFreq, color='g')

            # plot closest point in IQ plane
            ch = self.spinbox_channel.value()
            data = self.dataList[-1]
            I = data['I'][ch]
            Q = data['Q'][ch]
            freqs = data['freqOffsets'] + self.roach.roachController.freqList[ch]
            arg = np.abs(np.atleast_1d(freqs) - newFreq).argmin()
            i_click = I[arg]
            q_click = Q[arg]
            try:
                self.clickPoint.remove()
            except:
                pass
            self.clickPoint, = self.ax.plot(i_click, q_click, 'ro')

            self.draw()

    def plotNewFreqLine(self, newFreq):
        ch = self.spinbox_channel.value()
        self.textbox_modifyFreq.setText("%.9e" % newFreq)
        try:
            self.clickLine.remove()
        except:
            pass
        self.clickLine = self.ax2.axvline(x=newFreq, color='g')

        # plot closest point in IQ plane
        data = self.dataList[-1]
        I = data['I'][ch]
        Q = data['Q'][ch]
        freqs = data['freqOffsets'] + self.roach.roachController.freqList[ch]
        arg = np.abs(np.atleast_1d(freqs) - newFreq).argmin()
        i_click = I[arg]
        q_click = Q[arg]
        try:
            self.clickPoint.remove()
        except:
            pass
        self.clickPoint, = self.ax.plot(i_click, q_click, 'ro')

        self.draw()

    def snapAllFreqs(self):
        newFreqs = self.getBestFreqs(-1)
        self.snapFreq()  # plot it on current ch
        attens = self.roach.roachController.attenList
        self.resetRoach.emit(RoachStateMachine.LOADFREQ)
        self.channelsModified = self.channelsModified | set(range(len(newFreqs)))
        self.writeNewFreqFile(np.copy(newFreqs), np.copy(attens))
        self.label_modifyFlag.show()

    def shiftAllFreqs(self):
        deltaFreq = float(self.textbox_shiftAllFreqs.text())
        newFreqs = self.roach.roachController.freqList
        newFreqs = newFreqs + deltaFreq
        self.plotNewFreqLine(newFreq=newFreqs[self.spinbox_channel.value()])
        attens = self.roach.roachController.attenList
        self.resetRoach.emit(RoachStateMachine.LOADFREQ)
        self.channelsModified = self.channelsModified | set(range(len(newFreqs)))
        self.writeNewFreqFile(np.copy(newFreqs), np.copy(attens))
        self.label_modifyFlag.show()

    def saveResonator(self):
        freqs = self.roach.roachController.freqList
        attens = self.roach.roachController.attenList
        ch = self.spinbox_channel.value()

        #getLogger(__name__).debug('P:Old freq: {}'.format(freqs[ch]))
        newFreq = float(self.textbox_modifyFreq.text())
        #getLogger(__name__).debug('P:New freq: {}'.format(newFreq))
        newAtten = self.spinbox_modifyAtten.value()
        if newFreq != freqs[ch]:
            self.resetRoach.emit(RoachStateMachine.LOADFREQ)
        elif newAtten != attens[ch]:
            self.resetRoach.emit(RoachStateMachine.DEFINEDACLUT)

        freqs[ch] = newFreq  # This changes it in the Roach2Control object as well. That's what we want
        attens[ch] = newAtten
        self.channelsModified = self.channelsModified | set([ch])

        self.writeNewFreqFile(np.copy(freqs), np.copy(attens))
        self.label_modifyFlag.show()

    def writeNewFreqFile(self, freqs=None, attens=None):
        if freqs is None:
            freqs = np.copy(self.roach.roachController.freqList)
        if attens is None:
            attens = np.copy(self.roach.roachController.attenList)

        resIDs = np.copy(self.roach.roachController.resIDs)

        freqFile = self.roach.roachController.tagfile(self.config.roaches.get('r{}.freqfileroot'.format(self.roachNum)),
                                                      dir=self.config.paths.data)
        sd = sweepdata.SweepMetadata(file=freqFile)
        sd.file = '{0}_new.{1}'.format(*freqFile.rpartition('.')[::2])
        sd.update_from_roach(resIDs, freqs=freqs, attens=attens)
        getLogger(__name__).info("Saving %s", sd)
        sd.save()

    def changedSetting(self, settingID, setting):
        """
        When a setting is changed, reflect the change in the config object which is shared across all GUI elements.
        
        INPUTS:
            settingID - the key in the configparser
            setting - the value
        """
        old = self.config.roaches.get('r{}.{}'.format(self.roachNum, settingID))
        self.config.roaches.update('r{}.{}'.format(self.roachNum, settingID), setting)
        new = self.config.roaches.get('r{}.{}'.format(self.roachNum, settingID))
        getLogger(__name__).info('setting {} from {} to {}'.format(settingID, old, new))

    def keyPressed(self, event):
        """
        This function is called when a key is pressed on the central widget
        Only works when in focus :-/
        
        INPUTS:
            event - QKeyEvent
        """
        if event.key() == Qt.Key_Left or event.key() == Qt.Key_A:
            getLogger(__name__).debug("decrease channel from keyboard")
        elif event.key() == Qt.Key_Right or event.key() == Qt.Key_D:
            getLogger(__name__).debug("increase channel from keyboard")
        raise NotImplementedError

    def figureClicked(self, event):
        """
        INPUTS: 
            event - matplotlib.backend_bases.MouseEvent
        """
        if self.mpl_toolbar._active is None:
            if event.inaxes == self.ax2:
                clickFreq = event.xdata
                self.textbox_modifyFreq.setText("%.9e" % clickFreq)
                try:
                    self.clickLine.remove()
                except:
                    pass
                self.clickLine = self.ax2.axvline(x=clickFreq, color='g')

                # plot closest point in IQ plane
                ch = self.spinbox_channel.value()
                data = self.dataList[-1]
                I = data['I'][ch]
                Q = data['Q'][ch]
                freqs = data['freqOffsets'] + self.roach.roachController.freqList[ch]
                arg = np.abs(np.atleast_1d(freqs) - clickFreq).argmin()
                i_click = I[arg]
                q_click = Q[arg]
                try:
                    self.clickPoint.remove()
                except:
                    pass
                self.clickPoint, = self.ax.plot(i_click, q_click, 'ro')

                self.draw()
        # print event.inaxes
        # print self.ax
        # print self.ax2
        # print event.xdata

    def updateDACAttenSpinBox(self, dacAtten):
        self.spinbox_dacAttenStart.setValue(dacAtten)
        self.changedSetting('dacatten_start', self.spinbox_dacAttenStart.value())

    def updateADCAttenSpinBox(self, adcAtten):
        self.spinbox_adcAtten.setValue(adcAtten)
        self.changedSetting('adcatten', self.spinbox_adcAtten.value())

    def changeADCAtten(self, ):
        """
        This function executes when change the adc attenuation spinbox
        It tells the ADC attenuator to change
        
        Works similiarly to phaseSnapShot()
        """
        adcAtten = self.spinbox_adcAtten.value()
        if adcAtten != self.config.roaches.get('r{}.adcatten'.format(self.roachNum)):
            self.changedSetting('adcatten', self.spinbox_adcAtten.value())
            QtCore.QMetaObject.invokeMethod(self.roach, 'loadADCAtten', Qt.QueuedConnection,
                                            QtCore.Q_ARG(float, adcAtten))
            self.adcAttenChanged.emit()
            self.resetRoach.emit(RoachStateMachine.ROTATE)
            self.resetRoach.emit(RoachStateMachine.SWEEP)

    def changeDACAtten(self):
        """
        This function executes when we modify the dac atten start spinbox
        If we're keeping the resonators at a fixed atten, then we need to redefine the LUTs (which sets the DAC atten)
        Otherwise, we need to directly change the DAC atten
        INPUTS:
            dacAtten - the dac attenuation
        """
        dacAtten = self.spinbox_dacAttenStart.value()
        if dacAtten != self.config.roaches.get('r{}.dacatten_start'.format(self.roachNum)):
            self.changedSetting('dacatten_start', self.spinbox_dacAttenStart.value())
            if self.checkbox_resAttenFixed.isChecked():
                self.resetRoach.emit(RoachStateMachine.DEFINEDACLUT)
            else:
                QtCore.QMetaObject.invokeMethod(self.roach, 'loadDACAtten', Qt.QueuedConnection,
                                                QtCore.Q_ARG(float, dacAtten))
                self.dacAttenChanged.emit()  # starts the roach thread
                attens = self.roach.roachController.attenList
                attens = attens + (dacAtten - self.dacAtten)
                self.roach.roachController.attenList = attens  # force this to update
                self.writeNewFreqFile(attens=attens)
                self.resetRoach.emit(RoachStateMachine.ROTATE)
                self.resetRoach.emit(RoachStateMachine.SWEEP)
                self.initFreqs(False)
            self.dacAtten = dacAtten

    def toggleResAttenFixed(self):
        """ This function executes when you check/uncheck the keep res atten fixed box """
        if self.checkbox_resAttenFixed.isChecked():
            self.checkbox_resAttenFixed.setText('Keep Res Atten Fixed')
        else:
            dacAtten = self.config.roaches.get('r{}.dacatten_start'.format(self.roachNum))
            self.checkbox_resAttenFixed.setText('Keep Res Atten Fixed (was ' + str(dacAtten) + ' dB)')

    def create_main_frame(self):
        """ Makes GUI elements on the window """
        self.main_frame = QWidget()
        # self.main_frame.keyPressEvent = self.keyPressed

        self.dpi = 100
        self.fig = Figure((9.0, 5.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.canvas.mpl_connect('button_press_event', self.figureClicked)
        self.ax = self.fig.add_subplot(121)
        self.ax.set_xlabel('I')
        self.ax.set_ylabel('Q')
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_xlabel('Freq')
        self.ax2.set_ylabel('IQ velocity')

        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        label_channel = QLabel('Channel:')
        self.spinbox_channel = QSpinBox()  # initializes to 0
        self.spinbox_channel.setRange(0, 0)  # set the range after we read the freq file
        self.spinbox_channel.setWrapping(True)
        # self.spinbox_channel.valueChanged.connect(lambda x: self.plotData())
        # self.spinbox_channel.valueChanged.connect(lambda x: self.initFreqs(False))
        self.spinbox_channel.editingFinished.connect(self.plotData)
        self.spinbox_channel.editingFinished.connect(partial(self.initFreqs, False))

        self.label_resID = QLabel('ResID: ')
        self.label_resID.setMinimumWidth(110)
        self.label_resID.setMaximumWidth(110)

        self.label_freq = QLabel('Freq: 0 GHz')
        self.label_freq.setMinimumWidth(175)
        self.label_freq.setMaximumWidth(175)
        self.label_atten = QLabel('Atten: 0 dB')
        self.label_atten.setMinimumWidth(90)
        self.label_atten.setMaximumWidth(90)
        self.label_lofreq = QLabel('LO Freq: 0 GHz')
        self.label_lofreq.setMinimumWidth(160)
        self.label_lofreq.setMaximumWidth(160)

        label_num2Plot = QLabel('Num Plots:')
        spinbox_num2Plot = QSpinBox()
        spinbox_num2Plot.setWrapping(False)
        spinbox_num2Plot.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_num2Plot.setRange(1, self.maxDataListLength)
        spinbox_num2Plot.setValue(self.numData2Show)

        def changeNum2Plot(x): self.numData2Show = x

        spinbox_num2Plot.valueChanged.connect(changeNum2Plot)

        label_modify = QLabel('Modify Resonator - ')
        label_modifyFreq = QLabel('Freq [Hz]: ')
        self.textbox_modifyFreq = QLineEdit('')
        self.textbox_modifyFreq.setMaximumWidth(150)
        self.textbox_modifyFreq.setMinimumWidth(150)

        label_modifyAtten = QLabel('Atten: ')
        self.spinbox_modifyAtten = QSpinBox()
        self.spinbox_modifyAtten.setSuffix(' dB')
        self.spinbox_modifyAtten.setWrapping(False)
        self.spinbox_modifyAtten.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)

        self.button_modifyRes = QPushButton('Save Resonators')
        self.button_modifyRes.setEnabled(True)
        self.button_modifyRes.clicked.connect(self.saveResonator)
        # self.button_modifyRes.clicked.connect(lambda x: self.resetRoach.emit(RoachStateMachine.LOADFREQ))

        self.label_modifyFlag = QLabel('MODIFIED')
        self.label_modifyFlag.setStyleSheet('color: red')
        self.label_modifyFlag.hide()

        label_shiftAllFreqs = QLabel('[Hz]')
        self.textbox_shiftAllFreqs = QLineEdit('1.0e4.')
        self.textbox_shiftAllFreqs.setMaximumWidth(90)
        self.textbox_shiftAllFreqs.setMinimumWidth(90)
        button_shiftAllFreqs = QPushButton('Shift All Freqs')
        button_shiftAllFreqs.clicked.connect(self.shiftAllFreqs)

        button_snapFreq = QPushButton('Snap Freq')
        button_snapFreq.clicked.connect(self.snapFreq)
        button_snapAllFreqs = QPushButton('Snap All Freqs')
        button_snapAllFreqs.clicked.connect(self.snapAllFreqs)
        button_snapAllFreqs.setEnabled(True)

        dacAttenStart = self.config.roaches.get('r{}.dacatten_start'.format(self.roachNum))
        self.dacAtten = dacAttenStart
        label_dacAttenStart = QLabel('DAC atten:')
        self.spinbox_dacAttenStart = QDoubleSpinBox()
        self.spinbox_dacAttenStart.setValue(dacAttenStart)
        self.spinbox_dacAttenStart.setSuffix(' dB')
        self.spinbox_dacAttenStart.setRange(0, 31.75 * 2)  # There are 2 DAC attenuators
        self.spinbox_dacAttenStart.setToolTip("Rounded to the nearest 1/4 dB")
        self.spinbox_dacAttenStart.setSingleStep(1.)
        self.spinbox_dacAttenStart.setWrapping(False)
        self.spinbox_dacAttenStart.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)

        dacAttenStop = self.config.roaches.get('r{}.dacatten_stop'.format(self.roachNum))
        label_dacAttenStop = QLabel(' to ')
        spinbox_dacAttenStop = QDoubleSpinBox()
        spinbox_dacAttenStop.setValue(dacAttenStop)
        spinbox_dacAttenStop.setSuffix(' dB')
        spinbox_dacAttenStop.setRange(0, 31.75 * 2)
        spinbox_dacAttenStop.setToolTip("Rounded to the nearest 1/4 dB")
        spinbox_dacAttenStop.setSingleStep(1.)
        spinbox_dacAttenStop.setWrapping(False)
        spinbox_dacAttenStop.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_dacAttenStop.setEnabled(False)

        # self.spinbox_dacAttenStart.valueChanged.connect(partial(self.changedSetting,'dacatten_start'))
        self.spinbox_dacAttenStart.valueChanged.connect(spinbox_dacAttenStop.setValue)  # Automatically change value of dac atten stop when start value changes
        spinbox_dacAttenStop.valueChanged.connect(partial(self.changedSetting, 'dacatten_stop'))
        spinbox_dacAttenStop.valueChanged.connect(lambda x: spinbox_dacAttenStop.setValue(
            max(self.spinbox_dacAttenStart.value(), x)))  # Force stop value to be larger than start value

        # self.spinbox_dacAttenStart.valueChanged.connect(lambda x: self.resetRoach.emit(RoachStateMachine.DEFINEDACLUT))      # reset roach to so that the new dac atten is loaded in next time we sweep
        # self.spinbox_dacAttenStart.valueChanged.connect(self.changeDACAtten)
        self.spinbox_dacAttenStart.editingFinished.connect(self.changeDACAtten)

        adcAtten = self.config.roaches.get('r{}.adcatten'.format(self.roachNum))
        label_adcAtten = QLabel('ADC Atten:')
        self.spinbox_adcAtten = QDoubleSpinBox()
        self.spinbox_adcAtten.setValue(adcAtten)
        self.spinbox_adcAtten.setSuffix(' dB')
        self.spinbox_adcAtten.setRange(0, 63.5)
        self.spinbox_adcAtten.setToolTip("Rounded to the nearest 1/4 dB")
        self.spinbox_adcAtten.setSingleStep(1.)
        self.spinbox_adcAtten.setWrapping(False)
        self.spinbox_adcAtten.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        # self.spinbox_adcAtten.valueChanged.connect(partial(self.changedSetting,'adcatten'))
        # self.spinbox_adcAtten.editingFinished.connect(partial(self.changedSetting,'adcatten', self.spinbox_adcAtten.value()))
        # self.spinbox_adcAtten.valueChanged.connect(lambda x: self.resetRoach.emit(RoachStateMachine.SWEEP))       # reset state of roach
        # self.spinbox_adcAtten.valueChanged.connect(self.changeADCAtten)
        self.spinbox_adcAtten.editingFinished.connect(self.changeADCAtten)

        self.checkbox_resAttenFixed = QCheckBox('Keep Res Atten Fixed')
        self.checkbox_resAttenFixed.setChecked(True)
        self.checkbox_resAttenFixed.stateChanged.connect(lambda x: self.toggleResAttenFixed())

        psFile = self.config.roaches.get('r{}.powersweeproot'.format(self.roachNum))
        label_psFile = QLabel('Powersweep File:')
        textbox_psFile = QLineEdit(psFile)
        textbox_psFile.setMaximumWidth(300)
        textbox_psFile.setMinimumWidth(300)
        textbox_psFile.textChanged.connect(partial(self.changedSetting, 'powersweeproot'))


        loSpan = self.config.roaches.get('r{}.sweeplospan'.format(self.roachNum))
        label_loSpan = QLabel('LO Span [Hz]:')
        loSpan_str = "%.3e" % loSpan
        textbox_loSpan = QLineEdit(loSpan_str)
        textbox_loSpan.setMaximumWidth(90)
        textbox_loSpan.setMinimumWidth(90)
        # textbox_loSpan.textChanged.connect(partial(self.changedSetting,'sweeplospan'))     # This just saves whatever string you type in
        textbox_loSpan.textChanged.connect(lambda x: self.changedSetting('sweeplospan', float(x)))

        loStep = self.config.roaches.get('r{}.sweeplostep'.format(self.roachNum))
        label_loStep = QLabel('LO Step [Hz]:')
        loStep_str = "%.3e" % loStep
        textbox_loStep = QLineEdit(loStep_str)
        textbox_loStep.setMaximumWidth(90)
        textbox_loStep.setMinimumWidth(90)
        textbox_loStep.textChanged.connect(lambda x: self.changedSetting('sweeplostep', float(x)))

        button_sweep = QPushButton("Sweep Freqs")
        button_sweep.setEnabled(True)
        button_sweep.clicked.connect(self.sweepClicked)  # You can connect signals to more signals!

        # button_fit = QPushButton("Fit Loops")
        # button_fit.setEnabled(True)
        # button_fit.clicked.connect(self.fitClicked)
        button_rotate = QPushButton("Rotate Loops")
        button_rotate.setEnabled(True)
        button_rotate.clicked.connect(self.rotateClicked)
        button_translate = QPushButton("Translate Loops")
        button_translate.setEnabled(True)
        button_translate.clicked.connect(self.translateClicked)

        # centerBool = self.config.get('r{}.centerbool'.format(self.roachNum))
        # checkbox_center = QCheckBox('Recenter')
        # checkbox_center.setChecked(centerBool)
        # checkbox_center.stateChanged.connect(lambda x: self.changedSetting('centerbool',checkbox_center.isChecked()))

        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        vbox_plot.addWidget(self.mpl_toolbar)

        hbox_res = QHBoxLayout()
        hbox_res.addWidget(label_channel)
        hbox_res.addWidget(self.spinbox_channel)
        hbox_res.addWidget(self.label_resID)
        hbox_res.addSpacing(5)
        hbox_res.addWidget(self.label_freq)
        hbox_res.addWidget(self.label_atten)
        hbox_res.addWidget(self.label_lofreq)
        hbox_res.addSpacing(5)
        hbox_res.addWidget(label_num2Plot)
        hbox_res.addWidget(spinbox_num2Plot)
        hbox_res.addStretch()

        hbox_modifyRes = QHBoxLayout()
        hbox_modifyRes.addWidget(label_modify)
        hbox_modifyRes.addWidget(label_modifyFreq)
        hbox_modifyRes.addWidget(self.textbox_modifyFreq)
        hbox_modifyRes.addSpacing(5)
        hbox_modifyRes.addWidget(label_modifyAtten)
        hbox_modifyRes.addWidget(self.spinbox_modifyAtten)
        hbox_modifyRes.addWidget(self.button_modifyRes)
        hbox_modifyRes.addWidget(self.label_modifyFlag)
        hbox_modifyRes.addStretch()

        hbox_snapFreqs = QHBoxLayout()
        hbox_snapFreqs.addWidget(self.textbox_shiftAllFreqs)
        hbox_snapFreqs.addWidget(label_shiftAllFreqs)
        hbox_snapFreqs.addWidget(button_shiftAllFreqs)
        hbox_snapFreqs.addSpacing(10)
        hbox_snapFreqs.addWidget(button_snapFreq)
        hbox_snapFreqs.addWidget(button_snapAllFreqs)
        hbox_snapFreqs.addStretch()

        hbox_atten = QHBoxLayout()
        hbox_atten.addWidget(label_dacAttenStart)
        hbox_atten.addWidget(self.spinbox_dacAttenStart)
        hbox_atten.addWidget(label_dacAttenStop)
        hbox_atten.addWidget(spinbox_dacAttenStop)
        hbox_atten.addSpacing(30)
        hbox_atten.addWidget(label_adcAtten)
        hbox_atten.addWidget(self.spinbox_adcAtten)
        hbox_atten.addSpacing(30)
        hbox_atten.addWidget(self.checkbox_resAttenFixed)
        hbox_atten.addStretch()

        hbox_powersweep = QHBoxLayout()
        hbox_powersweep.addWidget(label_psFile)
        hbox_powersweep.addWidget(textbox_psFile)
        hbox_powersweep.addStretch()

        hbox_sweep = QHBoxLayout()
        hbox_sweep.addWidget(label_loSpan)
        hbox_sweep.addWidget(textbox_loSpan)
        hbox_sweep.addWidget(label_loStep)
        hbox_sweep.addWidget(textbox_loStep)
        hbox_sweep.addWidget(button_sweep)
        hbox_sweep.addSpacing(50)
        # hbox_sweep.addWidget(button_fit)
        hbox_sweep.addWidget(button_rotate)
        hbox_sweep.addWidget(button_translate)
        # hbox_sweep.addWidget(checkbox_center)
        hbox_sweep.addStretch()

        box = QVBoxLayout()
        box.addLayout(vbox_plot)
        box.addLayout(hbox_res)
        box.addLayout(hbox_modifyRes)
        box.addLayout(hbox_snapFreqs)
        box.addLayout(hbox_atten)
        box.addLayout(hbox_powersweep)
        box.addLayout(hbox_sweep)

        label_note = QLabel("NOTE: Changing Settings won't take effect until "
                            "you reload them into the ROACH2")
        label_note.setWordWrap(True)
        box.addSpacing(20)
        box.addWidget(label_note)

        self.main_frame.setLayout(box)
        self.setCentralWidget(self.main_frame)

    def closeEvent(self, event):
        if self._want_to_close:
            event.accept()
            self.close()
        else:
            event.ignore()
            self.hide()
