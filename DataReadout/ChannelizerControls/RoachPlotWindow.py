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
import time, warnings
import ConfigParser
from functools import partial
from PyQt4.QtGui import *
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
try:
	from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
except ImportError: #Named changed in some newer matplotlib versions
	from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from RoachStateMachine import RoachStateMachine

class RoachPhaseStreamWindow(QMainWindow):

    thresholdClicked = QtCore.pyqtSignal()
    phaseSnapClicked = QtCore.pyqtSignal()
    phaseTimestreamClicked = QtCore.pyqtSignal()
    resetRoach = QtCore.pyqtSignal(int)
    
    def __init__(self,roach,config,parent=None):
        """
        Window for showing snapshot of phase timestream of resonators
        
        INPUTS:
            roach - A RoachStateMachine Object. We need this to access all the relevent settings
            config - ConfigParser object
            parent - Leave as default
        """
        QMainWindow.__init__(self,parent=parent)
        self._want_to_close = False
        self.roach=roach
        self.roachNum = self.roach.num
        self.config = config
        self.setWindowTitle('r'+str(self.roachNum)+': Phase Timestream')
        
        self.create_main_frame()
        
        self.snapDataList = []
        self.phaseNoiseDataList = []

        
        self.roach.snapPhase.connect(self.plotSnap)
        self.roach.timestreamPhase.connect(self.plotPhaseNoise)
    
    def initFreqs(self):
        """
        After we've loaded the frequency file in RoachStateMachine object then we can initialize some GUI elements
        """
        freqs = self.roach.roachController.freqList
        ch=self.spinbox_channel.value()
        
        self.spinbox_channel.setRange(0,len(freqs)-1)
        self.label_freq.setText('Freq: '+str(freqs[ch]/1.e9)+' GHz')
        
        if len(self.snapDataList)!=len(freqs):
            self.snapDataList = [None]*len(freqs)
        if len(self.phaseNoiseDataList)!=len(freqs):
            self.phaseNoiseDataList = [None]*len(freqs)
    
    def initThresh(self):
        """
        After we've loaded the thresholds we can show them
        """
        try:
            ch=self.spinbox_channel.value()
            thresh = self.roach.roachController.thresholds[ch]
            self.label_thresh.setText('Threshold: '+str(thresh)+' deg')
            self.plotSnap()
        except AttributeError:
            pass
    
    def plotPhaseNoise(self,ch=None, data=None,**kwargs):
        #self.spinbox_channel.setEnabled(False)
        currentCh = self.spinbox_channel.value()
        if ch is None: ch=currentCh
        if data is not None:
            self.appendPhaseNoiseData(ch,data)
        if self.isVisible() and ch==currentCh:
            self.makePhaseNoisePlot(**kwargs)
            self.draw()
        #self.spinbox_channel.setEnabled(True)
    
    def appendPhaseNoiseData(self, ch, data):
        fftlen = self.config.getint('Roach '+str(self.roachNum),'nLongsnapFftSamples')
        nFftAvg = int(np.floor(len(data)/fftlen))
        noiseData = np.zeros(fftlen)
        
        data = np.reshape(data[:nFftAvg*fftlen],(nFftAvg,fftlen))
        noiseData=np.fft.rfft(data)
        noiseData=np.abs(noiseData)**2  #power spectrum
        noiseData = np.average(noiseData,axis=0)
        
        self.phaseNoiseDataList[ch]=noiseData
    
    def makePhaseNoisePlot(self, **kwargs):
        ch = self.spinbox_channel.value()
        #self.ax1.clear()
        if self.phaseNoiseDataList[ch] is not None:
            ydata = np.copy(self.phaseNoiseDataList[ch])
            x = np.fft.fftfreq(len(ydata),10.**-6.)
            #fmt='b.-'
            #self.ax1.loglog(x,data)
            self.line1.set_data(x,ydata)
            self.ax1.relim()
            self.ax1.autoscale_view(True,True,True)
            #self.ax1.set_xscale("log")
        else:
            self.line1.set_data([],[])
        #self.ax1.set_ylabel('Noise Power Spectrum')
        #self.ax1.set_xlabel('f [Hz]')
    
    def phaseTimeStream(self):
        """
        This function executes when you press the collect phase timestream button.
        After running any commands that need running, it gets a new phase timestream and plots the noise
        
        Works similiarly to phaseSnapShot()
        """
        ch=self.spinbox_channel.value()
        timelen = self.config.getfloat('Roach '+str(self.roachNum),'longsnaptime')
        QtCore.QMetaObject.invokeMethod(self.roach, 'getPhaseStream', Qt.QueuedConnection,
                                        QtCore.Q_ARG(int, ch),QtCore.Q_ARG(float, timelen))
        self.phaseTimestreamClicked.emit()
        
    def plotSnap(self,ch=None, data=None,**kwargs):
        self.spinbox_channel.setEnabled(False)
        currentCh = self.spinbox_channel.value()
        if ch is None: ch=currentCh
        if data is not None:
            self.appendSnapData(ch,data)
        if self.isVisible() and ch==currentCh:
            self.makeSnapPlot(**kwargs)
            self.draw()
        self.spinbox_channel.setEnabled(True)
    
    def appendSnapData(self,ch,data):
        self.snapDataList[ch]=data

    
    def makeSnapPlot(self,**kwargs):
        ch = self.spinbox_channel.value()
        self.ax2.clear()
        if self.snapDataList[ch] is not None:
            data = np.copy(self.snapDataList[ch])
            data*=180./np.pi
            fmt = 'b.-'
            self.ax2.plot(data, fmt,**kwargs)
            median=np.median(data)
            self.label_median.setText('Median: '+str(median)+' deg')
            self.ax2.axhline(y=median,color='k')
            try:
                thresh = self.roach.roachController.thresholds[ch]*180./np.pi
                self.ax2.axhline(y=median-thresh,color='r')
                #self.ax2.axhline(y=median+thresh,color='r')
            except AttributeError:
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
            This window object sees that signal and updates the plot
        """
        ch=self.spinbox_channel.value()
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
        self.ax1.set_ylabel('Noise Power Spectrum')
        self.ax1.set_xlabel('f [Hz]')
        self.line1, = self.ax1.loglog([],[])
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_xlabel('Time [us]')
        self.ax2.set_ylabel('Phase [Deg]')
        
        
        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        
        
        label_channel = QLabel('Channel:')
        self.spinbox_channel = QSpinBox()       #initializes to 0
        self.spinbox_channel.setRange(0,0)      #set the range after we read the freq file
        self.spinbox_channel.setWrapping(True)
        self.spinbox_channel.valueChanged.connect(lambda x: self.plotSnap())
        self.spinbox_channel.valueChanged.connect(lambda x: self.plotPhaseNoise())
        self.spinbox_channel.valueChanged.connect(lambda x: self.initFreqs())
        self.spinbox_channel.valueChanged.connect(lambda x: self.initThresh())
        
        self.label_freq = QLabel('Freq: 0 GHz')
        self.label_freq.setMinimumWidth(150)
        self.label_freq.setMaximumWidth(150)
        self.label_thresh = QLabel('Thresh: 0 deg')
        self.label_thresh.setMinimumWidth(150)
        self.label_thresh.setMaximumWidth(150)
        self.label_median = QLabel('Median: 0 deg')
        
        button_snapPhase = QPushButton("Phase Snapshot")
        button_snapPhase.setEnabled(True)
        button_snapPhase.clicked.connect(self.phaseSnapShot)
        
        
        numSnapsThresh = self.config.getint('Roach '+str(self.roachNum),'numsnaps_thresh')
        spinbox_numSnapsThresh = QSpinBox()
        spinbox_numSnapsThresh.setValue(numSnapsThresh)
        spinbox_numSnapsThresh.setRange(1,100)
        spinbox_numSnapsThresh.setSuffix(" *2 msec")
        spinbox_numSnapsThresh.setWrapping(False)
        spinbox_numSnapsThresh.setButtonSymbols(QAbstractSpinBox.NoButtons)
        spinbox_numSnapsThresh.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_numSnapsThresh.valueChanged.connect(partial(self.changedSetting,'numsnaps_thresh'))
        
        threshSigs = self.config.getfloat('Roach '+str(self.roachNum),'numsigs_thresh')
        spinbox_threshSigs = QDoubleSpinBox()
        spinbox_threshSigs.setValue(threshSigs)
        spinbox_threshSigs.setRange(0,100)
        spinbox_threshSigs.setSuffix(" sigmas")
        spinbox_threshSigs.setWrapping(False)
        spinbox_threshSigs.setButtonSymbols(QAbstractSpinBox.NoButtons)
        spinbox_threshSigs.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_threshSigs.valueChanged.connect(partial(self.changedSetting,'numsigs_thresh'))
        spinbox_threshSigs.valueChanged.connect(lambda x: self.resetRoach.emit(RoachStateMachine.LOADTHRESHOLD))       # reset state of roach
        
        button_loadThresh = QPushButton("Load Thresholds")
        button_loadThresh.setEnabled(True)
        button_loadThresh.clicked.connect(self.thresholdClicked)
        
        longSnapTime = self.config.getfloat('Roach '+str(self.roachNum),'longsnaptime')
        spinbox_longSnapTime = QDoubleSpinBox()
        spinbox_longSnapTime.setValue(longSnapTime)
        spinbox_longSnapTime.setRange(0,1000)
        spinbox_longSnapTime.setSuffix(" seconds")
        spinbox_longSnapTime.setWrapping(False)
        spinbox_longSnapTime.setButtonSymbols(QAbstractSpinBox.NoButtons)
        spinbox_longSnapTime.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_longSnapTime.valueChanged.connect(partial(self.changedSetting,'longsnaptime'))
        
        button_longSnap = QPushButton("Collect Phase Timestream")
        button_longSnap.setEnabled(True)
        button_longSnap.clicked.connect(self.phaseTimeStream)
        
        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        vbox_plot.addWidget(self.mpl_toolbar)
        
        hbox_ch = QHBoxLayout()
        hbox_ch.addWidget(label_channel)
        hbox_ch.addWidget(self.spinbox_channel)
        
        vbox_res = QVBoxLayout()
        vbox_res.addStretch()
        vbox_res.addLayout(hbox_ch)
        vbox_res.addWidget(self.label_freq)
        vbox_res.addWidget(self.label_thresh)
        vbox_res.addWidget(self.label_median)
        vbox_res.addWidget(button_snapPhase)
        vbox_res.addStretch()
        
        hbox_thresh = QHBoxLayout()
        hbox_thresh.addWidget(spinbox_numSnapsThresh)
        hbox_thresh.addWidget(spinbox_threshSigs)
        hbox_thresh.addWidget(button_loadThresh)
        hbox_thresh.addStretch()
        
        hbox_phaseTimestream = QHBoxLayout()
        hbox_phaseTimestream.addWidget(spinbox_longSnapTime)
        hbox_phaseTimestream.addWidget(button_longSnap)
        hbox_phaseTimestream.addStretch()
        
        hbox1 = QHBoxLayout()
        hbox1.addLayout(vbox_plot)
        hbox1.addLayout(vbox_res)
        
        vbox1 = QVBoxLayout()
        vbox1.addLayout(hbox1)
        vbox1.addLayout(hbox_thresh)
        vbox1.addLayout(hbox_phaseTimestream)
        
        self.main_frame.setLayout(vbox1)
        self.setCentralWidget(self.main_frame)
    
    def draw(self):
        #print 'r'+str(self.roachNum)+' drawing data - '+str(self.counter)
        self.canvas.draw()
        self.canvas.flush_events()
    
    def changedSetting(self,settingID,setting):
        """
        When a setting is changed, reflect the change in the config object which is shared across all GUI elements.
        
        INPUTS:
            settingID - the key in the configparser
            setting - the value
        """
        self.config.set('Roach '+str(self.roachNum),settingID,str(setting))
        #If we don't force the setting value to be a string then the configparser has trouble grabbing the value later on for some unknown reason
        newSetting = self.config.get('Roach '+str(self.roachNum),settingID)
        print 'setting ',settingID,' to ',newSetting
    
    def closeEvent(self, event):
        if self._want_to_close:
            event.accept()
            self.close()
        else:
            event.ignore()
            self.hide()
    
        
class RoachSweepWindow(QMainWindow):
    
    sweepClicked = QtCore.pyqtSignal()
    fitClicked = QtCore.pyqtSignal()
    resetRoach = QtCore.pyqtSignal(int)
    
    
    def __init__(self,roach,config,parent=None):
        """
        Window for showing IQ plot of resonators
        
        INPUTS:
            roach - A RoachStateMachine Object. We need this to access all the relevent settings
            config - ConfigParser object
            parent - Leave as default
        """
        QMainWindow.__init__(self,parent=parent)
        self._want_to_close = False
        self.roach=roach
        self.roachNum = self.roach.num
        self.config = config
        self.setWindowTitle('r'+str(self.roachNum)+': IQ Plot')
        
        self.create_main_frame()

        #self.channel=None
        self.dataList = []      #Save data from sweeps in memory
        self.numData2Show = 4  #number of previous sweeps to show
        self.maxDataListLength = 10 #maximum number of sweeps to save in memory

        #self.counter = 0

    
    def plotData(self,data=None,fit=False,**kwargs):
        #print 'Plotting Data: ',data
        if data is not None and not fit:
            self.appendData(data)
        if fit:
            pass # plot center and angle from fit!
        if self.isVisible():
            self.makePlot(**kwargs)
            self.draw()
    
    def appendData(self,data):
        self.dataList.append(data)
        if len(self.dataList) > self.maxDataListLength:
            self.dataList = self.dataList[-1*self.maxDataListLength:]

    def makePlot(self, **kwargs):
        self.ax.clear()
        numData2Show = min(self.numData2Show,len(self.dataList))
        ch = self.spinbox_channel.value()
        for i in range(numData2Show):
            data = self.dataList[-(i+1)]
            I=data['I']
            Q=data['Q']
            kwargs['alpha']=1. if i==0 else .6 - 0.5*(i-1)/(numData2Show-1)
            fmt = 'b.-' if i==0 else 'c.-'
            self.ax.plot(I[ch], Q[ch], fmt,**kwargs)
        self.ax.set_xlabel('I')
        self.ax.set_ylabel('Q')
        
    
    def draw(self):
        #print 'r'+str(self.roachNum)+' drawing data - '+str(self.counter)
        self.canvas.draw()
        self.canvas.flush_events()
    
    def initFreqs(self):
        """
        After we've loaded the frequency file in RoachStateMachine object then we can initialize some GUI elements
        """
        freqs = self.roach.roachController.freqList
        attens = self.roach.roachController.attenList
        lofreq = self.roach.roachController.LOFreq
        ch=self.spinbox_channel.value()
        
        self.spinbox_channel.setRange(0,len(freqs)-1)
        self.label_freq.setText('Freq: '+str(freqs[ch]/1.e9)+' GHz')
        self.label_atten.setText('Atten: '+str(attens[ch])+' dB')
        self.label_lofreq.setText('LO Freq: '+str(lofreq/1.e9)+' GHz')
    
    def changedSetting(self,settingID,setting):
        """
        When a setting is changed, reflect the change in the config object which is shared across all GUI elements.
        
        INPUTS:
            settingID - the key in the configparser
            setting - the value
        """
        self.config.set('Roach '+str(self.roachNum),settingID,str(setting))
        #If we don't force the setting value to be a string then the configparser has trouble grabbing the value later on for some unknown reason
        newSetting = self.config.get('Roach '+str(self.roachNum),settingID)
        print 'setting ',settingID,' to ',newSetting
        
        
    
    def create_main_frame(self):
        """
        Makes GUI elements on the window
        
        
        """
        self.main_frame = QWidget()
        
        self.dpi = 100
        self.fig = Figure((9.0, 5.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('I')
        self.ax.set_ylabel('Q')
        
        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)


        label_channel = QLabel('Channel:')
        self.spinbox_channel = QSpinBox()       #initializes to 0
        self.spinbox_channel.setRange(0,0)      #set the range after we read the freq file
        self.spinbox_channel.setWrapping(True)
        #self.spinbox_channel.valueChanged.connect(self.plotData)
        self.spinbox_channel.valueChanged.connect(lambda x: self.plotData())
        self.spinbox_channel.valueChanged.connect(lambda x: self.initFreqs())
        
        self.label_freq = QLabel('Freq: 0 GHz')
        self.label_freq.setMinimumWidth(150)
        self.label_freq.setMaximumWidth(150)
        self.label_atten = QLabel('Atten: 0 dB')
        self.label_lofreq = QLabel('LO Freq: 0 GHz')
        
        
        
        
        dacAttenStart = self.config.getfloat('Roach '+str(self.roachNum),'dacatten_start')
        label_dacAttenStart = QLabel('DAC atten:')
        spinbox_dacAttenStart = QDoubleSpinBox()
        spinbox_dacAttenStart.setValue(dacAttenStart)
        spinbox_dacAttenStart.setSuffix(' dB')
        spinbox_dacAttenStart.setRange(0,31.75)
        spinbox_dacAttenStart.setSingleStep(1.)
        spinbox_dacAttenStart.setWrapping(False)
        spinbox_dacAttenStart.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        
        dacAttenStop = self.config.getfloat('Roach '+str(self.roachNum),'dacatten_stop')
        label_dacAttenStop = QLabel(' to ')
        spinbox_dacAttenStop = QDoubleSpinBox()
        spinbox_dacAttenStop.setValue(dacAttenStop)
        spinbox_dacAttenStop.setSuffix(' dB')
        spinbox_dacAttenStop.setRange(0,31.75)
        spinbox_dacAttenStop.setSingleStep(1.)
        spinbox_dacAttenStop.setWrapping(False)
        spinbox_dacAttenStop.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        
        spinbox_dacAttenStart.valueChanged.connect(partial(self.changedSetting,'dacatten_start'))
        spinbox_dacAttenStart.valueChanged.connect(spinbox_dacAttenStop.setValue)                   #Automatically change value of dac atten stop when start value changes
        spinbox_dacAttenStop.valueChanged.connect(partial(self.changedSetting,'dacatten_stop'))
        spinbox_dacAttenStop.valueChanged.connect(lambda x: spinbox_dacAttenStop.setValue(max(spinbox_dacAttenStart.value(),x)))       #Force stop value to be larger than start value
        
        spinbox_dacAttenStart.valueChanged.connect(lambda x: self.resetRoach.emit(RoachStateMachine.LOADFREQ))      # reset roach to so that the new dac atten is loaded in next time we sweep
        
        
        
        adcAtten = self.config.getfloat('Roach '+str(self.roachNum),'adcatten')
        label_adcAtten = QLabel('ADC Atten:')
        spinbox_adcAtten = QDoubleSpinBox()
        spinbox_adcAtten.setValue(adcAtten)
        spinbox_adcAtten.setSuffix(' dB')
        spinbox_adcAtten.setRange(0,31.75)
        spinbox_adcAtten.setSingleStep(1.)
        spinbox_adcAtten.setWrapping(False)
        spinbox_adcAtten.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_adcAtten.valueChanged.connect(partial(self.changedSetting,'adcatten'))
        spinbox_adcAtten.valueChanged.connect(lambda x: self.resetRoach.emit(RoachStateMachine.LOADFREQ))       # reset state of roach
        
        loSpan = self.config.getfloat('Roach '+str(self.roachNum),'sweeplospan')
        label_loSpan = QLabel('LO Span [Hz]:')
        loSpan_str = "%.9e" % loSpan
        textbox_loSpan = QLineEdit(loSpan_str)
        textbox_loSpan.setMaximumWidth(150)
        textbox_loSpan.setMinimumWidth(150)
        #textbox_loSpan.textChanged.connect(partial(self.changedSetting,'sweeplospan'))     # This just saves whatever string you type in
        textbox_loSpan.textChanged.connect(lambda x: self.changedSetting('sweeplospan',"%.9e" % float(x)))
        

        button_sweep = QPushButton("Sweep Freq")
        button_sweep.setEnabled(True)
        button_sweep.clicked.connect(self.sweepClicked) #You can connect signals to more signals!   
        
        button_fit = QPushButton("Fit Loops")
        button_fit.setEnabled(True)
        button_fit.clicked.connect(self.fitClicked)
        
        centerBool = self.config.getboolean('Roach '+str(self.roachNum),'centerbool')
        checkbox_center = QCheckBox('Recenter')
        checkbox_center.setChecked(centerBool)
        #checkbox_center.stateChanged.connect(partial(self.changedSetting,'centerbool'))    # This has some weird tristate int
        checkbox_center.stateChanged.connect(lambda x: self.changedSetting('centerbool',checkbox_center.isChecked()))
        
        
        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        vbox_plot.addWidget(self.mpl_toolbar)
        
        hbox_ch = QHBoxLayout()
        hbox_ch.addWidget(label_channel)
        hbox_ch.addWidget(self.spinbox_channel)
        
        vbox_res = QVBoxLayout()
        vbox_res.addStretch()
        vbox_res.addLayout(hbox_ch)
        vbox_res.addWidget(self.label_freq)
        vbox_res.addWidget(self.label_atten)
        vbox_res.addWidget(self.label_lofreq)
        vbox_res.addStretch()
        
        hbox1 = QHBoxLayout()
        hbox1.addLayout(vbox_plot)
        hbox1.addLayout(vbox_res)
        
        hbox_atten = QHBoxLayout()
        hbox_atten.addWidget(label_dacAttenStart)
        hbox_atten.addWidget(spinbox_dacAttenStart)
        hbox_atten.addWidget(label_dacAttenStop)
        hbox_atten.addWidget(spinbox_dacAttenStop)
        hbox_atten.addSpacing(50)
        hbox_atten.addWidget(label_adcAtten)
        hbox_atten.addWidget(spinbox_adcAtten)
        hbox_atten.addStretch()
        
        hbox3 = QHBoxLayout()
        hbox3.addWidget(label_loSpan)
        hbox3.addWidget(textbox_loSpan)
        hbox3.addWidget(button_sweep)
        hbox3.addSpacing(50)
        hbox3.addWidget(button_fit)
        hbox3.addWidget(checkbox_center)
        hbox3.addStretch()
        
        vbox_buttons = QVBoxLayout()
        vbox_buttons.addLayout(hbox_atten)
        vbox_buttons.addLayout(hbox3)
        
        label_note = QLabel("NOTE: Changing Settings won't take effect into you reload them into the ROACH2")
        label_note.setWordWrap(True)
        vbox_buttons.addSpacing(20)
        vbox_buttons.addWidget(label_note)
        
        box = QVBoxLayout()
        box.addLayout(hbox1)
        box.addLayout(vbox_buttons)
        
        self.main_frame.setLayout(box)
        self.setCentralWidget(self.main_frame)
    
    def closeEvent(self, event):
        if self._want_to_close:
            event.accept()
            self.close()
        else:
            event.ignore()
            self.hide()
        
    
        
