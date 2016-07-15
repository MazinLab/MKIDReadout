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
import time, warnings, traceback
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
        self.oldPhaseNoiseDataList = []
        self.thresholdDataList = []

        
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
        if len(self.oldPhaseNoiseDataList)!=len(freqs):
            self.oldPhaseNoiseDataList = [None]*len(freqs)    
    
    def initThresh(self):
        """
        After we've loaded the thresholds we can show them
        Also everytime we change the channel we need to replot
        """
        try:
            ch=self.spinbox_channel.value()
            #thresh = self.roach.roachController.thresholds[ch]
            thresh = self.thresholdDataList[ch]
            self.label_thresh.setText('Threshold: '+str(thresh)+' deg')
            self.plotSnap()
        except:
            pass
    
    def appendThresh(self, thresholds):
        '''
        This function is called whenever the roach finishes a load threshold command
        
        INPUTS:
            thresholds - [nFreqs, 1] list of thresholds for each channel in radians
        '''
        self.thresholdDataList = thresholds
        self.initThresh()
    
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
        print 'here'
        fftlen = self.config.getint('Roach '+str(self.roachNum),'nLongsnapFftSamples')
        nFftAvg = int(np.floor(len(data)/fftlen))
        noiseData = np.zeros(fftlen)
        
        data = np.reshape(data[:nFftAvg*fftlen],(nFftAvg,fftlen))
        noiseData=np.fft.rfft(data)
        noiseData=np.abs(noiseData)**2  #power spectrum
        noiseData = 1.*np.average(noiseData,axis=0)/fftlen
        
        if self.phaseNoiseDataList[ch] is not None: self.oldPhaseNoiseDataList[ch] = np.copy(self.phaseNoiseDataList[ch])
        self.phaseNoiseDataList[ch]=noiseData
    
    def makePhaseNoisePlot(self, **kwargs):
        ch = self.spinbox_channel.value()
        #self.ax1.clear()
        
        if self.oldPhaseNoiseDataList[ch] is not None:
            ydata = np.copy(self.oldPhaseNoiseDataList[ch])
            print yData
            dt = 1.*self.roach.roachController.params['nChannelsPerStream']/self.roach.roachController.params['fpgaClockRate']
            x = np.fft.fftfreq(len(ydata),dt)
            self.line2.set_data(x,ydata)
            self.ax1.relim()
            self.ax1.autoscale_view(True,True,True)
        else:
            self.line2.set_data([],[])
        
        if self.phaseNoiseDataList[ch] is not None:
            ydata = np.copy(self.phaseNoiseDataList[ch])
            print 'phsedata: ',str(ydata)
            #dt = 2.^8/(250e.6)
            dt = 1.*self.roach.roachController.params['nChannelsPerStream']/self.roach.roachController.params['fpgaClockRate']
            x = np.fft.fftfreq(len(ydata),dt)
            #fmt='b.-'
            #self.ax1.loglog(x,data)
            self.line1.set_data(x,ydata)
            self.ax1.relim()
            self.ax1.autoscale_view(True,True,True)
            #self.ax1.set_xscale("log")
        else:
            self.line1.set_data([],[])    
    
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
            snapDict = self.snapDataList[ch]
            t = np.asarray(snapDict['time'])*1.e6
            data=np.asarray(snapDict['phase'])
            #print snapDict['trig']
            #print np.where(np.asarray(snapDict['time']))
            #print np.where(np.asarray(snapDict['time'])>0)
            trig=np.asarray(snapDict['trig'])
            data*=180./np.pi
            fmt = 'b.-'
            self.ax2.plot(t, data, fmt,**kwargs)
            print 'nPhotons: ',np.sum(trig)
            self.ax2.plot(t[np.where(trig)], data[np.where(trig)], 'ro')
            median=np.median(data)
            self.label_median.setText('Median: '+str(median)+' deg')
            self.ax2.axhline(y=median,color='k')
            try:
                #thresh = self.roach.roachController.thresholds[ch]*180./np.pi
                thresh = self.thresholdDataList[ch]
                thresh*=180./np.pi
                #self.ax2.axhline(y=median-thresh,color='r')
                self.ax2.axhline(y=median+thresh,color='r')
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
        self.line1, = self.ax1.loglog([],[],color='blue')
        self.line2, = self.ax1.loglog([],[],color='cyan')
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
        hbox_ch.addWidget(self.label_freq)
        hbox_ch.addWidget(self.label_thresh)
        hbox_ch.addWidget(self.label_median)
        hbox_ch.addWidget(button_snapPhase)
        hbox_ch.addStretch()
        
        hbox_thresh = QHBoxLayout()
        hbox_thresh.addWidget(spinbox_numSnapsThresh)
        hbox_thresh.addWidget(spinbox_threshSigs)
        hbox_thresh.addWidget(button_loadThresh)
        hbox_thresh.addStretch()
        
        hbox_phaseTimestream = QHBoxLayout()
        hbox_phaseTimestream.addWidget(spinbox_longSnapTime)
        hbox_phaseTimestream.addWidget(button_longSnap)
        hbox_phaseTimestream.addStretch()
        
        vbox1 = QVBoxLayout()
        vbox1.addLayout(vbox_plot)
        vbox1.addLayout(hbox_ch)
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
    #fitClicked = QtCore.pyqtSignal()
    rotateClicked = QtCore.pyqtSignal()
    translateClicked = QtCore.pyqtSignal()
    adcAttenChanged = QtCore.pyqtSignal()
    dacAttenChanged = QtCore.pyqtSignal()
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
        
        self.channelsModified=set()     # Remember which channels we've already modified but haven't reloaded into roach
        self.dataList = []              # Save data from sweeps and translates in memory
        self.rotatedList = []           # Save data from rotate corresponding to data in dataList
        self.numData2Show = 4           # number of previous sweeps to show
        self.maxDataListLength = 10     # maximum number of sweeps to save in memory

        self.create_main_frame()
        #self.counter = 0

    
    def plotData(self,data=None,rotated=False,**kwargs):
        #print 'Plotting Data: ',data
        if data is not None and not rotated:
            self.appendData(data.copy())
        if rotated:
            self.appendRotated(data.copy())
        if self.isVisible():
            self.makePlot(**kwargs)
            self.draw()
    
    def appendData(self,data):
        self.dataList.append(data)
        self.rotatedList.append(None)
        if len(self.dataList) > self.maxDataListLength:
            self.dataList = self.dataList[-1*self.maxDataListLength:]
            self.rotatedList = self.rotatedList[-1*self.maxDataListLength:]
    
    def appendRotated(self,data):
        self.rotatedList[-1] = data

    def makePlot(self, **kwargs):
        print "Making sweep plot"
        self.ax.clear()
        self.ax2.clear()
        numData2Show = min(self.numData2Show,len(self.dataList))
        ch = self.spinbox_channel.value()
        #for i in range(numData2Show):
        for i in range(len(self.dataList) - numData2Show, len(self.dataList)):
            print 'i:',i
            data = self.dataList[i]
            I=data['I']
            Q=data['Q']
            #kwargs['alpha']=1. if i==0 else .6 - 0.5*(i-1)/(numData2Show-1)
            kwargs['alpha'] = 1. if i==len(self.dataList)-1 else .6 - .5*(len(self.dataList)-i-1)/numData2Show
            fmt = 'b.-' if i==len(self.dataList)-1 else 'c.-'
            self.ax.plot(I[ch], Q[ch], fmt,**kwargs)
            center = data['centers'][ch]
            print 'center1 ',center
            self.ax.plot(center[0],center[1],'gx',alpha=kwargs['alpha'])
            iOnRes = data['IonRes']
            qOnRes = data['QonRes']
            self.ax.plot(iOnRes[ch],qOnRes[ch],'g.',alpha=kwargs['alpha'])

            #resFreq = self.roach.roachController.freqList[ch]
            #loSpan = self.config.getfloat('Roach '+str(self.roachNum),'sweeplospan')
            #nSteps = len(I[ch])
            #loStep = self.config.getfloat('Roach '+str(self.roachNum),'sweeplostep')
            #freqs = np.linspace(resFreq-loSpan/2., resFreq+loSpan/2., nSteps)
            try:
                freqs = data['freqOffsets'] + self.roach.roachController.freqList[ch]
                vel = np.sqrt((I[ch][1:] - I[ch][:-1])**2 + (Q[ch][1:] - Q[ch][:-1])**2)
                freqs = freqs[:-1]+(freqs[1]-freqs[0])/2.
                self.ax2.plot(freqs,vel,fmt,alpha=kwargs['alpha'])
                #self.ax2.semilogy(freqs, np.sqrt(I[ch]**2 + Q[ch]**2),fmt,alpha=kwargs['alpha'])
            except:
                print 'Couldn\'t make transmission log plot'
                traceback.print_exc()
            self.ax2.axvline(x=self.roach.roachController.freqList[ch],color='r')
            
            if self.rotatedList[i] is not None:
                iOnRes2 = self.rotatedList[i]['IonRes'][ch]
                qOnRes2 = self.rotatedList[i]['QonRes'][ch]
                center2 = np.copy(center)     # needs local copy, not pointer
                print 'center2 ',center2
                avgI = np.average(iOnRes2)
                avgQ = np.average(qOnRes2)
                rotation = self.rotatedList[i]['rotation']
                print 'Rotated ch'+str(ch)+' '+str(-1*rotation[ch]*180./np.pi)+' deg'
                
                self.ax.plot(iOnRes2,qOnRes2,'r.',alpha=kwargs['alpha'])
                self.ax.plot([center2[0],avgI],[center2[1],avgQ],'r--',alpha=kwargs['alpha'])
                self.ax.plot([center2[0],avgI],[center2[1],center2[1]],'r--',alpha=kwargs['alpha'])
                
                #self.ax.plot(centers[ch][0],centers[ch][1],'rx',**kwargs)
                #self.ax.plot(iqOnRes[ch][0]+centers[ch][0],iqOnRes[ch][1]+centers[ch][1],'ro',**kwargs)
                #print 'channel center plot',ch,centers[ch][0],centers[ch][1]
            
        self.ax.set_xlabel('I')
        self.ax.set_ylabel('Q')
        self.ax2.set_xlabel('Freqs [Hz]')
        self.ax2.set_ylabel('Transmission')
        
        
        
    
    def draw(self):
        #print 'r'+str(self.roachNum)+' drawing data - '+str(self.counter)
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
        ch=self.spinbox_channel.value()
        if fromRoach:
            self.channelsModified=set()
        if ch in self.channelsModified:
            self.label_modifyFlag.show()
        else:
            self.label_modifyFlag.hide()
        
        freqs = self.roach.roachController.freqList
        attens = self.roach.roachController.attenList
        try: lofreq = self.roach.roachController.LOFreq
        except AttributeError: lofreq=0
        
        self.spinbox_channel.setRange(0,len(freqs)-1)
        self.label_freq.setText('Freq: '+str(freqs[ch]/1.e9)+' GHz')
        self.label_atten.setText('Atten: '+str(attens[ch])+' dB')
        self.label_lofreq.setText('LO Freq: '+str(lofreq/1.e9)+' GHz')
        
        self.textbox_modifyFreq.setText("%.9e" % freqs[ch])
        self.spinbox_modifyAtten.setValue(int(attens[ch]))
    
    def saveResonator(self):
        freqs = self.roach.roachController.freqList
        attens = self.roach.roachController.attenList
        ch=self.spinbox_channel.value()

        newFreq = float(self.textbox_modifyFreq.text())
        newAtten = self.spinbox_modifyAtten.value()
        if newFreq!=freqs[ch]: self.resetRoach.emit(RoachStateMachine.LOADFREQ)
        elif newAtten!= attens[ch]: self.resetRoach.emit(RoachStateMachine.DEFINEDACLUT)
        
        freqs[ch] = newFreq     # This changes it in the Roach2Control object as well. That's what we want
        attens[ch] = newAtten
        self.channelsModified = self.channelsModified | set([ch])
        
        self.writeNewFreqFile(freqs, attens)
        self.label_modifyFlag.show()
    
    def writeNewFreqFile(self, freqs=None, attens=None):
        if freqs is None:
            freqs = np.copy(self.roach.roachController.freqList)
        if attens is None:
            attens = np.copy(self.roach.roachController.attenList)
        
        keepRes = np.where(attens < 99)     # remove any resonators with atten=99
        nFreqs = len(freqs)
        freqs = freqs[keepRes]
        attens=attens[keepRes]
        if len(freqs) != nFreqs:
            self.resetRoach.emit(RoachStateMachine.LOADFREQ)
        freqFile = self.config.get('Roach '+str(self.roachNum),'freqfile')
        newFreqFile=freqFile.rsplit('.',1)[0]+str('_NEW.')+freqFile.rsplit('.',1)[1]
        data=np.transpose([freqs,attens])
        print "Saving "+newFreqFile
        np.savetxt(newFreqFile,data,['%15.8e', '%4i'])
    
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
        
    def keyPressed(self, event):
        """
        This function is called when a key is pressed on the central widget
        Only works when in focus :-/
        
        INPUTS:
            event - QKeyEvent
        """
        if event.key() == Qt.Key_Left or event.key() == Qt.Key_A:
            print "decrease channel from keyboard"
        elif event.key() == Qt.Key_Right or event.key() == Qt.Key_D:
            print "increase channel from keyboard"
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
                try: self.clickLine.remove()
                except: pass
                self.clickLine = self.ax2.axvline(x=clickFreq,color='g')
                
                #plot point in IQ plane by estimating IQ between points
                ch=self.spinbox_channel.value()
                data = self.dataList[-1]
                I=data['I'][ch]
                Q=data['Q'][ch]
                freqs = data['freqOffsets'] + self.roach.roachController.freqList[ch]
                arg_below = np.searchsorted(freqs, clickFreq, side='left')
                arg_above = np.searchsorted(freqs, clickFreq, side='right')
                i_click = (I[arg_above]+I[arg_below])/2.
                q_click = (Q[arg_above]+Q[arg_below])/2.
                try: self.clickPoint.remove()
                except: pass
                self.clickPoint, = self.ax.plot(i_click, q_click, 'go')

                self.draw()
        #print event.inaxes
        #print self.ax
        #print self.ax2
        #print event.xdata
    
    def changeADCAtten(self, adcAtten):
        """
        This function executes when change the adc attenuation spinbox
        It tells the ADC attenuator to change
        
        Works similiarly to phaseSnapShot()
        """
        QtCore.QMetaObject.invokeMethod(self.roach, 'loadADCAtten', Qt.QueuedConnection,
                                        QtCore.Q_ARG(float, adcAtten))
        self.adcAttenChanged.emit()
        self.resetRoach.emit(RoachStateMachine.SWEEP)
    
    def changeDACAtten(self, dacAtten):
        """
        This function executes when we modify the dac atten start spinbox
        If we're keeping the resonators at a fixed atten, then we need to redefine the LUTs (which sets the DAC atten)
        Otherwise, we need to directly change the DAC atten
        INPUTS:
            dacAtten - the dac attenuation
        """
        if self.checkbox_resAttenFixed.isChecked():
            self.resetRoach.emit(RoachStateMachine.DEFINEDACLUT)
        else:
            QtCore.QMetaObject.invokeMethod(self.roach, 'loadDACAtten', Qt.QueuedConnection,
                                            QtCore.Q_ARG(float, dacAtten))
            self.dacAttenChanged.emit()
            attens = self.roach.roachController.attenList
            attens=attens+(dacAtten-self.dacAtten)
            self.roach.roachController.attenList=attens     # force this to update
            self.writeNewFreqFile(attens=attens)
            self.resetRoach.emit(RoachStateMachine.SWEEP)
            self.initFreqs(False)
        self.dacAtten=dacAtten
    
    def toggleResAttenFixed(self):
        '''
        This function executes when you check/uncheck the keep res atten fixed box
        
        '''
        if self.checkbox_resAttenFixed.isChecked():
            self.checkbox_resAttenFixed.setText('Keep Res Atten Fixed')
        else:
            
            dacAtten = self.config.getfloat('Roach '+str(self.roachNum),'dacatten_start')
            self.checkbox_resAttenFixed.setText('Keep Res Atten Fixed (was '+str(dacAtten)+' dB)')
    
    def create_main_frame(self):
        """
        Makes GUI elements on the window
        
        
        """
        self.main_frame = QWidget()
        #self.main_frame.keyPressEvent = self.keyPressed
        
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
        self.ax2.set_ylabel('Transmission')
        
        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)


        label_channel = QLabel('Channel:')
        self.spinbox_channel = QSpinBox()       #initializes to 0
        self.spinbox_channel.setRange(0,0)      #set the range after we read the freq file
        self.spinbox_channel.setWrapping(True)
        #self.spinbox_channel.valueChanged.connect(self.plotData)
        self.spinbox_channel.valueChanged.connect(lambda x: self.plotData())
        self.spinbox_channel.valueChanged.connect(lambda x: self.initFreqs(False))
        
        self.label_freq = QLabel('Freq: 0 GHz')
        self.label_freq.setMinimumWidth(150)
        self.label_freq.setMaximumWidth(150)
        self.label_atten = QLabel('Atten: 0 dB')
        self.label_lofreq = QLabel('LO Freq: 0 GHz')
        
        label_num2Plot = QLabel('Num Plots:')
        spinbox_num2Plot = QSpinBox()
        spinbox_num2Plot.setWrapping(False)
        spinbox_num2Plot.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_num2Plot.setRange(1,self.maxDataListLength)
        spinbox_num2Plot.setValue(self.numData2Show)
        def changeNum2Plot(x): self.numData2Show=x
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
        #self.button_modifyRes.clicked.connect(lambda x: self.resetRoach.emit(RoachStateMachine.LOADFREQ))

        self.label_modifyFlag = QLabel('MODIFIED')
        self.label_modifyFlag.setStyleSheet('color: red')
        self.label_modifyFlag.hide()
        
        dacAttenStart = self.config.getfloat('Roach '+str(self.roachNum),'dacatten_start')
        self.dacAtten=dacAttenStart
        label_dacAttenStart = QLabel('DAC atten:')
        spinbox_dacAttenStart = QDoubleSpinBox()
        spinbox_dacAttenStart.setValue(dacAttenStart)
        spinbox_dacAttenStart.setSuffix(' dB')
        spinbox_dacAttenStart.setRange(0,31.75*2)   # There are 2 DAC attenuators
        spinbox_dacAttenStart.setToolTip("Rounded to the nearest 1/4 dB")
        spinbox_dacAttenStart.setSingleStep(1.)
        spinbox_dacAttenStart.setWrapping(False)
        spinbox_dacAttenStart.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        
        dacAttenStop = self.config.getfloat('Roach '+str(self.roachNum),'dacatten_stop')
        label_dacAttenStop = QLabel(' to ')
        spinbox_dacAttenStop = QDoubleSpinBox()
        spinbox_dacAttenStop.setValue(dacAttenStop)
        spinbox_dacAttenStop.setSuffix(' dB')
        spinbox_dacAttenStop.setRange(0,31.75*2)
        spinbox_dacAttenStop.setToolTip("Rounded to the nearest 1/4 dB")
        spinbox_dacAttenStop.setSingleStep(1.)
        spinbox_dacAttenStop.setWrapping(False)
        spinbox_dacAttenStop.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        
        spinbox_dacAttenStart.valueChanged.connect(partial(self.changedSetting,'dacatten_start'))
        spinbox_dacAttenStart.valueChanged.connect(spinbox_dacAttenStop.setValue)                   #Automatically change value of dac atten stop when start value changes
        spinbox_dacAttenStop.valueChanged.connect(partial(self.changedSetting,'dacatten_stop'))
        spinbox_dacAttenStop.valueChanged.connect(lambda x: spinbox_dacAttenStop.setValue(max(spinbox_dacAttenStart.value(),x)))       #Force stop value to be larger than start value
        
        #spinbox_dacAttenStart.valueChanged.connect(lambda x: self.resetRoach.emit(RoachStateMachine.DEFINEDACLUT))      # reset roach to so that the new dac atten is loaded in next time we sweep
        spinbox_dacAttenStart.valueChanged.connect(self.changeDACAtten)
        
        
        adcAtten = self.config.getfloat('Roach '+str(self.roachNum),'adcatten')
        label_adcAtten = QLabel('ADC Atten:')
        spinbox_adcAtten = QDoubleSpinBox()
        spinbox_adcAtten.setValue(adcAtten)
        spinbox_adcAtten.setSuffix(' dB')
        spinbox_adcAtten.setRange(0,31.75)
        spinbox_adcAtten.setToolTip("Rounded to the nearest 1/4 dB")
        spinbox_adcAtten.setSingleStep(1.)
        spinbox_adcAtten.setWrapping(False)
        spinbox_adcAtten.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_adcAtten.valueChanged.connect(partial(self.changedSetting,'adcatten'))
        #spinbox_adcAtten.valueChanged.connect(lambda x: self.resetRoach.emit(RoachStateMachine.SWEEP))       # reset state of roach
        spinbox_adcAtten.valueChanged.connect(self.changeADCAtten)
        
        
        self.checkbox_resAttenFixed = QCheckBox('Keep Res Atten Fixed')
        self.checkbox_resAttenFixed.setChecked(True)
        self.checkbox_resAttenFixed.stateChanged.connect(lambda x: self.toggleResAttenFixed())
        
        psFile = self.config.get('Roach '+str(self.roachNum),'powersweepfile')
        label_psFile = QLabel('Powersweep File:')
        textbox_psFile = QLineEdit(psFile)
        textbox_psFile.setMaximumWidth(300)
        textbox_psFile.setMinimumWidth(300)
        textbox_psFile.textChanged.connect(partial(self.changedSetting,'powersweepfile'))
        
        
        loSpan = self.config.getfloat('Roach '+str(self.roachNum),'sweeplospan')
        label_loSpan = QLabel('LO Span [Hz]:')
        loSpan_str = "%.3e" % loSpan
        textbox_loSpan = QLineEdit(loSpan_str)
        textbox_loSpan.setMaximumWidth(90)
        textbox_loSpan.setMinimumWidth(90)
        #textbox_loSpan.textChanged.connect(partial(self.changedSetting,'sweeplospan'))     # This just saves whatever string you type in
        textbox_loSpan.textChanged.connect(lambda x: self.changedSetting('sweeplospan',"%.9e" % float(x)))
        
        loStep = self.config.getfloat('Roach '+str(self.roachNum),'sweeplostep')
        label_loStep = QLabel('LO Step [Hz]:')
        loStep_str = "%.3e" % loStep
        textbox_loStep = QLineEdit(loStep_str)
        textbox_loStep.setMaximumWidth(90)
        textbox_loStep.setMinimumWidth(90)
        textbox_loStep.textChanged.connect(lambda x: self.changedSetting('sweeplostep',"%.9e" % float(x)))
        
        

        button_sweep = QPushButton("Sweep Freqs")
        button_sweep.setEnabled(True)
        button_sweep.clicked.connect(self.sweepClicked) #You can connect signals to more signals!   
        
        #button_fit = QPushButton("Fit Loops")
        #button_fit.setEnabled(True)
        #button_fit.clicked.connect(self.fitClicked)
        button_rotate = QPushButton("Rotate Loops")
        button_rotate.setEnabled(True)
        button_rotate.clicked.connect(self.rotateClicked)
        button_translate = QPushButton("Translate Loops")
        button_translate.setEnabled(True)
        button_translate.clicked.connect(self.translateClicked)
        
        '''
        centerBool = self.config.getboolean('Roach '+str(self.roachNum),'centerbool')
        checkbox_center = QCheckBox('Recenter')
        checkbox_center.setChecked(centerBool)
        #checkbox_center.stateChanged.connect(partial(self.changedSetting,'centerbool'))    # This has some weird tristate int
        checkbox_center.stateChanged.connect(lambda x: self.changedSetting('centerbool',checkbox_center.isChecked()))
        '''
        
        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        vbox_plot.addWidget(self.mpl_toolbar)
        
        hbox_res = QHBoxLayout()
        hbox_res.addWidget(label_channel)
        hbox_res.addWidget(self.spinbox_channel)
        hbox_res.addSpacing(20)
        hbox_res.addWidget(self.label_freq)
        hbox_res.addWidget(self.label_atten)
        hbox_res.addWidget(self.label_lofreq)
        hbox_res.addSpacing(20)
        hbox_res.addWidget(label_num2Plot)
        hbox_res.addWidget(spinbox_num2Plot)
        hbox_res.addStretch()
        
        hbox_modifyRes = QHBoxLayout()
        hbox_modifyRes.addWidget(label_modify)
        hbox_modifyRes.addWidget(label_modifyFreq)
        hbox_modifyRes.addWidget(self.textbox_modifyFreq)
        hbox_modifyRes.addWidget(label_modifyAtten)
        hbox_modifyRes.addWidget(self.spinbox_modifyAtten)
        hbox_modifyRes.addWidget(self.button_modifyRes)
        hbox_modifyRes.addWidget(self.label_modifyFlag)
        hbox_modifyRes.addStretch()
        
        hbox_atten = QHBoxLayout()
        hbox_atten.addWidget(label_dacAttenStart)
        hbox_atten.addWidget(spinbox_dacAttenStart)
        hbox_atten.addWidget(label_dacAttenStop)
        hbox_atten.addWidget(spinbox_dacAttenStop)
        hbox_atten.addSpacing(30)
        hbox_atten.addWidget(label_adcAtten)
        hbox_atten.addWidget(spinbox_adcAtten)
        hbox_atten.addSpacing(30)
        hbox_atten.addWidget(self.checkbox_resAttenFixed)
        hbox_atten.addStretch()
        
        hbox_sweep = QHBoxLayout()
        hbox_sweep.addWidget(label_loSpan)
        hbox_sweep.addWidget(textbox_loSpan)
        hbox_sweep.addWidget(label_loStep)
        hbox_sweep.addWidget(textbox_loStep)
        hbox_sweep.addWidget(button_sweep)
        hbox_sweep.addSpacing(50)
        #hbox_sweep.addWidget(button_fit)
        hbox_sweep.addWidget(button_rotate)
        hbox_sweep.addWidget(button_translate)
        #hbox_sweep.addWidget(checkbox_center)
        hbox_sweep.addStretch()
        
        box = QVBoxLayout()
        box.addLayout(vbox_plot)
        box.addLayout(hbox_res)
        box.addLayout(hbox_modifyRes)
        box.addLayout(hbox_atten)
        box.addWidget(textbox_psFile)
        box.addLayout(hbox_sweep)
        
        label_note = QLabel("NOTE: Changing Settings won't take effect until you reload them into the ROACH2")
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
        
    
        
