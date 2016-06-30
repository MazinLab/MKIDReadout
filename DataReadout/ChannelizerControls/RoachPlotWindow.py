"""
Author: Alex Walter
Date: May 24, 2016

Classes for the plot windows for HighTemplar.py. Sweep or phase timestream


Classes:
    RoachSweepWindow - class for plotting IQ sweep
    SweepWindowWorker - class for creating IQ Sweep plot in seperate thread

Abstract Classes:
    RoachPlotWindow - abstract QMainWindow widget for a pop up plot window
    WindowWorker - abstract class for performing plotting functions in seperate thread



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
except ImportError:
	from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
        
class RoachSweepWindow(QMainWindow):
    
    sweepClicked = QtCore.pyqtSignal()
    
    
    def __init__(self,roach,config,parent=None):
        """
        Window for showing IQ plot of resonators
        
        INPUTS:
            roach - A RoachStateMachine Object. We need this to access all the relevent settings
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

        self.counter = 0

    
    def plotData(self,data=None,fit=False,**kwargs):
        #print 'Plotting Data: ',data
        if data is not None and not fit:
            self.appendSweepData(data)
        if fit:
            pass # plot center and angle from fit!
        if self.isVisible():
            self.makePlot(**kwargs)
            self.draw()
    
    def appendSweepData(self,data):
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
        self.label_freq.setMinimumWidth(120)
        self.label_freq.setMaximumWidth(120)
        self.label_atten = QLabel('Atten: 0 dB')
        self.label_lofreq = QLabel('LO Freq: 0 GHz')
        

        button_sweep = QPushButton("Sweep Freq")
        button_sweep.setEnabled(True)
        button_sweep.clicked.connect(self.sweepClicked) #You can connect signals to more signals!
        #self.connect(sweep_button,QtCore.SIGNAL('clicked()'),sweepClicked)        
        
        
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
        
        box = QVBoxLayout()
        box.addLayout(hbox1)
        box.addWidget(button_sweep)
        
        self.main_frame.setLayout(box)
        self.setCentralWidget(self.main_frame)
    
    def closeEvent(self, event):
        if self._want_to_close:
            event.accept()
            self.close()
        else:
            event.ignore()
            self.hide()
        
    
        
