"""
Author: Alex Walter
Date: May 24, 2016

Classes for the plot windows for HighTemplar.py. 


Classes:
    RoachPlotWindow - abstract QMainWindow widget for a pop up plot window
    RoachSweepWindow - class for plotting IQ sweep
    
    WindowWorker - abstract class for performing plotting functions in seperate thread
    SweepWindowWorker - class for creating IQ Sweep plot in seperate thread



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
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar



mutex=QtCore.QMutex()

class WindowWorker(QtCore.QObject):
    """
    This is an abstract class. It does the behind the scenes work for plotting and emits a signal when done
    The functions are run in it's own thread in order to not slow down the main GUI
    
    INPUTS:
        axes - The axes object to manipulate
        threadName - Name of process
        
    SIGNALS:
        plotReady - emited when done executing self.plotData()
    """
    plotReady = QtCore.pyqtSignal()

    def __init__(self,axes,threadName='t'):
        QtCore.QObject.__init__(self)
        self.ax = axes
        self.threadName = threadName
        #self.mutex = QtCore.QMutex()
    
    @QtCore.pyqtSlot()
    @QtCore.pyqtSlot(object)
    @QtCore.pyqtSlot(object,object)
    def start(self,*args):       # This slot function is called and run in a seperate thread
        # Slot functions can't be called with keyword arguments
        mutex.lock()             # Not sure if this does anything. Trying to make sure the axes object isn't manipulated in two different threads at the same time
        self.plotData(*args)
        mutex.unlock()
        self.plotReady.emit()
    
    def plotData(self,*args):
        """
        Reimplement this function in a subclass to collect and plot data with the axes object
        """
        raise NotImplementedError('subclasses must override plotData()!')
    

class SweepWindowWorker(WindowWorker):
    """
    Extends the WindowWorker abstract class for the IQ sweep window.
    Takes the axes object from the GUI canvas and plots the correct lines on it.
    
    Calling the start() function inherited from WindowWorker automatically calls plotData() below
    
    INPUTS:
        axes - The axes object to manipulate
        
    SIGNALS:
        plotReady - emited when done executing self.plotData()
    """
    def __init__(self,axes):
        WindowWorker.__init__(self,axes=axes,threadName = "plotIQ")
        
        self.dataList = []      #Save data from sweeps in memory
        self.numData2Show = 4  #number of previous sweeps to show
        self.maxDataListLength = 10 #maximum number of sweeps to save in memory
        self.channel = 0        #resonator channel number
    
    def plotData(self,data=None,fit=False):
        if data is not None and not fit:
            self.appendSweepData(data)
        self.makePlot()
    
    def appendSweepData(self,data):
        self.dataList.append(data)
        if len(self.dataList) > self.maxDataListLength:
            self.dataList = self.dataList[-1*self.maxDataListLength:]

    def makePlot(self, **kwargs):
        #time.sleep(10)
        self.ax.clear()
        numData2Show = min(self.numData2Show,len(self.dataList))
        for i in range(numData2Show):
            time.sleep(2)
            data = self.dataList[-(i+1)]
            I=data['I']
            Q=data['Q']
            kwargs['alpha']=1. if i==0 else .6 - 0.5*(i-1)/(numData2Show-1)
            fmt = 'b.-' if i==0 else 'c.-'
            self.ax.plot(I[self.channel], Q[self.channel], fmt,**kwargs)
            #self.plotReady.emit()


        
        
        

class RoachPlotWindow(QMainWindow):
    """
    Abstract class for Plot windows for HighTemplar GUI
    
    subclasses must reimplement setupControlButtons()
    """    

    def __init__(self,roachNum,workerType,windowTitle='',parent=None):
        QMainWindow.__init__(self,parent=parent)
        self._want_to_close = False
        self.roachNum = roachNum
        self.setWindowTitle('r'+str(roachNum)+': '+windowTitle)
        
        self.create_main_frame()
        self.setUpWorkerThread(workerType)
        
        self.counter = 0

    
    def setUpWorkerThread(self, workerType):
        self.worker = workerType(self.ax)
        threadName = self.worker.threadName
        self.thread = QtCore.QThread(parent=self)
        self.thread.setObjectName(threadName+"_r"+str(self.roachNum))
        self.worker.plotReady.connect(self.thread.quit)
        self.worker.plotReady.connect(self.draw)
        self.worker.moveToThread(self.thread)
    

    def plotData(self,data):
        #self.thread.started.connect(partial(self.worker.start,{'data':data}))
        #print 'r'+str(self.roachNum)+' Queued data - '+str(self.counter)
        #self.thread.start()
        #time.sleep(.1)
        #self.thread.started.disconnect()
    
    
        self.thread.start()
        self.counter+=1
        print 'r'+str(self.roachNum)+' Queued data - '+str(self.counter)
        QtCore.QMetaObject.invokeMethod(self.worker, 'start', Qt.QueuedConnection,QtCore.Q_ARG(object, data))


    def draw(self):
        print 'r'+str(self.roachNum)+' drawing data - '+str(self.counter)
        self.canvas.draw()

        
    def create_main_frame(self):
        """
        Makes everything on the tab page
        
        
        """
        self.main_frame = QWidget()
        
        self.dpi = 100
        self.fig = Figure((9.0, 5.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.ax = self.fig.add_subplot(111)
        
        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        
        hbox = QHBoxLayout()
        self.setupControlButtons(hbox)
        
        box = QVBoxLayout()
        box.addLayout(vbox)
        box.addLayout(hbox)
        
        self.main_frame.setLayout(box)
        self.setCentralWidget(self.main_frame)
    
    def setupControlButtons(self, hbox):
        """
        This is an abstract function that needs to be overrided in a subclass
        It should setup all the specific buttons and controls needed for the plot window
        
        INPUTS:
            QHBoxLayout object to place all the control widgets
        """
        raise NotImplementedError('subclasses must override setupControlButtons()!')
    
    def closeEvent(self, event):
        if self._want_to_close:
            if self.thread.isRunning():
                self.thread.quit()
                time.sleep(.1) #make sure thread recieves quit signal
            if self.thread.isRunning():
                #self.thread.wait()         # This can cause the GUI to hang when closing. It's fine, you just have to wait for it
                #self.thread.terminate()    # This can throw an error but its fine
                pass                        # Seems to just abandon the threads which close by themselves
            event.accept()
            self.close()
        else:
            event.ignore()
            self.hide()
        
        
class RoachSweepWindow(RoachPlotWindow):
    
    sweepClicked = QtCore.pyqtSignal()
    
    
    def __init__(self,roachNum,parent=None):
        RoachPlotWindow.__init__(self,roachNum, SweepWindowWorker, windowTitle='IQ Plot', parent=parent)
    
    def setupControlButtons(self, hbox):
        sweep_button = QPushButton("Sweep")
        sweep_button.setEnabled(True)
        sweep_button.clicked.connect(self.sweepClicked) #You can connect signals to more signals!
        #self.connect(sweep_button,QtCore.SIGNAL('clicked()'),sweepClicked)        
        
        hbox.addWidget(sweep_button)
        
        
        
        
        
