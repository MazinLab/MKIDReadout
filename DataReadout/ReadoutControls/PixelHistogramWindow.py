"""
Author:    Alex Walter
Date:      Jul 6, 2016

This class is a GUI window for plotting pixel timestreams

"""

import numpy as np
from PyQt4 import QtCore
from PyQt4 import QtGui
from PyQt4.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
try:
	from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
except ImportError: #Named changed in some newer matplotlib versions
	from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

class PixelHistogramWindow(QMainWindow):
    """
    We use this for realtime plotting with MkidDashboard
    """
    closeWindow = QtCore.pyqtSignal()
    
    def __init__(self, pixelList, parent):
        super(QMainWindow, self).__init__(parent)
        self.setWindowTitle("Histogram")
        self.parent=parent
        self.pixelList=np.asarray(pixelList,dtype=np.int)
        self.create_main_frame()
        self.plotData()
        parent.newImageProcessed.connect(self.plotData)
        
    
    def plotData(self,**kwargs):
        self.setCheckboxText(currentPix=True)
        if self.checkbox_plotPix.isChecked():
            countRateHist,bin_edges = self.getCountRateHist()
            self.line.set_data(bin_edges[:-1],countRateHist)
        else: self.line.set_data([],[])
        if self.checkbox_plotCurrentPix.isChecked():
            countRateHist,bin_edges = self.getCountRateHist(True)
            self.line.set_data(bin_edges[:-1],countRateHist)
        else: self.line2.set_data([],[])
        self.ax.relim()
        self.ax.autoscale_view(True,True,True)
        self.draw()
    
    '''
    def plotData(self, **kwargs):
        image = self.parent.imageList[-1]
        countsX = np.sum(image,axis=0)
        countsY = np.sum(image,axis=1)
        self.line.set_data(image.shape[0],countsX)
        self.line2.set_data(image.shape[1],countsY)
    '''    

    def getCountRateHist(self, forCurrentPix=False):
        imageList=self.parent.imageList
        pixList = self.pixelList
        if forCurrentPix: pixList = np.asarray([[p[0],p[1]] for p in self.parent.selectedPixels])
        if len(imageList)>0 and len(pixList)>0:
            x = pixList[:,0]
            y = pixList[:,1]
            c = np.asarray(imageList)[-1,y,x]
            #countRates = np.sum(c,axis=0)
            #if self.checkbox_normalize.isChecked():
            #    countRates/=len(pixList)
            countRateHist, bin_edges = np.histogram(c, bins=50, range=(0,2500))
            return countRateHist, bin_edges
        return []
        
    def addData(self, imageList):
        #countRate = np.sum(np.asarray(image)[self.pixelList[:,1],self.pixelList[:,0]])
        #self.countTimestream = np.append(self.countTimestream,countRate)
        self.plotData()
    
    def draw(self):
        self.canvas.draw()
        self.canvas.flush_events()
    
    def setCheckboxText(self, currentPix=False):
        pixList = self.pixelList
        checkbox = self.checkbox_plotPix
        label="Pixels: "
        if currentPix:
            pixList = np.asarray([[p[0],p[1]] for p in self.parent.selectedPixels])
            checkbox=self.checkbox_plotCurrentPix
            label="Current Pixels: "
        
        label = label+('{}, '*len(pixList)).format(*pixList)[:-2]
        checkbox.setText(label)
    
    def create_main_frame(self):
        """
        Makes GUI elements on the window
        
        
        """
        self.main_frame = QWidget()
        
        # Figure
        self.dpi = 100
        self.fig = Figure((9.0, 5.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Count Rate [#/s]')
        self.ax.set_ylabel('#')
        self.line, = self.ax.plot([],[],'g-')
        self.line2, = self.ax.plot([],[],'c-')
        
        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        
        # Controls
        self.checkbox_plotPix = QCheckBox()
        self.checkbox_plotPix.setMaximumWidth(700)
        self.checkbox_plotPix.setMinimumWidth(100)
        self.setCheckboxText()
        self.checkbox_plotPix.setChecked(True)
        self.checkbox_plotPix.setStyleSheet('color: green')
        self.checkbox_plotPix.stateChanged.connect(lambda x: self.plotData())
        
        
        self.checkbox_plotCurrentPix = QCheckBox()
        self.checkbox_plotCurrentPix.setMaximumWidth(700)
        self.checkbox_plotCurrentPix.setMinimumWidth(100)
        self.setCheckboxText(currentPix=True)
        self.checkbox_plotCurrentPix.setChecked(False)
        self.checkbox_plotCurrentPix.setStyleSheet('color: cyan')
        self.checkbox_plotCurrentPix.stateChanged.connect(lambda x: self.plotData())
        
        
        self.checkbox_normalize = QCheckBox('Normalize by number of pixels')
        self.checkbox_normalize.setChecked(False)
        self.checkbox_normalize.stateChanged.connect(lambda x: self.plotData())
        
        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        vbox_plot.addWidget(self.mpl_toolbar)
        
        vbox_plot.addWidget(self.checkbox_plotPix)
        vbox_plot.addWidget(self.checkbox_plotCurrentPix)
        vbox_plot.addWidget(self.checkbox_normalize)
        
        self.main_frame.setLayout(vbox_plot)
        self.setCentralWidget(self.main_frame)
    
    def closeEvent(self, event):
        self.closeWindow.emit()
        event.accept()
