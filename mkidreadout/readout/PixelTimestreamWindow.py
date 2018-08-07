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

class PixelTimestreamWindow(QMainWindow):
    """
    We use this for realtime plotting with MkidDashboard
    """
    closeWindow = QtCore.pyqtSignal()
    
    def __init__(self, pixelList, parent):
        super(QMainWindow, self).__init__(parent)
        self.setWindowTitle("Timestream")
        self.parent=parent
        self.pixelList=np.asarray(pixelList,dtype=np.int)
        self.create_main_frame()
        self.plotData()
        parent.newImageProcessed.connect(self.plotData)
        
    def plotData(self,**kwargs):
        self.setCheckboxText(currentPix=True)
        countRate=[]
        countRate_cur=[]
        if self.checkbox_plotPix.isChecked():
            countRate = self.getCountRate()
        if self.checkbox_plotCurrentPix.isChecked():
            countRate_cur = self.getCountRate(True)
        oldx_lim = self.ax.get_xlim()
        oldy_lim = self.ax.get_ylim()
        self.ax.cla()
        self.ax.plot(countRate,'g-')
        self.ax.plot(countRate_cur,'c-')
        if self.mpl_toolbar._active is None:
            self.ax.relim()
            self.ax.autoscale_view(True,True,True)
        else:
            self.ax.set_xlim(oldx_lim)
            self.ax.set_ylim(oldy_lim)
        self.draw()

    def getCountRate(self, forCurrentPix=False):
        imageList=self.parent.imageList
        pixList = self.pixelList
        if forCurrentPix: pixList = np.asarray([[p[0],p[1]] for p in self.parent.selectedPixels])
        if len(imageList)>0 and len(pixList)>0:
            x = pixList[:,0]
            y = pixList[:,1]
            c = np.asarray(imageList)[:,y,x]
            countRate = np.sum(c,axis=1)
            if self.checkbox_normalize.isChecked():
                numZeroPix = len(np.where(np.sum(c,axis=0)==0)[0])
                if len(pixList)>numZeroPix:
                    countRate/=(len(pixList)-numZeroPix)
            return countRate
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
        self.ax.set_xlabel('Time [s]')
        self.ax.set_ylabel('Count Rate [#/s]')
        
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
        self.parent.newImageProcessed.disconnect(self.plotData)
        self.closeWindow.emit()
        event.accept()

