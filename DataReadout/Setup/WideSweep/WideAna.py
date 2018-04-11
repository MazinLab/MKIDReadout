"""
A replacement for WideAna.pro for those who prefer python to idl.

Usage: python WideAna.py filename feedline#

Read the input file.

Interactively add and remove resonance locations.

Create file that plot dB vs frequency, in panels that are 0.15 GHz
wide if it does not exist. This file is the base input file name with
-good.ps appended.

Create file of resonance positions.
This file is the base input file name with -good.txt appended.
This file contains one line for each resonance, with the columns:
  -- id number (sequential from 0)
  -- index of the wavelength in the input file
  -- frequency in GHz

If this file already exists when the program starts, it is moved to
the same file name with at date and time string appended, and a new
copy is made with all found resonances.  The file is updated each time
a line is added or subtracted.


The view window is controlled by:
- self.segment:  value from the segment; slider , varies from 0 to 1000
- self.zoomFactor

- from this calculate fMiddle, the frequency at the middle of the plot,
  ranges from self.wsf.x[0] to self.wsf[-1], xMin and xMax

Matt Strader
Chris Stoughton
Rupert Dodkins
"""
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtGui
import matplotlib.pyplot as plt
import numpy as np
import sys, os, time, shutil
from multiprocessing import Process
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
from functools import partial
from scipy import signal
from scipy.signal import filter_design as fd
from scipy.interpolate import UnivariateSpline
#import Peaks as Peaks
from WideSweepFile import WideSweepFile

class WideAna(QMainWindow):
    def __init__(self, parent=None,plotFunc=None,title='',separateProcess=False, image=None,showMe=True, initialFile=None, flNum=None):

        self.coll_thresh=1.E-4     #100 kHz
        
        self.parent = parent
        if self.parent == None:
            self.app = QApplication([])
        super(WideAna,self).__init__(parent)
        self.fitParams = { "filter":{"order":4,"rs":40,"wn":0.1},
                           'spline':{"splineS":1,"splineK":3}}
        self.initialFile = initialFile
        self.baseFile = ('.').join(initialFile.split('.')[:-1])
        self.goodFile = self.baseFile+"-freqs-good.txt"
        self.allFile = self.baseFile+"-freqs-all.txt"
        if os.path.exists(self.goodFile):
            self.goodFile = self.goodFile+time.strftime("-%Y-%m-%d-%H-%M-%S")
            #shutil.copy(self.goodFile,self.goodFile+time.strftime("-%Y-%m-%d-%H-%M-%S"))
        if os.path.exists(self.allFile):
            self.allFile = self.allFile+time.strftime("-%Y-%m-%d-%H-%M-%S")
        self.pdfFile = self.baseFile+"-good.pdf"
        self.fitLineEdits = {}
        self.flNum = flNum
        self.fitLabels = {}
        self.splineS = 1
        self.splineK = 3
        self.setWindowTitle(title)
        self.plotFunc = plotFunc
        self.create_main_frame(title)
        if plotFunc != None:
            plotFunc(fig=self.fig,axes=self.axes)
        if showMe == True:
            self.show()

        self.load_file(initialFile)
        # make the PDF file

        if not os.path.exists(self.pdfFile):
            print "Create overview PDF file:",self.pdfFile
            self.wsf.createPdf(self.pdfFile)
        else:
            print "Overview PDF file already on disk:",self.pdfFile
        # plot the first segment
        self.deltaXDisplay = 0.100 # display width in GHz
        self.zoomFactor = 2.0
        self.segment = 0
        self.calcXminXmax()
        self.plotSegment()
        self.ylim = self.axes.get_ylim()
        print "Ready to add and delete peaks."

    def draw(self):
        self.fig.canvas.draw()

    def on_key_press(self,event):
        #print "WideAna.on_key_press:  event.key=",event.key
        if event.key == "right" or event.key == 'r':
            self.segmentIncrement(None,0.1)
            return
        elif event.key == "left" or event.key == 'l':
            self.segmentDecrement(None,0.1)
            return
        elif event.key == "up" or event.key == '+':
            self.zoom(1.25)
            return
        elif event.key == "down" or event.key == '-':
            self.zoom(0.8)
            return
        self.on_key_or_button(event, event.key)

    def on_button_press(self,event):
        self.on_key_or_button(event,event.button)


    def on_key_or_button(self, event, pressed):
        xdata = getattr(event, 'xdata', None)
        if xdata is not None:
            ind = np.argmin(np.abs(self.wsf.x-xdata))
            xFound = self.wsf.x[ind]
            indPk = np.searchsorted(self.wsf.pk, ind)
            xPkFound0 = self.wsf.x[self.wsf.pk[indPk-1]]
            xPkFound1 = self.wsf.x[self.wsf.pk[indPk]]
            if abs(xPkFound0-xdata) < abs(xPkFound1-xdata):
                bestIndex = indPk-1
            else:
                bestIndex = indPk
            bestWsfIndex = self.wsf.pk[bestIndex]
            bestX = self.wsf.x[bestWsfIndex]
            if pressed == "d":
                #self.goodPeakMask[bestWsfIndex-7:bestWsfIndex+7] = False # larger area of indicies to pinpoint false resonator location
                #self.badPeakMask[bestWsfIndex-7:bestWsfIndex+7] = False # larger area of indicies to pinpoint false resonator location
                self.goodPeakMask[np.where(np.logical_and(self.wsf.x>(xFound-self.coll_thresh), self.wsf.x<(xFound+self.coll_thresh)))]=False
                self.badPeakMask[np.where(np.logical_and(self.wsf.x>(xFound-self.coll_thresh), self.wsf.x<(xFound+self.coll_thresh)))]=False
                self.setCountLabel()
                self.replot()
                self.writeToGoodFile()
                self.writeToAllFile()

            if pressed == "a":
                if not self.goodPeakMask[bestWsfIndex]:
                    self.goodPeakMask[bestWsfIndex] = True
                    self.badPeakMask[bestWsfIndex] = False
                    self.setCountLabel()
                    self.replot()
                    self.writeToGoodFile()
                    self.writeToAllFile()
           
            if pressed == "s":
                if not self.badPeakMask[bestWsfIndex]:
                    self.badPeakMask[bestWsfIndex] = True
                    self.goodPeakMask[bestWsfIndex] = False
                    self.setCountLabel()
                    self.replot()
                    self.writeToGoodFile()
                    self.writeToAllFile()

    def replot(self):
            xlim = self.axes.set_xlim()
            self.ylim = self.axes.get_ylim()
            self.xMin=xlim[0]
            self.xMax=xlim[1]
            self.plotSegment()
            

    def create_main_frame(self,title):
        self.main_frame = QWidget()

      # Create the mpl Figure and FigCanvas objects. 
        self.dpi = 100
        self.fig = Figure((7, 5), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

        self.axes = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect('key_press_event',self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event',self.on_button_press)
        # Create the segment slider
        self.segmentSlider = QtGui.QSlider(Qt.Horizontal, self)
        self.segmentSlider.setToolTip("Slide to change segment")
        self.segmentMax = 100000.0
        self.segmentSlider.setRange(0,int(self.segmentMax))
        self.segmentSlider.setFocusPolicy(Qt.NoFocus)
        self.segmentSlider.setGeometry(30,40,100,30)
        self.segmentSlider.valueChanged[int].connect(self.changeSegmentValue)

        # Create the left and right buttons
        segmentDecrement = QtGui.QPushButton('<',self)
        segmentDecrement.setToolTip("Back to previous segment")
        segmentDecrement.clicked[bool].connect(self.segmentDecrement)
        segmentIncrement = QtGui.QPushButton('>',self)
        segmentIncrement.setToolTip("Forward to next segment")
        segmentIncrement.clicked[bool].connect(self.segmentIncrement)

        # create display mode button
        self.yDisplay = QtGui.QPushButton("raw")
        self.yDisplay.setToolTip("Toggle y axis: raw data or difference=raw-baseline") 
        self.yDisplay.setCheckable(True)
        self.yDisplay.clicked[bool].connect(self.yDisplayClicked)

        # create information boxes
        self.instructionsLabel = QtGui.QLabel()
        self.instructionsLabel.setText("ADD peak:  a;  REMOVE peak:  d  ZOOM:  +/- SCAN l/r ")
        self.countLabel = QtGui.QLabel()
        self.countLabel.setText("count label")

        self.inputLabel = QtGui.QLabel()
        self.inputLabel.setText("Input File:%s"%self.initialFile)

        self.goodLabel = QtGui.QLabel()
        self.goodLabel.setText("Good File:%s"%self.goodFile)

        # create controls for baseline fitting
        #self.baseline = QtGui.QPushButton("filter")
        #self.baseline.setCheckable(True)
        #self.baseline.clicked[bool].connect(self.baselineClicked)


        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        # Do the layout

        # segment box
        segmentBox = QHBoxLayout()
        segmentBox.addWidget(segmentDecrement)
        segmentBox.addWidget(self.segmentSlider)
        segmentBox.addWidget(segmentIncrement)
        segmentBox.addWidget(self.yDisplay)

        # baseline box
        #self.baselineBox = QHBoxLayout()
        #self.updateBaselineBox()

        # info box
        self.infoBox = QVBoxLayout()
        self.infoBox.addWidget(self.inputLabel)
        self.infoBox.addWidget(self.goodLabel)
        self.infoBox.addWidget(self.countLabel)
        self.infoBox.addWidget(self.instructionsLabel)
        # entire box
        vbox = QVBoxLayout()
        vbox.addLayout(self.infoBox)
        vbox.addLayout(segmentBox)
        #vbox.addLayout(self.baselineBox)
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)        

    def updateBaselineBox(self):
        for i in range(self.baselineBox.count()):
            item = self.baselineBox.itemAt(i)
            self.baselineBox.removeItem(item)
        mode = str(self.baseline.text())
        self.baseline.setFixedSize(70,40)
        self.baselineBox.addWidget(self.baseline)
        keys = self.fitParams[mode]

    def load_file(self, fileName):
        self.wsf = WideSweepFile(fileName)
        #self.wsf.fitSpline(splineS=1.0, splineK=1)
        self.wsf.fitFilter(wn=0.01)
        self.wsf.findPeaks(m=2)
        self.goodPeakMask = np.zeros(len(self.wsf.x),dtype=np.bool)
        self.badPeakMask = np.zeros(len(self.wsf.x),dtype=np.bool)
        self.collMask = np.zeros(len(self.wsf.x),dtype=np.bool)
        if os.path.isfile(self.baseFile+"-ml.txt"):             # update: use machine learning peak loacations if they've been made
            print 'loading peak location predictions from', self.baseFile+"-ml.txt"
            peaks = np.loadtxt(self.baseFile+"-ml.txt")
            peaks = map(int,peaks)
        else:
            peaks = self.wsf.peaks
        



        #remove collisions
        freqsAtPeaks = self.wsf.x[peaks]
        args = np.argsort(freqsAtPeaks)
        peaks=peaks[args]
        freqsAtPeaks=freqsAtPeaks[args]
        diffs = freqsAtPeaks[1:] - freqsAtPeaks[:-1]
        diffs = np.append(diffs, [peaks[-1]])
        peaks=np.delete(peaks, np.where(diffs<2.E-9))    #remove duplicate peaks
        freqsAtPeaks=np.delete(freqsAtPeaks, np.where(diffs<2.E-9))    #remove duplicate peaks
        
        diffs = freqsAtPeaks[1:] - freqsAtPeaks[:-1]
        diffs = np.append(diffs, [peaks[-1]])
        colls = peaks[np.where(diffs<=self.coll_thresh)]

        '''
        #coll_thresh = self.wsf.x[0]
        dist = abs(np.roll(peaks, -1) - peaks)
        #colls = np.delete(peaks,np.where(dist>=9))

        
        colls=[]
        diff, coll_thresh = 0, 0
        while diff <= 5e-4:
            diff = self.wsf.x[coll_thresh]-self.wsf.x[0]
            coll_thresh += 1
        
        print coll_thresh
        for i in range(len(peaks)):
            if dist[i]<coll_thresh:
                if self.wsf.mag[peaks[i+1]] - self.wsf.mag[peaks[i]] > 1.5:
                    colls.append(peaks[i+1])
                    #print 'for ', self.wsf.x[peaks[i]], 'chosing the one before'
                else:
                    colls.append(peaks[i])
                    #print 'for ', self.wsf.x[peaks[i]], 'chosing this one'
        '''

        #print colls
        if colls != []:
            #colls=np.array(map(int,colls))
            self.collMask[colls] = True # update: so unidentified peaks can be identified as unusable
            
        peaks = np.delete(peaks,colls) #remove collisions (peaks < 0.5MHz apart = < 9 steps apart)
        #peaks = np.delete(peaks,np.where(dist<9)) #remove collisions (peaks < 0.5MHz apart = < 9 steps apart)
        #self.goodPeakMask[peaks] = True
        #self.badPeakMask[colls] = True
        #self.goodPeakMask[colls] = False

        self.setCountLabel()
        self.writeToGoodFile()
        self.writeToAllFile()

    def setCountLabel(self):
        self.countLabel.setText("Good peaks = %d, All peaks = %d" % (self.goodPeakMask.sum(), self.goodPeakMask.sum() + self.badPeakMask.sum()))

    def writeToGoodFile(self):
        #gf = open(self.goodFile,'wb')
        ##id = (self.flNum-1)*2000
        #resId = self.flNum*10000
        #for index in range(len(self.goodPeakMask)):
        #    if self.goodPeakMask[index]:
        #        line = "%8d %12d %16.7f\n"%(resId,index,self.wsf.x[index])
        #        gf.write(line)
        #        resId += 1
        #    elif self.badPeakMask[index]:
        #        resId += 1
        #gf.close()

        ws_good_inds = np.where(self.goodPeakMask>0)
        ws_bad_inds = np.where(self.badPeakMask>0)
        freqs = np.append(self.wsf.x[ws_good_inds], self.wsf.x[ws_bad_inds])
        sort_inds = np.argsort(freqs)
        resIds = np.asarray(range(len(freqs)))+self.flNum*10000

        d=np.diff(self.wsf.x[ws_good_inds])
        if np.any(d<0):
            print "Warning, Some freqs were out of order, check the output file to make sure they were handled correctly"
        

        data = np.asarray([resIds[np.where(sort_inds<len(ws_good_inds[0]))], \
                ws_good_inds[0][sort_inds[np.where(sort_inds<len(ws_good_inds[0]))]], \
                freqs[sort_inds][np.where(sort_inds<len(ws_good_inds[0]))]]).T
        gf=file(self.goodFile,'wb') #Can open file in append mode this way. But doesn't matter here...
        np.savetxt(gf, data,fmt="%8d %12d %16.7f")
        gf.close()

    def writeToAllFile(self):
        #af = open(self.allFile,'wb')
        ##id = (self.flNum-1)*2000
        #resId = self.flNum*10000
        #print len(np.where(self.goodPeakMask==1)[0]), len(np.where(self.badPeakMask==1)[0])
        #for index in range(len(self.goodPeakMask)):
        #    if self.goodPeakMask[index] or self.badPeakMask[index]:
        #        line = "%8d %12d %16.7f\n"%(resId,index,self.wsf.x[index])
        #        af.write(line)
        #        resId += 1
        #af.close()
        
        ws_good_inds = np.where(self.goodPeakMask>0)
        ws_bad_inds = np.where(self.badPeakMask>0)
        freqs = np.append(self.wsf.x[ws_good_inds], self.wsf.x[ws_bad_inds])
        sort_inds = np.argsort(freqs)
        resIds = np.asarray(range(len(freqs)))+self.flNum*10000
        data = np.asarray([resIds, np.append(ws_good_inds[0], ws_bad_inds[0])[sort_inds], freqs[sort_inds]])
        np.savetxt(self.allFile, data.T,fmt='%8d %12d %16.7f')

    # deal with zooming and plotting one segment
    def zoom(self,zoom):
        self.zoomFactor *= zoom
        self.calcXminXmax()
        self.plotSegment()

    def changeSegmentValue(self,value):
        self.segment = value
        self.calcXminXmax()
        self.ylim = self.axes.get_ylim()
        self.plotSegment()

    def segmentDecrement(self, value, amount=0.9):
        wsfDx = self.wsf.x[-1]-self.wsf.x[0]
        plotDx = self.xMax-self.xMin
        dsdx = self.segmentMax / wsfDx
        ds = amount * dsdx * plotDx
        self.segment = max(0,self.segment-ds)
        self.segmentSlider.setSliderPosition(self.segment)

    def segmentIncrement(self, value, amount=0.9):
        wsfDx = self.wsf.x[-1]-self.wsf.x[0]
        plotDx = self.xMax-self.xMin
        dsdx = self.segmentMax / wsfDx
        ds = amount * dsdx * plotDx
        self.segment = min(self.segmentMax,self.segment+ds)
        self.segmentSlider.setSliderPosition(self.segment)

    def calcXminXmax(self):
        xMiddle = self.wsf.x[0] + \
            (self.segment/self.segmentMax)*(self.wsf.x[-1]-self.wsf.x[0])
        dx = self.deltaXDisplay/self.zoomFactor
        self.xMin = xMiddle-dx/2.0
        self.xMax = xMiddle+dx/2.0

    def plotSegment(self):
        ydText = self.yDisplay.text()
        if self.wsf != None:
            if ydText == "raw":
                yPlot = 20*np.log10(self.wsf.mag)
                yName = "log-magnitude"
            else:
                yPlot = self.wsf.mag-self.wsf.baseline
                yName = "log(mag-baseline)"
            stride = self.wsf.data1.shape[0]/self.segmentMax
            # plot all values and then set xmin and xmax to show this segment
            self.axes.clear()
            self.axes.plot(self.wsf.x, yPlot, label=yName)

            for x in self.wsf.x[self.goodPeakMask]:
                if x > self.xMin and x < self.xMax:
                    self.axes.axvline(x=x,color='g')
                    self.axes.axvline(x=x+self.coll_thresh,color='g',linestyle='-.',linewidth=0.5)
                    self.axes.axvline(x=x-self.coll_thresh,color='g',linestyle='-.',linewidth=0.5)
            #for c in self.wsf.x[self.collMask]:
            #    if c > self.xMin and c < self.xMax:
            #        self.axes.axvline(x=c,color='g')

            for x in self.wsf.x[self.badPeakMask]:
                if x > self.xMin and x < self.xMax:
                    self.axes.axvline(x=x,color='r')
                    self.axes.axvline(x=x+self.coll_thresh,color='r',linestyle='-.',linewidth=0.5)
                    self.axes.axvline(x=x-self.coll_thresh,color='r',linestyle='-.',linewidth=0.5)
            
            self.axes.set_xlim((self.xMin,self.xMax))
            try: self.axes.set_ylim(self.ylim)
            except: pass
            self.axes.set_title("segment=%.1f/%.1f"%(self.segment,self.segmentMax))
            #self.axes.legend().get_frame().set_alpha(0.5)
            self.draw()
        self.canvas.setFocus()

    def yDisplayClicked(self, value):
        if value:
            self.yDisplay.setText("diff")
        else:
            self.yDisplay.setText("raw")
        self.replot()

    def baselineClicked(self,value):
        if value:
            self.baseline.setText("spline")
        else:
            self.baseline.setText("filter")
        self.updateBaselineBox()

    def show(self):
        super(WideAna,self).show()
        if self.parent == None:
            self.app.exec_()

def main(initialFile=None, flNum=None):
    form = WideAna(showMe=False,title='WideSweep',initialFile=initialFile,flNum=flNum)
    form.show()

if __name__ == "__main__":
    initialFile = None
    try: 
        initialFileName = sys.argv[1]
        initialFile = initialFileName
        if not os.path.isfile(initialFile):
            mdd = os.environ['MKID_DATA_DIR']
            initialFile = os.path.join(mdd,initialFileName)
    except:
        print "Can not find",initialFile
        print "Usage: python WideAna.py filename feedline#"
        exit()
    try: flNum = int(sys.argv[2])
    except:
        print "Usage: python WideAna.py filename feedline#"
        exit()
    main(initialFile=initialFile, flNum=flNum)

    
    
    
