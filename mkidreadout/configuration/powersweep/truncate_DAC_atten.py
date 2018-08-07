'''
author: Clint Bockstiegel
May 2017

Instructions:
1.) Place the DAC atten files into their own directory, with no other files. There should be
    one file/roach. If there are files ending with "_trun.txt" they will be ignored.

2.) Run this python script with the command:
    python truncate_DAC_atten.py

3.) Click Open. This will display an "Open file" dialog box. Select one of the DAC 
    atten files that you want to truncate. Click OK.
    This code will find all the other .txt files in the same directory and load them. 
    (Except ones that end with "_trunc.txt". )

4.) Set the cutoff attenuations for each roach as desired.

5.) Save individual roaches, or click Save All at the end when you're 
    happy with all the truncation values. Both Save and Save All will overwrite
    existing _trunc files.
'''
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
import os, sys
import shutil



class roach:
#roach class that will contain all the relevant information for each roach:
    def __init__(self, filename):
        self.filename = filename
        self.data = np.loadtxt(self.filename)
        self.resID = self.data[:,0]
        self.freq = self.data[:,1]
        self.attens = self.data[:,2]
        self.truncatedAttens = np.copy(self.attens)
        self.cutoff = np.amin(self.attens)
        self.dataTruncated = np.copy(self.data)


class testWindow(QMainWindow):

    def __init__(self,parent=None):
        QMainWindow.__init__(self,parent=parent)
        self.setWindowTitle('Truncate DAC Attenuation')
        self.create_main_frame()

        self.powerSweepDataPath = '/'



    def plotTruncatedHistogram(self,*args):
        #clear the axes
        self.ax1.clear()    

        #get the roach number
        roachNum = self.spinbox_roachNumber.value()

        #set the new range for the cutoff atten spinbox, custom for each roach
        self.spinbox_cutoffAtten.setRange(np.amin(self.roachList[roachNum].attens),100)
        self.cutoff = self.spinbox_cutoffAtten.value()
        #print a warning if the cutoff value is out of range of DAC attens in file
        if np.logical_or(self.cutoff<np.amin(self.roachList[roachNum].attens),self.cutoff>np.amax(self.roachList[roachNum].attens)):
            print 'cutoff attenuation is out of range of measured values'

        #find where the DAC atten array has values < cutoff value, update the truncatedAttens array
        #modify the roach object
        self.violatingIndices = np.where(self.roachList[roachNum].attens < self.cutoff)[0]       
        self.roachList[roachNum].truncatedAttens = np.copy(self.roachList[roachNum].attens)
        self.roachList[roachNum].truncatedAttens[self.violatingIndices] = self.cutoff

        #set up the bins for the histogram plot
        self.bins = range(int(np.amin(self.roachList[roachNum].attens)),int(np.amax(self.roachList[roachNum].attens)))
        #plot the truncatedAttens histogram
        self.ax1.hist(self.roachList[roachNum].truncatedAttens,self.bins)
        self.draw()
        #plot the pre-truncated attens histogram for comparison
        self.ax1.hist(self.roachList[roachNum].attens,self.bins,alpha = 0.7)  #alpha sets the transparency
        self.ax1.set_ylabel('number of resonators')
        self.ax1.set_xlabel('DAC atten [dB]')
        #set plot label as filename
        self.ax1.set_title(self.roachList[roachNum].filename[len(self.powerSweepDataPath)+1:]) 
        self.draw()
        
        #update the roach object with new cutoff value
        self.roachList[roachNum].cutoff = self.cutoff
        
        if 0:
            print 'plotting roach: ',self.roachList[roachNum].filename[len(self.powerSweepDataPath):]



    def setCutoffAttenSpinboxValue(self):
        #call this function when changing the roach number spinbox value
        roachNum = self.spinbox_roachNumber.value() #get the roach number
        #catch the situation when the new cutoff value is the same as the previous value
        if self.spinbox_cutoffAtten.value() == self.roachList[roachNum].cutoff: 
            self.plotTruncatedHistogram()
        self.spinbox_cutoffAtten.setRange(0,100)
        self.spinbox_cutoffAtten.setValue(self.roachList[roachNum].cutoff)
    


    def getFilenameFromUser(self):
        #create an open file dialog box to prompt user to select a file to load
        filename = str(QFileDialog.getOpenFileName(self, 'Select One File', '/'))
        print os.path.dirname(filename)
        #set the data path variable
        self.powerSweepDataPath = os.path.dirname(filename)

        #load in the data. Look for all files ending with ".txt" and add them to list of files to load in.
        #ignore files that end with "_trunc.txt", because those are what this program outputs
        fileListRaw = []
        for file in os.listdir(self.powerSweepDataPath):
            if file.endswith(".txt"):
                if file.endswith("_trunc.txt"): 
                    print 'skipping loading of ' + str(file)
                else:
                    fileListRaw = fileListRaw + [os.path.join(self.powerSweepDataPath, file)]
        self.fileListRaw = fileListRaw
        print '\nloading data\n'

        #create list of roach objects. One roach per file loaded. 
        self.roachList = []
        for ii in range(len(self.fileListRaw)):
            self.roachList = self.roachList + [roach(self.fileListRaw[ii])]

        #plot the first roach by setting the cutoff atten spinbox value
        self.setCutoffAttenSpinboxValue()


    def saveOneRoach(self):
        #save data of currently displayed histogram in a new file with "_trunc.txt" at end.
        roachNum = self.spinbox_roachNumber.value()
        self.roachList[roachNum].dataTruncated[:,2] = self.roachList[roachNum].truncatedAttens
        print '\nWriting ',self.fileListRaw[roachNum][:-4]+'_trunc.txt'
        np.savetxt(self.fileListRaw[roachNum][:-4]+'_trunc.txt', self.roachList[roachNum].dataTruncated,fmt='%6i %10.9e %4i')


    def saveAllRoach(self):
        #save all roach truncated histograms.
        for ii in range(len(self.fileListRaw)):
            self.roachList[ii].dataTruncated[:,2] = self.roachList[ii].truncatedAttens
            print '\nWriting ',self.fileListRaw[ii][:-4]+'_trunc.txt'
            np.savetxt(self.fileListRaw[ii][:-4]+'_trunc.txt', self.roachList[ii].dataTruncated,fmt='%6i %10.9e %4i')
        print '\nSAVE ALL OPERATION COMPLETE.'


        
            
        


    def create_main_frame(self):
        """
        Makes GUI elements on the window
        """

        #Define the plot window. 
        self.main_frame = QWidget()
        self.dpi = 100
        self.fig = Figure((9.0, 5.0), dpi=self.dpi) #define the figure, set the size and resolution
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.ax1 = self.fig.add_subplot(111)
#        self.ax1.set_ylabel('y axis label')
#        self.ax1.set_xlabel('resonator attenuation [dB]')

        #Define the button that will save one roach.
        button_save = QPushButton("Save")
        button_save.setEnabled(True)
        button_save.setToolTip('Saves current roach.')
        button_save.clicked.connect(self.saveOneRoach)


        #Define the button that will save all roaches.
        button_saveAll = QPushButton("Save All")
        button_saveAll.setEnabled(True)
        button_saveAll.clicked.connect(self.saveAllRoach)


        #Define the button that will load data 
        button_loadData = QPushButton("Open")
        button_loadData.setEnabled(True)
        button_loadData.setToolTip('Select one of the power sweep files you want to look at.\n All other files in the same directory will be loaded as well.')
        button_loadData.clicked.connect(self.getFilenameFromUser)


        #make a label for the roach number spinbox
        label_roachNumber = QLabel('Roach Number')


        #make a label for the roach cutoff attenuation
        label_cutoff = QLabel('Cutoff Attenuation')


        #Let's draw a spin box for the roach number.
        roachNumber = 0
        self.spinbox_roachNumber = QSpinBox()
        self.spinbox_roachNumber.setValue(roachNumber)
        self.spinbox_roachNumber.setRange(0,9)
        self.spinbox_roachNumber.setWrapping(False)
        self.spinbox_roachNumber.setToolTip('Use up/down arrows on keyboard!.')
        self.spinbox_roachNumber.valueChanged.connect(self.setCutoffAttenSpinboxValue)  
        # Using .connect will send self.makePlot an argument which is equal to the new value. 


        #Let's draw spin box for the cutoff attenuation.
        initCutoff = 0
        self.spinbox_cutoffAtten = QSpinBox()
        self.spinbox_cutoffAtten.setValue(initCutoff)
        self.spinbox_cutoffAtten.setRange(0,100)
        self.spinbox_cutoffAtten.setWrapping(False)
        self.spinbox_cutoffAtten.setToolTip('Use up/down arrows on keyboard!.')
        self.spinbox_cutoffAtten.valueChanged.connect(self.plotTruncatedHistogram)  


        #create a vertical box for the plot to go in.
        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)

        #create a vbox for the roach label and spinbox to go in
        vbox_roachNumber = QVBoxLayout()
        vbox_roachNumber.addWidget(label_roachNumber)
        vbox_roachNumber.addWidget(self.spinbox_roachNumber)


        #create a vbox for the atten cutoff label and atten cutoff spinbox to go in
        vbox_cutoffAtten = QVBoxLayout()
        vbox_cutoffAtten.addWidget(label_cutoff)
        vbox_cutoffAtten.addWidget(self.spinbox_cutoffAtten)

        #Create an h box for the buttons and spinboxes to go in.
        hbox_ch = QHBoxLayout()
        hbox_ch.addWidget(button_loadData)
        hbox_ch.addLayout(vbox_roachNumber)
        hbox_ch.addLayout(vbox_cutoffAtten)
        hbox_ch.addWidget(button_save)
        hbox_ch.addWidget(button_saveAll)

        #Now create another vbox, and add the plot vbox and the button's hbox to the new vbox.
        vbox_combined = QVBoxLayout()
        vbox_combined.addLayout(vbox_plot)
        vbox_combined.addLayout(hbox_ch)

        #Set the main_frame's layout to be vbox_combined
        self.main_frame.setLayout(vbox_combined)

        #Set the overall QWidget to have the layout of the main_frame.
        self.setCentralWidget(self.main_frame)


    def draw(self):
        #The plot window calls this function
        self.canvas.draw()
        self.canvas.flush_events()


if __name__ == "__main__":
    a = QApplication(sys.argv)
    foo = testWindow()
    foo.show()
    sys.exit(a.exec_())



