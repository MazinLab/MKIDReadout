import numpy as np
from tables import *
import os
import time
import matplotlib.pyplot as plt
import sys
from matplotlib.backends.backend_pdf import PdfPages
import scipy.signal as signal
from scipy import optimize
import scipy.stats as stats

from PyQt4.QtGui import *
from PyQt4.QtGui import *
from beammap_gui import Ui_beammap_gui

# Define the various classes and functions needed for the beam mapping
# Define a standard Gaussian distribution function
def gaussian(pars, x):
    center, width, height, back = pars
    width = float(width)
    return back + height*np.exp(-(((center-x)/width)**2)/2)

# Define an error function between data and a Gaussian
def errorfunction(params, data, x):
    errorfunction = data - gaussian(params,x)
    return errorfunction

# Find an optimal Guassian fit for the data, return parameters of that Gaussian
def fitgaussian(data,x):
    params=(x.mean(),2.*(x[1]-x[0]),data.max(), 0.)
    p, success = optimize.leastsq(errorfunction, params, args=(data, x))
    return p

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Input Parameter Gui
class Sweep_Number(QWidget):
    
    def __init__(self):
        super(Sweep_Number, self).__init__()
        
        self.initUI()
        
    def initUI(self):

        # Default values
        self.sweep_count = [1,1]
        self.freqpath = os.getcwd()
        self.savepath = os.getcwd()
        self.roachnumber = 10
        self.maxpix = 10000

        self.btnx = QPushButton('X Sweeps', self)
        self.btnx.move(20, 20)
        self.btnx.clicked.connect(self.showDialogX)

        self.btny = QPushButton('Y Sweeps', self)
        self.btny.move(20, 50)
        self.btny.clicked.connect(self.showDialogY)

        self.btn1 = QPushButton('Frequency Path', self)
        self.btn1.move(20, 140)
        self.btn1.clicked.connect(self.freq_dialog)

        self.btn2 = QPushButton('Save Path', self)
        self.btn2.move(20, 170)
        self.btn2.clicked.connect(self.save_dialog)

        self.btn3 = QPushButton('Number of Roaches', self)
        self.btn3.move(20, 80)
        self.btn3.clicked.connect(self.roach_dialog)

        self.btn4 = QPushButton('Max Pixel Count', self)
        self.btn4.move(20, 110)
        self.btn4.clicked.connect(self.pix_dialog)
        
        self.lex = QLabel(self)
        self.lex.setGeometry(150, 20, 40 , 20)
        self.lex.setFrameStyle(QFrame.Panel)
        self.lex.setText(str(self.sweep_count[0]))

        self.ley = QLabel(self)
        self.ley.setGeometry(150, 50 , 40 , 20)
        self.ley.setFrameStyle(QFrame.Panel)
        self.ley.setText(str(self.sweep_count[1]))

        self.le1 = QLabel(self)
        self.le1.setGeometry(150, 140, 400, 20)
        self.le1.setFrameStyle(QFrame.Panel)
        self.le1.setText(str(self.freqpath))

        self.le2 = QLabel(self)
        self.le2.setGeometry(150, 170, 400, 20)
        self.le2.setFrameStyle(QFrame.Panel)
        self.le2.setText(str(self.savepath))

        self.le3 = QLabel(self)
        self.le3.setGeometry(150, 80, 40 , 20)
        self.le3.setFrameStyle(QFrame.Panel)
        self.le3.setText(str(self.roachnumber))

        self.le4 = QLabel(self)
        self.le4.setGeometry(150, 110, 40 , 20)
        self.le4.setFrameStyle(QFrame.Panel)
        self.le4.setText(str(self.maxpix))

        self.okbtn = QPushButton('OK', self)
        self.okbtn.move(20, 200)
        self.okbtn.clicked.connect(self.close)
        
        self.setGeometry(300, 300, 600, 250)
        self.setWindowTitle('Inputs')
        self.show()

    def freq_dialog(self):
        text =QFileDialog.getExistingDirectory()
        self.freqpath=str(text)
        self.le1.setText(str(text))
        
    def save_dialog(self):
        text =QFileDialog.getExistingDirectory()
        self.savepath=str(text)
        self.le2.setText(str(text))

    def roach_dialog(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Number of ROACH Boards:')
        
        if ok:
            self.le3.setText(str(text))
            self.roachnumber=str(text)


    def pix_dialog(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Number of total pixels:')
        
        if ok:
            self.le4.setText(str(text))
            self.maxpix=str(text)
        
        
    def showDialogX(self):
        
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Number of X Sweeps:')
        
        if ok:
            self.lex.setText(str(text))
            self.sweep_count[0] = int(text)

    def showDialogY(self):
        
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Number of Y Sweeps:')
        
        if ok:
            self.ley.setText(str(text))
            self.sweep_count[1] = int(text)
        
def numberrun():
    
    app1 = QApplication(sys.argv)
    ex1 = Sweep_Number()
    app1.exec_()
    return [ex1.sweep_count[0],ex1.sweep_count[1],ex1.freqpath,ex1.savepath, ex1.roachnumber, ex1.maxpix]


# X Sweep Gui
class Sweep_DialogX(QWidget):    
    def __init__(self):
        super(Sweep_DialogX, self).__init__()       
        self.initUI()
        
    def initUI(self):

        self.btn = QPushButton('Add', self)
        self.btn.move(20, 20)
        self.btn.clicked.connect(self.file_dialog)

        self.okbtn = QPushButton('Clear', self)
        self.okbtn.move(20, 50)
        self.okbtn.clicked.connect(self.clear_sweeps)
        
        self.okbtn = QPushButton('OK', self)
        self.okbtn.move(20, 100)
        self.okbtn.clicked.connect(self.close)

        self.le = []
        for i in range(input_params[0]):
            self.le.append(QLineEdit(self))
            self.le[i].setGeometry(130, 22+22*i,400,20)
        
        self.x_array = []
        
        self.setGeometry(300, 300, 600, 150)
        self.setWindowTitle('Choose X sweep files')
        self.show()

    def clear_sweeps(self):
        for i in range(len(self.x_array)):
            self.le[i].setText('')
        self.x_array = []
        
            
    def file_dialog(self):
        text =QFileDialog.getOpenFileName(parent=None, caption=str("Choose Sweep File"), filter=str("H5 (*.h5)")) 
        self.x_array.append(str(text))
        for i in range(len(self.x_array)):
            self.le[i].setText(self.x_array[i])

def xrun():
    
    appx = QApplication(sys.argv)
    ex = Sweep_DialogX()
    appx.exec_()
    return ex.x_array

# Y Sweep Gui
class Sweep_DialogY(QWidget):    
    def __init__(self):
        super(Sweep_DialogY, self).__init__()       
        self.initUI()
        
    def initUI(self):

        self.btn = QPushButton('Add', self)
        self.btn.move(20, 20)
        self.btn.clicked.connect(self.file_dialog)

        self.okbtn = QPushButton('Clear', self)
        self.okbtn.move(20, 50)
        self.okbtn.clicked.connect(self.clear_sweeps)
        
        self.okbtn = QPushButton('OK', self)
        self.okbtn.move(20, 100)
        self.okbtn.clicked.connect(self.close)

        self.le = []
        for i in range(input_params[1]):
            self.le.append(QLineEdit(self))
            self.le[i].setGeometry(130, 22+22*i,400,20)
        
        self.y_array = []
        
        self.setGeometry(300, 300, 600, 150)
        self.setWindowTitle('Choose Y sweep files')
        self.show()

    def clear_sweeps(self):
        for i in range(len(self.y_array)):
            self.le[i].setText('')
        self.y_array = []
        
            
    def file_dialog(self):
        text =QFileDialog.getOpenFileName(parent=None, caption=str("Choose Sweep File"), filter=str("H5 (*.h5)")) 
        self.y_array.append(str(text))
        for i in range(len(self.y_array)):
            self.le[i].setText(self.y_array[i])

def yrun():
    
    appy = QApplication(sys.argv)
    ey = Sweep_DialogY()
    appy.exec_()
    return ey.y_array

class StartQt4(QMainWindow):
    def __init__(self,xtime,ytime,xfilelength,yfilelength):
        QWidget.__init__(self, parent=None)
        self.ui = Ui_beammap_gui()
        self.ui.setupUi(self)

        # Initialize arrays that will contain h5 data
        self.crx_median = np.zeros((maximum_pixels,xtime))
        self.cry_median = np.zeros((maximum_pixels,ytime))
        self.crx = np.zeros(((xfilelength,maximum_pixels,xtime)))
        self.cry = np.zeros(((yfilelength,maximum_pixels,ytime)))
        self.peakpos = np.zeros((2,maximum_pixels))
        self.mypeakpos = np.zeros((2,maximum_pixels))
        self.doublepos = np.zeros((2,maximum_pixels))
        self.holder = [0,1,2,3,4]
        self.flagarray = np.zeros(maximum_pixels)
        self.currentroach = 0
        self.xfit = np.zeros((maximum_pixels,xtime))
        self.yfit = np.zeros((maximum_pixels,ytime))
        
        # Calculate data used for plots
        self.calculate_plot_data()

        # Store median data
        self.xvals=np.arange(len(self.crx_median[0][:]))
        self.yvals=np.arange(len(self.cry_median[0][:]))

        # Calculate fits from h5
        self.perform_fits()
        
        self.make_plots()
        
        self.ui.pp0x.setText(str(self.peakpos[0][0]))
        self.ui.pp0y.setText(str(self.peakpos[1][0]))
        self.ui.pp1x.setText(str(self.peakpos[0][1]))
        self.ui.pp1y.setText(str(self.peakpos[1][1]))
        self.ui.pp2x.setText(str(self.peakpos[0][2]))
        self.ui.pp2y.setText(str(self.peakpos[1][2]))
        self.ui.pp3x.setText(str(self.peakpos[0][3]))
        self.ui.pp3y.setText(str(self.peakpos[1][3]))
        self.ui.pp4x.setText(str(self.peakpos[0][4]))
        self.ui.pp4y.setText(str(self.peakpos[1][4]))

        self.ui.le0x.setText(str(self.mypeakpos[0][0]))
        self.ui.le0y.setText(str(self.mypeakpos[1][0]))
        self.ui.le1x.setText(str(self.mypeakpos[0][1]))
        self.ui.le1y.setText(str(self.mypeakpos[1][1]))
        self.ui.le2x.setText(str(self.mypeakpos[0][2]))
        self.ui.le2y.setText(str(self.mypeakpos[1][2]))
        self.ui.le3x.setText(str(self.mypeakpos[0][3]))
        self.ui.le3y.setText(str(self.mypeakpos[1][3]))
        self.ui.le4x.setText(str(self.mypeakpos[0][4]))
        self.ui.le4y.setText(str(self.mypeakpos[1][4]))

        self.ui.dle0x.setText(str(self.doublepos[0][0]))
        self.ui.dle0y.setText(str(self.doublepos[1][0]))
        self.ui.dle1x.setText(str(self.doublepos[0][1]))
        self.ui.dle1y.setText(str(self.doublepos[1][1]))
        self.ui.dle2x.setText(str(self.doublepos[0][2]))
        self.ui.dle2y.setText(str(self.doublepos[1][2]))
        self.ui.dle3x.setText(str(self.doublepos[0][3]))
        self.ui.dle3y.setText(str(self.doublepos[1][3]))
        self.ui.dle4x.setText(str(self.doublepos[0][4]))
        self.ui.dle4y.setText(str(self.doublepos[1][4]))

        # Start the next, save, or go to process when button clicked
        self.ui.nextbtn.clicked.connect(self.next_process)
        self.ui.savebtn.clicked.connect(self.save_process)
        self.ui.gobtn.clicked.connect(self.go_process)

        # Show enlarged plots when enlarge buttons are clicked
        self.ui.ebtn0x.clicked.connect(self.enlarge0x)
        self.ui.ebtn0y.clicked.connect(self.enlarge0y)
        self.ui.ebtn1x.clicked.connect(self.enlarge1x)
        self.ui.ebtn1y.clicked.connect(self.enlarge1y)
        self.ui.ebtn2x.clicked.connect(self.enlarge2x)
        self.ui.ebtn2y.clicked.connect(self.enlarge2y)
        self.ui.ebtn3x.clicked.connect(self.enlarge3x)
        self.ui.ebtn3y.clicked.connect(self.enlarge3y)
        self.ui.ebtn4x.clicked.connect(self.enlarge4x)
        self.ui.ebtn4y.clicked.connect(self.enlarge4y)

        # Input self peak position when return pressed
        self.ui.le0x.returnPressed.connect(self.le0x_pressed)
        self.ui.le1x.returnPressed.connect(self.le1x_pressed)
        self.ui.le2x.returnPressed.connect(self.le2x_pressed)
        self.ui.le3x.returnPressed.connect(self.le3x_pressed)
        self.ui.le4x.returnPressed.connect(self.le4x_pressed)
        self.ui.le0y.returnPressed.connect(self.le0y_pressed)
        self.ui.le1y.returnPressed.connect(self.le1y_pressed)
        self.ui.le2y.returnPressed.connect(self.le2y_pressed)
        self.ui.le3y.returnPressed.connect(self.le3y_pressed)
        self.ui.le4y.returnPressed.connect(self.le4y_pressed)

        # Input self double position when return pressed
        self.ui.dle0x.returnPressed.connect(self.dle0x_pressed)
        self.ui.dle1x.returnPressed.connect(self.dle1x_pressed)
        self.ui.dle2x.returnPressed.connect(self.dle2x_pressed)
        self.ui.dle3x.returnPressed.connect(self.dle3x_pressed)
        self.ui.dle4x.returnPressed.connect(self.dle4x_pressed)
        self.ui.dle0y.returnPressed.connect(self.dle0y_pressed)
        self.ui.dle1y.returnPressed.connect(self.dle1y_pressed)
        self.ui.dle2y.returnPressed.connect(self.dle2y_pressed)
        self.ui.dle3y.returnPressed.connect(self.dle3y_pressed)
        self.ui.dle4y.returnPressed.connect(self.dle4y_pressed)

        # Accept, decline, x only, or y only a pixel when radio button is clicked
        self.ui.a0.clicked.connect(self.a0_clicked)
        self.ui.r0.clicked.connect(self.r0_clicked)
        self.ui.x0.clicked.connect(self.x0_clicked)
        self.ui.y0.clicked.connect(self.y0_clicked)
        self.ui.a1.clicked.connect(self.a1_clicked)
        self.ui.r1.clicked.connect(self.r1_clicked)
        self.ui.x1.clicked.connect(self.x1_clicked)
        self.ui.y1.clicked.connect(self.y1_clicked)
        self.ui.a2.clicked.connect(self.a2_clicked)
        self.ui.r2.clicked.connect(self.r2_clicked)
        self.ui.x2.clicked.connect(self.x2_clicked)
        self.ui.y2.clicked.connect(self.y2_clicked)
        self.ui.a3.clicked.connect(self.a3_clicked)
        self.ui.r3.clicked.connect(self.r3_clicked)
        self.ui.x3.clicked.connect(self.x3_clicked)
        self.ui.y3.clicked.connect(self.y3_clicked)
        self.ui.a4.clicked.connect(self.a4_clicked)
        self.ui.r4.clicked.connect(self.r4_clicked)
        self.ui.x4.clicked.connect(self.x4_clicked)
        self.ui.y4.clicked.connect(self.y4_clicked)

    # Functions to set the fixed peak position when return is pressed
    def le0x_pressed(self):
        self.mypeakpos[0][self.holder[0]] = float(self.ui.le0x.text())
    def le1x_pressed(self):
        self.mypeakpos[0][self.holder[1]] = float(self.ui.le1x.text())
    def le2x_pressed(self):
        self.mypeakpos[0][self.holder[2]] = float(self.ui.le2x.text())
    def le3x_pressed(self):
        self.mypeakpos[0][self.holder[3]] = float(self.ui.le3x.text())
    def le4x_pressed(self):
        self.mypeakpos[0][self.holder[4]] = float(self.ui.le4x.text())
    def le0y_pressed(self):
        self.mypeakpos[1][self.holder[0]] = float(self.ui.le0y.text())
    def le1y_pressed(self):
        self.mypeakpos[1][self.holder[1]] = float(self.ui.le1y.text())
    def le2y_pressed(self):
        self.mypeakpos[1][self.holder[2]] = float(self.ui.le2y.text())
    def le3y_pressed(self):
        self.mypeakpos[1][self.holder[3]] = float(self.ui.le3y.text())
    def le4y_pressed(self):
        self.mypeakpos[1][self.holder[4]] = float(self.ui.le4y.text())

    # Functions to set the fixed peak position when return is pressed
    def dle0x_pressed(self):
        self.doublepos[0][self.holder[0]] = float(self.ui.dle0x.text())
    def dle1x_pressed(self):
        self.doublepos[0][self.holder[1]] = float(self.ui.dle1x.text())
    def dle2x_pressed(self):
        self.doublepos[0][self.holder[2]] = float(self.ui.dle2x.text())
    def dle3x_pressed(self):
        self.doublepos[0][self.holder[3]] = float(self.ui.dle3x.text())
    def dle4x_pressed(self):
        self.doublepos[0][self.holder[4]] = float(self.ui.dle4x.text())
    def dle0y_pressed(self):
        self.doublepos[1][self.holder[0]] = float(self.ui.dle0y.text())
    def dle1y_pressed(self):
        self.doublepos[1][self.holder[1]] = float(self.ui.dle1y.text())
    def dle2y_pressed(self):
        self.doublepos[1][self.holder[2]] = float(self.ui.dle2y.text())
    def dle3y_pressed(self):
        self.doublepos[1][self.holder[3]] = float(self.ui.dle3y.text())
    def dle4y_pressed(self):
        self.doublepos[1][self.holder[4]] = float(self.ui.dle4y.text())

    # Functions to set the flags and update radio buttons when a radio button is clicked
    def a0_clicked(self):
        self.flagarray[self.holder[0]] = 0
        self.ui.a0.setChecked(True)
        self.ui.r0.setChecked(False)
        self.ui.x0.setChecked(False)
        self.ui.y0.setChecked(False)
    def r0_clicked(self):
        self.flagarray[self.holder[0]] = 1
        self.ui.a0.setChecked(False)
        self.ui.r0.setChecked(True)
        self.ui.x0.setChecked(False)
        self.ui.y0.setChecked(False)
    def x0_clicked(self):
        self.flagarray[self.holder[0]] = 2
        self.ui.a0.setChecked(False)
        self.ui.r0.setChecked(False)
        self.ui.x0.setChecked(True)
        self.ui.y0.setChecked(False)
    def y0_clicked(self):
        self.flagarray[self.holder[0]] = 3
        self.ui.a0.setChecked(False)
        self.ui.r0.setChecked(False)
        self.ui.x0.setChecked(False)
        self.ui.y0.setChecked(True)
    def a1_clicked(self):
        self.flagarray[self.holder[1]] = 0
        self.ui.a1.setChecked(True)
        self.ui.r1.setChecked(False)
        self.ui.x1.setChecked(False)
        self.ui.y1.setChecked(False)
    def r1_clicked(self):
        self.flagarray[self.holder[1]] = 1
        self.ui.a1.setChecked(False)
        self.ui.r1.setChecked(True)
        self.ui.x1.setChecked(False)
        self.ui.y1.setChecked(False)
    def x1_clicked(self):
        self.flagarray[self.holder[1]] = 2
        self.ui.a1.setChecked(False)
        self.ui.r1.setChecked(False)
        self.ui.x1.setChecked(True)
        self.ui.y1.setChecked(False)
    def y1_clicked(self):
        self.flagarray[self.holder[1]] = 3
        self.ui.a1.setChecked(False)
        self.ui.r1.setChecked(False)
        self.ui.x1.setChecked(False)
        self.ui.y1.setChecked(True)
    def a2_clicked(self):
        self.flagarray[self.holder[2]] = 0
        self.ui.a2.setChecked(True)
        self.ui.r2.setChecked(False)
        self.ui.x2.setChecked(False)
        self.ui.y2.setChecked(False)
    def r2_clicked(self):
        self.flagarray[self.holder[2]] = 1
        self.ui.a2.setChecked(False)
        self.ui.r2.setChecked(True)
        self.ui.x2.setChecked(False)
        self.ui.y2.setChecked(False)
    def x2_clicked(self):
        self.flagarray[self.holder[2]] = 2
        self.ui.a2.setChecked(False)
        self.ui.r2.setChecked(False)
        self.ui.x2.setChecked(True)
        self.ui.y2.setChecked(False)
    def y2_clicked(self):
        self.flagarray[self.holder[2]] = 3
        self.ui.a2.setChecked(False)
        self.ui.r2.setChecked(False)
        self.ui.x2.setChecked(False)
        self.ui.y2.setChecked(True)
    def a3_clicked(self):
        self.flagarray[self.holder[3]] = 0
        self.ui.a3.setChecked(True)
        self.ui.r3.setChecked(False)
        self.ui.x3.setChecked(False)
        self.ui.y3.setChecked(False)
    def r3_clicked(self):
        self.flagarray[self.holder[3]] = 1
        self.ui.a3.setChecked(False)
        self.ui.r3.setChecked(True)
        self.ui.x3.setChecked(False)
        self.ui.y3.setChecked(False)
    def x3_clicked(self):
        self.flagarray[self.holder[3]] = 2
        self.ui.a3.setChecked(False)
        self.ui.r3.setChecked(False)
        self.ui.x3.setChecked(True)
        self.ui.y3.setChecked(False)
    def y3_clicked(self):
        self.flagarray[self.holder[3]] = 3
        self.ui.a3.setChecked(False)
        self.ui.r3.setChecked(False)
        self.ui.x3.setChecked(False)
        self.ui.y3.setChecked(True)
    def a4_clicked(self):
        self.flagarray[self.holder[4]] = 0
        self.ui.a4.setChecked(True)
        self.ui.r4.setChecked(False)
        self.ui.x4.setChecked(False)
        self.ui.y4.setChecked(False)
    def r4_clicked(self):
        self.flagarray[self.holder[4]] = 1
        self.ui.a4.setChecked(False)
        self.ui.r4.setChecked(True)
        self.ui.x4.setChecked(False)
        self.ui.y4.setChecked(False)
    def x4_clicked(self):
        self.flagarray[self.holder[4]] = 2
        self.ui.a4.setChecked(False)
        self.ui.r4.setChecked(False)
        self.ui.x4.setChecked(True)
        self.ui.y4.setChecked(False)
    def y4_clicked(self):
        self.flagarray[self.holder[4]] = 3
        self.ui.a4.setChecked(False)
        self.ui.r4.setChecked(False)
        self.ui.x4.setChecked(False)
        self.ui.y4.setChecked(True)

    def calculate_plot_data(self):
        for roachno in range(number_of_roaches):
            for pixelno in range(int(roach_pixel_count[roachno])):

                pn = []
                data = np.empty(((len(xsweep),exptime_x[0])), dtype = object)
                for i in range(len(xsweep)):
                    pn.append('/r%d/p%d/t%d' % ( roachno ,pixelno, ts_x[i]))       
                try:
                    for i in range(len(xsweep)):
                        data[i][:] = h5file_x[i].root._f_getChild(pn[i]).read()
                    for j in xrange(0,exptime_x[0]):
                        median_array = []
                        for i in range(len(xsweep)):
                            median_array.append(len(data[i][j]))
                        self.crx_median[roachno*ppr + pixelno][j] = np.median(median_array)
                        for i in range(len(xsweep)):
                            self.crx[i][roachno*ppr + pixelno][j] = len(data[i][j])
                except:
                    pass

                pn = []
                data = np.empty(((len(ysweep),exptime_y[0])), dtype = object)
                for i in range(len(ysweep)):
                    pn.append('/r%d/p%d/t%d' % ( roachno ,pixelno, ts_y[i]))       
                try:
                    for i in range(len(ysweep)):
                        data[i][:] = h5file_y[i].root._f_getChild(pn[i]).read()
                    for j in xrange(0,exptime_y[0]):
                        median_array = []
                        for i in range(len(ysweep)):
                            median_array.append(len(data[i][j]))
                        self.cry_median[roachno*ppr + pixelno][j] = np.median(median_array)
                        for i in range(len(ysweep)):
                            self.cry[i][roachno*ppr + pixelno][j] = len(data[i][j])
                except:
                    pass

            
            print 'roach', str(roachno), 'done'

    def perform_fits(self):
        for roachno in range(number_of_roaches):
            for pixelno in range(int(roach_pixel_count[roachno])):
                self.xpeakguess=np.where(self.crx_median[roachno*ppr + pixelno][:] == self.crx_median[roachno*ppr + pixelno][:].max())[0][0]
                self.xfitstart=max([self.xpeakguess-20,0])
                self.xfitend=min([self.xpeakguess+20,len(self.xvals)])
                params_x = fitgaussian(self.crx_median[roachno*ppr + pixelno][self.xfitstart:self.xfitend],self.xvals[self.xfitstart:self.xfitend])
                self.xfit[roachno*ppr + pixelno][:] = gaussian(params_x,self.xvals)
                self.peakpos[0][roachno*ppr + pixelno] = params_x[0]
                self.mypeakpos[0][roachno*ppr + pixelno] = params_x[0]

                self.ypeakguess=np.where(self.cry_median[roachno*ppr + pixelno][:] == self.cry_median[roachno*ppr + pixelno][:].max())[0][0]
                self.yfitstart=max([self.ypeakguess-20,0])
                self.yfitend=min([self.ypeakguess+20,len(self.yvals)])
                params_y = fitgaussian(self.cry_median[roachno*ppr + pixelno][self.yfitstart:self.yfitend],self.yvals[self.yfitstart:self.yfitend])
                self.yfit[roachno*ppr + pixelno][:] = gaussian(params_y,self.yvals)
                self.peakpos[1][roachno*ppr + pixelno] = params_y[0]
                self.mypeakpos[1][roachno*ppr + pixelno] = params_y[0]

    def enlarge0x(self):
        plt.clf()
        plt.plot(self.xvals,self.crx_median[self.holder[0]][:])
        plt.plot(self.xvals,self.xfit[self.holder[0]][:])
        for i in range(len(xsweep)):
            plt.plot(self.xvals,self.crx[i][self.holder[0]][:],alpha = .2)
        plt.show()
    def enlarge0y(self):
        plt.clf()
        plt.plot(self.yvals,self.cry_median[self.holder[0]][:])
        plt.plot(self.yvals,self.yfit[self.holder[0]][:])
        for i in range(len(ysweep)):
            plt.plot(self.yvals,self.cry[i][self.holder[0]][:],alpha = .2)
        plt.show()
    def enlarge1x(self):
        plt.clf()
        plt.plot(self.xvals,self.crx_median[self.holder[1]][:])
        plt.plot(self.xvals,self.xfit[self.holder[1]][:])
        for i in range(len(xsweep)):
            plt.plot(self.xvals,self.crx[i][self.holder[1]][:],alpha = .2)
        plt.show()
    def enlarge1y(self):
        plt.clf()
        plt.plot(self.yvals,self.cry_median[self.holder[1]][:])
        plt.plot(self.yvals,self.yfit[self.holder[1]][:])
        for i in range(len(ysweep)):
            plt.plot(self.yvals,self.cry[i][self.holder[1]][:],alpha = .2)
        plt.show()
    def enlarge2x(self):
        plt.clf()
        plt.plot(self.xvals,self.crx_median[self.holder[2]][:])
        plt.plot(self.xvals,self.xfit[self.holder[2]][:])
        for i in range(len(xsweep)):
            plt.plot(self.xvals,self.crx[i][self.holder[2]][:],alpha = .2)
        plt.show()
    def enlarge2y(self):
        plt.clf()
        plt.plot(self.yvals,self.cry_median[self.holder[2]][:])
        plt.plot(self.yvals,self.yfit[self.holder[2]][:])
        for i in range(len(ysweep)):
            plt.plot(self.yvals,self.cry[i][self.holder[2]][:],alpha = .2)
        plt.show()
    def enlarge3x(self):
        plt.clf()
        plt.plot(self.xvals,self.crx_median[self.holder[3]][:])
        plt.plot(self.xvals,self.xfit[self.holder[3]][:])
        for i in range(len(xsweep)):
            plt.plot(self.xvals,self.crx[i][self.holder[3]][:],alpha = .2)
        plt.show()
    def enlarge3y(self):
        plt.clf()
        plt.plot(self.yvals,self.cry_median[self.holder[3]][:])
        plt.plot(self.yvals,self.yfit[self.holder[3]][:])
        for i in range(len(ysweep)):
            plt.plot(self.yvals,self.cry[i][self.holder[3]][:],alpha = .2)
        plt.show()
    def enlarge4x(self):
        plt.clf()
        plt.plot(self.xvals,self.crx_median[self.holder[4]][:])
        plt.plot(self.xvals,self.xfit[self.holder[4]][:])
        for i in range(len(xsweep)):
            plt.plot(self.xvals,self.crx[i][self.holder[4]][:],alpha = .2)
        plt.show()
    def enlarge4y(self):
        plt.clf()
        plt.plot(self.yvals,self.cry_median[self.holder[4]][:])
        plt.plot(self.yvals,self.yfit[self.holder[4]][:])
        for i in range(len(ysweep)):
            plt.plot(self.yvals,self.cry[i][self.holder[4]][:],alpha = .2)
        plt.show()

    def make_plots(self):
        self.ui.mapplot_0x.canvas.ax.clear()
        self.ui.mapplot_0y.canvas.ax.clear()
        self.ui.mapplot_1x.canvas.ax.clear()
        self.ui.mapplot_1y.canvas.ax.clear()
        self.ui.mapplot_2x.canvas.ax.clear()
        self.ui.mapplot_2y.canvas.ax.clear()
        self.ui.mapplot_3x.canvas.ax.clear()
        self.ui.mapplot_3y.canvas.ax.clear()
        self.ui.mapplot_4x.canvas.ax.clear()
        self.ui.mapplot_4y.canvas.ax.clear()
        
        self.ui.mapplot_0x.canvas.ax.plot(self.xvals,self.crx_median[self.holder[0]][:])       
        self.ui.mapplot_0y.canvas.ax.plot(self.yvals,self.cry_median[self.holder[0]][:])        
        self.ui.mapplot_1x.canvas.ax.plot(self.xvals,self.crx_median[self.holder[1]][:])        
        self.ui.mapplot_1y.canvas.ax.plot(self.yvals,self.cry_median[self.holder[1]][:])       
        self.ui.mapplot_2x.canvas.ax.plot(self.xvals,self.crx_median[self.holder[2]][:])        
        self.ui.mapplot_2y.canvas.ax.plot(self.yvals,self.cry_median[self.holder[2]][:])        
        self.ui.mapplot_3x.canvas.ax.plot(self.xvals,self.crx_median[self.holder[3]][:])       
        self.ui.mapplot_3y.canvas.ax.plot(self.yvals,self.cry_median[self.holder[3]][:])        
        self.ui.mapplot_4x.canvas.ax.plot(self.xvals,self.crx_median[self.holder[4]][:])        
        self.ui.mapplot_4y.canvas.ax.plot(self.yvals,self.cry_median[self.holder[4]][:])

        self.ui.mapplot_0x.canvas.ax.plot(self.xvals,self.xfit[self.holder[0]][:])       
        self.ui.mapplot_0y.canvas.ax.plot(self.yvals,self.yfit[self.holder[0]][:])        
        self.ui.mapplot_1x.canvas.ax.plot(self.xvals,self.xfit[self.holder[1]][:])        
        self.ui.mapplot_1y.canvas.ax.plot(self.yvals,self.yfit[self.holder[1]][:])       
        self.ui.mapplot_2x.canvas.ax.plot(self.xvals,self.xfit[self.holder[2]][:])        
        self.ui.mapplot_2y.canvas.ax.plot(self.yvals,self.yfit[self.holder[2]][:])        
        self.ui.mapplot_3x.canvas.ax.plot(self.xvals,self.xfit[self.holder[3]][:])       
        self.ui.mapplot_3y.canvas.ax.plot(self.yvals,self.yfit[self.holder[3]][:])        
        self.ui.mapplot_4x.canvas.ax.plot(self.xvals,self.xfit[self.holder[4]][:])        
        self.ui.mapplot_4y.canvas.ax.plot(self.yvals,self.yfit[self.holder[4]][:])

        for i in range(len(xsweep)):
            self.ui.mapplot_0x.canvas.ax.plot(self.xvals,self.crx[i][self.holder[0]][:],alpha = .2)                          
            self.ui.mapplot_1x.canvas.ax.plot(self.xvals,self.crx[i][self.holder[1]][:],alpha = .2)                         
            self.ui.mapplot_2x.canvas.ax.plot(self.xvals,self.crx[i][self.holder[2]][:],alpha = .2)                           
            self.ui.mapplot_3x.canvas.ax.plot(self.xvals,self.crx[i][self.holder[3]][:],alpha = .2)                          
            self.ui.mapplot_4x.canvas.ax.plot(self.xvals,self.crx[i][self.holder[4]][:],alpha = .2)                  

        for i in range(len(ysweep)):
            self.ui.mapplot_0y.canvas.ax.plot(self.yvals,self.cry[i][self.holder[0]][:],alpha = .2)
            self.ui.mapplot_1y.canvas.ax.plot(self.yvals,self.cry[i][self.holder[1]][:],alpha = .2)
            self.ui.mapplot_2y.canvas.ax.plot(self.yvals,self.cry[i][self.holder[2]][:],alpha = .2)
            self.ui.mapplot_3y.canvas.ax.plot(self.yvals,self.cry[i][self.holder[3]][:],alpha = .2)
            self.ui.mapplot_4y.canvas.ax.plot(self.yvals,self.cry[i][self.holder[4]][:],alpha = .2)

        self.ui.mapplot_0x.canvas.draw()
        self.ui.mapplot_0y.canvas.draw()
        self.ui.mapplot_1x.canvas.draw()
        self.ui.mapplot_1y.canvas.draw()
        self.ui.mapplot_2x.canvas.draw()
        self.ui.mapplot_2y.canvas.draw()
        self.ui.mapplot_3x.canvas.draw()
        self.ui.mapplot_3y.canvas.draw()
        self.ui.mapplot_4x.canvas.draw()
        self.ui.mapplot_4y.canvas.draw()
        
    def next_process(self):
        
        # Switch index to next 5 plots
        if (self.holder[4] + 1 == (number_of_roaches-1)*ppr + roach_pixel_count[number_of_roaches - 1]):
            self.currentroach = 0
            self.ui.roachLabel.setText('Roach ' + str(self.currentroach))
            self.holder = [0,1,2,3,4]
        elif ((self.holder[4] + 1 == self.currentroach*ppr + roach_pixel_count[self.currentroach]) and (self.currentroach != number_of_roaches - 1)):
            self.currentroach += 1
            self.ui.roachLabel.setText('Roach ' + str(self.currentroach))
            for i in range(5):
                self.holder[i] = int(self.currentroach*ppr + i)
        elif (self.holder[4] + 6 <= self.currentroach*ppr + roach_pixel_count[self.currentroach]):
            for i in range(5):
                self.holder[i]+=5
        elif (self.holder[4] + 6 > self.currentroach*ppr + roach_pixel_count[self.currentroach]):
            for i in range(5):
                self.holder[4-i] = int(self.currentroach*ppr + roach_pixel_count[self.currentroach] - 1 - i)

        # Update the gui
        self.update_buttons()
        
        # Draw the new set of plots
        self.make_plots()

    def go_process(self):
        roachno = int(self.ui.roachgo.text())
        pixno = int(self.ui.pixelgo.text())

        if (pixno+5 <= roach_pixel_count[roachno]):
            for i in range(5):
                self.holder[i] = int(roachno*253 + pixno + i)
        else:
            for i in range(5):
                self.holder[4-i] = int(roachno*253 + roach_pixel_count[roachno] - i -1)

        self.update_buttons()

        self.make_plots()

    def update_buttons(self):

        # Update pixel number in pixel labels  
        self.ui.pix0.setText('Pixel ' + str(self.holder[0]%ppr))
        self.ui.pix1.setText('Pixel ' + str(self.holder[1]%ppr))
        self.ui.pix2.setText('Pixel ' + str(self.holder[2]%ppr))
        self.ui.pix3.setText('Pixel ' + str(self.holder[3]%ppr))
        self.ui.pix4.setText('Pixel ' + str(self.holder[4]%ppr))

        # Update radio state, default checked true
        if (self.flagarray[self.holder[0]] == 0):
            self.ui.a0.setChecked(True)
            self.ui.r0.setChecked(False)
            self.ui.x0.setChecked(False)
            self.ui.y0.setChecked(False)
        elif (self.flagarray[self.holder[0]] == 1):
            self.ui.a0.setChecked(False)
            self.ui.r0.setChecked(True)
            self.ui.x0.setChecked(False)
            self.ui.y0.setChecked(False)
        elif (self.flagarray[self.holder[0]] == 2):
            self.ui.a0.setChecked(False)
            self.ui.r0.setChecked(False)
            self.ui.x0.setChecked(True)
            self.ui.y0.setChecked(False)
        elif (self.flagarray[self.holder[0]] == 3):
            self.ui.a0.setChecked(False)
            self.ui.r0.setChecked(False)
            self.ui.x0.setChecked(False)
            self.ui.y0.setChecked(True)
        if (self.flagarray[self.holder[1]] == 0):
            self.ui.a1.setChecked(True)
            self.ui.r1.setChecked(False)
            self.ui.x1.setChecked(False)
            self.ui.y1.setChecked(False)
        elif (self.flagarray[self.holder[1]] == 1):
            self.ui.a1.setChecked(False)
            self.ui.r1.setChecked(True)
            self.ui.x1.setChecked(False)
            self.ui.y1.setChecked(False)
        elif (self.flagarray[self.holder[1]] == 2):
            self.ui.a1.setChecked(False)
            self.ui.r1.setChecked(False)
            self.ui.x1.setChecked(True)
            self.ui.y1.setChecked(False)
        elif (self.flagarray[self.holder[1]] == 3):
            self.ui.a1.setChecked(False)
            self.ui.r1.setChecked(False)
            self.ui.x1.setChecked(False)
            self.ui.y1.setChecked(True)
        if (self.flagarray[self.holder[2]] == 0):
            self.ui.a2.setChecked(True)
            self.ui.r2.setChecked(False)
            self.ui.x2.setChecked(False)
            self.ui.y2.setChecked(False)
        elif (self.flagarray[self.holder[2]] == 1):
            self.ui.a2.setChecked(False)
            self.ui.r2.setChecked(True)
            self.ui.x2.setChecked(False)
            self.ui.y2.setChecked(False)
        elif (self.flagarray[self.holder[2]] == 2):
            self.ui.a2.setChecked(False)
            self.ui.r2.setChecked(False)
            self.ui.x2.setChecked(True)
            self.ui.y2.setChecked(False)
        elif (self.flagarray[self.holder[2]] == 3):
            self.ui.a2.setChecked(False)
            self.ui.r2.setChecked(False)
            self.ui.x2.setChecked(False)
            self.ui.y2.setChecked(True)
        if (self.flagarray[self.holder[3]] == 0):
            self.ui.a3.setChecked(True)
            self.ui.r3.setChecked(False)
            self.ui.x3.setChecked(False)
            self.ui.y3.setChecked(False)
        elif (self.flagarray[self.holder[3]] == 1):
            self.ui.a3.setChecked(False)
            self.ui.r3.setChecked(True)
            self.ui.x3.setChecked(False)
            self.ui.y3.setChecked(False)
        elif (self.flagarray[self.holder[3]] == 2):
            self.ui.a3.setChecked(False)
            self.ui.r3.setChecked(False)
            self.ui.x3.setChecked(True)
            self.ui.y3.setChecked(False)
        elif (self.flagarray[self.holder[3]] == 3):
            self.ui.a3.setChecked(False)
            self.ui.r3.setChecked(False)
            self.ui.x3.setChecked(False)
            self.ui.y3.setChecked(True)
        if (self.flagarray[self.holder[4]] == 0):
            self.ui.a4.setChecked(True)
            self.ui.r4.setChecked(False)
            self.ui.x4.setChecked(False)
            self.ui.y4.setChecked(False)
        elif (self.flagarray[self.holder[4]] == 1):
            self.ui.a4.setChecked(False)
            self.ui.r4.setChecked(True)
            self.ui.x4.setChecked(False)
            self.ui.y4.setChecked(False)
        elif (self.flagarray[self.holder[4]] == 2):
            self.ui.a4.setChecked(False)
            self.ui.r4.setChecked(False)
            self.ui.x4.setChecked(True)
            self.ui.y4.setChecked(False)
        elif (self.flagarray[self.holder[4]] == 3):
            self.ui.a4.setChecked(False)
            self.ui.r4.setChecked(False)
            self.ui.x4.setChecked(False)
            self.ui.y4.setChecked(True)

        # Update the fit peak position labels
        self.ui.pp0x.setText(str(self.peakpos[0][self.holder[0]]))
        self.ui.pp0y.setText(str(self.peakpos[1][self.holder[0]]))
        self.ui.pp1x.setText(str(self.peakpos[0][self.holder[1]]))
        self.ui.pp1y.setText(str(self.peakpos[1][self.holder[1]]))
        self.ui.pp2x.setText(str(self.peakpos[0][self.holder[2]]))
        self.ui.pp2y.setText(str(self.peakpos[1][self.holder[2]]))
        self.ui.pp3x.setText(str(self.peakpos[0][self.holder[3]]))
        self.ui.pp3y.setText(str(self.peakpos[1][self.holder[3]]))
        self.ui.pp4x.setText(str(self.peakpos[0][self.holder[4]]))
        self.ui.pp4y.setText(str(self.peakpos[1][self.holder[4]]))

        # Update the self peak position line edits
        self.ui.le0x.setText(str(self.mypeakpos[0][self.holder[0]]))
        self.ui.le0y.setText(str(self.mypeakpos[1][self.holder[0]]))
        self.ui.le1x.setText(str(self.mypeakpos[0][self.holder[1]]))
        self.ui.le1y.setText(str(self.mypeakpos[1][self.holder[1]]))
        self.ui.le2x.setText(str(self.mypeakpos[0][self.holder[2]]))
        self.ui.le2y.setText(str(self.mypeakpos[1][self.holder[2]]))
        self.ui.le3x.setText(str(self.mypeakpos[0][self.holder[3]]))
        self.ui.le3y.setText(str(self.mypeakpos[1][self.holder[3]]))
        self.ui.le4x.setText(str(self.mypeakpos[0][self.holder[4]]))
        self.ui.le4y.setText(str(self.mypeakpos[1][self.holder[4]]))
        
        # Update the double peak position line edits
        self.ui.dle0x.setText(str(self.doublepos[0][self.holder[0]]))
        self.ui.dle0y.setText(str(self.doublepos[1][self.holder[0]]))
        self.ui.dle1x.setText(str(self.doublepos[0][self.holder[1]]))
        self.ui.dle1y.setText(str(self.doublepos[1][self.holder[1]]))
        self.ui.dle2x.setText(str(self.doublepos[0][self.holder[2]]))
        self.ui.dle2y.setText(str(self.doublepos[1][self.holder[2]]))
        self.ui.dle3x.setText(str(self.doublepos[0][self.holder[3]]))
        self.ui.dle3y.setText(str(self.doublepos[1][self.holder[3]]))
        self.ui.dle4x.setText(str(self.doublepos[0][self.holder[4]]))
        self.ui.dle4y.setText(str(self.doublepos[1][self.holder[4]]))
        

    def save_process(self):
        d=open(input_params[3] + '/doubles.pos','w')
        for roachno in range(number_of_roaches):
            f=open(input_params[3] + '/r%i.pos' %roachno,'w')
            for pixelno in range(int(roach_pixel_count[roachno])):
                f=open(input_params[3]+'/r%i.pos' %roachno,'a')
                if self.flagarray[roachno*ppr + pixelno] == 0:
                    f.write(str(self.mypeakpos[0,pixelno + roachno*ppr])+'\t'+str(self.mypeakpos[1,pixelno + roachno*ppr])+'\t0\n')
                elif self.flagarray[roachno*ppr + pixelno] == 1:
                    f.write('0.0\t0.0\t1\n')
                elif self.flagarray[roachno*ppr + pixelno] == 2:
                    f.write(str(self.mypeakpos[0,pixelno + roachno*ppr])+'\t0.0\t2\n')
                elif self.flagarray[roachno*ppr + pixelno] == 3:
                    f.write('0.0\t'+str(self.mypeakpos[1,pixelno + roachno*ppr])+'\t3\n')

                if (self.doublepos[0][roachno*ppr + pixelno] !=0 or self.doublepos[1][roachno*ppr + pixelno] !=0):
                    d=open(input_params[3] + '/doubles.pos','a')
                    d.write(str(self.mypeakpos[0,pixelno + roachno*ppr])+'\t'+str(self.mypeakpos[1,pixelno + roachno*ppr])+ '\t' +str(self.doublepos[0,pixelno + roachno*ppr])+'\t'+str(self.doublepos[1,pixelno + roachno*ppr])+ '\t/r' + str(roachno) + '/p' + str(pixelno) + '\n')
                    d.close()
                    
                f.close()
'''    
    # Try to find a peak position by manually selecting an approximate peak location
    def on_click(self,event):
    # If x sweep plot (top plot) is clicked
        if(event.y > 250):
            self.xvals=np.arange(len(self.crx_median[pixelno][:]))
            self.xpeakguess=event.xdata
            self.xfitstart=max([self.xpeakguess-20,0])
            self.xfitend=min([self.xpeakguess+20,len(self.xvals)])
            params = fitgaussian(self.crx_median[pixelno][self.xfitstart:self.xfitend],self.xvals[self.xfitstart:self.xfitend])
            self.xfit = gaussian(params,self.xvals)
            self.peakpos[0,self.pixelno]=params[0]
        # If y sweep plot (bottom plot) is clicked
        else:
            self.yvals=np.arange(len(self.cry_median[pixelno][:]))
            self.ypeakguess=event.xdata
            self.yfitstart=max([self.ypeakguess-20,0])
            self.yfitend=min([self.ypeakguess+20,len(self.yvals)])
            params = fitgaussian(self.cry_median[pixelno][self.yfitstart:self.yfitend],self.yvals[self.yfitstart:self.yfitend])
            self.yfit = gaussian(params,self.yvals)
            self.peakpos[1,self.pixelno]=params[0]
    # Connect to plot
    def connect(self):
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
'''

# Set input parameters and sweep files
input_params = numberrun()
print input_params
xsweep = xrun()
ysweep = yrun()
print 'X Sweep Files:', xsweep
print 'Y Sweep Files:', ysweep

number_of_roaches = input_params[4]
roach_pixel_count = np.zeros(number_of_roaches)
for roachno in xrange(0,number_of_roaches):
    roach_pixel_count[roachno] = file_len(input_params[2] + '/ps_freq%i.txt' %roachno)-1
maximum_pixels = input_params[5]
ppr = maximum_pixels/number_of_roaches

# Load the input files
# X sweep data
h5file_x = []
ts_x = []
exptime_x = []
for i in range(len(xsweep)):
#    h5file_x.append(openFile(path + xsweep[i], mode = 'r'))
    h5file_x.append(openFile(xsweep[i], mode = 'r'))
    try:
        ts_x.append(int(h5file_x[i].root.header.header.col('unixtime')[0]))
    except KeyError:
        ts_x.append(int(h5file_x[i].root.header.header.col('ut')[0]))
    exptime_x.append(int(h5file_x[i].root.header.header.col('exptime')[0]))
# Y sweep data
h5file_y = []
ts_y = []
exptime_y = []
for i in range(len(ysweep)):
#    h5file_y.append(openFile(path + ysweep[i], mode = 'r'))
    h5file_y.append(openFile(ysweep[i], mode = 'r'))
    try:
        ts_y.append(int(h5file_y[i].root.header.header.col('unixtime')[0]))
    except KeyError:
        ts_y.append(int(h5file_y[i].root.header.header.col('ut')[0]))
    exptime_y.append(int(h5file_y[i].root.header.header.col('exptime')[0]))


# Start up main gui
if __name__ == "__main__":
	app = QApplication(sys.argv)
	myapp = StartQt4(exptime_x[0],exptime_y[0],len(xsweep),len(ysweep))
	myapp.show()
	app.exec_()

# Close the h5 files in the end
for i in range(len(h5file_x)):
    h5file_x[i].close()
for i in range(len(h5file_y)):
    h5file_y[i].close()
