import numpy as np
import matplotlib.pyplot as plt
import sys

from PyQt4.QtGui import *
from PyQt4.QtGui import *
from pixels_gui import Ui_pixels_gui

from readDict import readDict

class StartQt4(QMainWindow):
    def __init__(self):
        QWidget.__init__(self, parent=None)
        self.ui = Ui_pixels_gui()
        self.ui.setupUi(self)

        # Initialize variables
        self.xpos=np.empty(0,dtype='float32')
        self.ypos=np.empty(0,dtype='float32')
        self.xpos0=np.empty(0,dtype='float32')
        self.ypos0=np.empty(0,dtype='float32')
        self.xpos1=np.empty(0,dtype='float32')
        self.ypos1=np.empty(0,dtype='float32')
        self.xvals=np.empty(0,dtype='float32')
        self.yvals=np.empty(0,dtype='float32')
        self.freqvals=np.empty(0,dtype='float32')
        self.attenvals=np.empty(0,dtype='float32')
        self.goodpixtag=np.empty(0,dtype='|S10')

        self.xsize=80;
        self.ysize=125;
        self.pixNumber = self.xsize*self.ysize
        
        self.pscale=10;
        self.xoff=0;
        self.yoff=0;
        
        self.angle=0;
        self.s=np.sin(np.pi*self.angle/180.)
        self.c=np.cos(np.pi*self.angle/180.)

        self.ui.pixPlot.canvas.ax.clear()
        
        self.ui.dirbtn.clicked.connect(self.dir_choose)
        self.ui.savebtn.clicked.connect(self.save_process)
        self.ui.dblbtn.clicked.connect(self.dbl_process)
        self.ui.hidebtn.clicked.connect(self.hide_process)
        self.ui.anglele.returnPressed.connect(self.anglele_pressed)
        self.ui.scalele.returnPressed.connect(self.scalele_pressed)
        self.ui.xoffle.returnPressed.connect(self.xoffle_pressed)
        self.ui.yoffle.returnPressed.connect(self.yoffle_pressed)
        

    def anglele_pressed(self):
        self.angle=float(self.ui.anglele.text())
        self.s=np.sin(np.pi*self.angle/180.)
        self.c=np.cos(np.pi*self.angle/180.)

        self.ui.pixPlot.canvas.ax.clear()
        self.plot_grid()
        self.ui.pixPlot.canvas.ax.plot(self.xvals[self.mask],self.yvals[self.mask],'b+')
        self.ui.pixPlot.canvas.draw()

    def scalele_pressed(self):
        self.pscale=float(self.ui.scalele.text())
        self.scale=[self.pscale]

        self.xvals=np.empty(0,dtype='float32')
        self.yvals=np.empty(0,dtype='float32')

        # Can fix this later to pick out individual roaches
        self.xvals = self.xpos/self.scale[0]
        self.xvals += self.origin[0][0]
        self.yvals = self.ypos/self.scale[0]
        self.yvals += self.origin[0][1]

        self.xpos0=np.empty(0,dtype='float32')
        self.ypos0=np.empty(0,dtype='float32')
        self.xpos1=np.empty(0,dtype='float32')
        self.ypos1=np.empty(0,dtype='float32')

        self.xpos0 = self.xpost0/self.scale[0]
        self.xpos0 += self.origin[0][0]
        self.ypos0 = self.ypost0/self.scale[0]
        self.ypos0 += self.origin[0][1]

        self.xpos1 = self.xpost1/self.scale[0]
        self.xpos1 += self.origin[0][0]
        self.ypos1 = self.ypost1/self.scale[0]
        self.ypos1 += self.origin[0][1]

        self.mask=(np.linspace(0,len(self.xvals)-1,len(self.xvals))).astype('int')
        self.ui.pixPlot.canvas.ax.clear()
        self.plot_grid()
        self.ui.pixPlot.canvas.ax.plot(self.xvals[self.mask],self.yvals[self.mask],'b+')
        self.ui.pixPlot.canvas.draw()

    def xoffle_pressed(self):
        self.xoff=float(self.ui.xoffle.text())
        #self.origin = np.zeros((len(self.infile),2))
        self.origin = np.zeros((1,2))
        #for i in range(len(self.infile)):
        for i in range(1):
            self.origin[i][0] = self.xoff
            self.origin[i][1] = self.yoff
        
        self.xvals=np.empty(0,dtype='float32')
        self.yvals=np.empty(0,dtype='float32')

        # Can fix this later to pick out individual roaches
        self.xvals = self.xpos/self.scale[0]
        self.xvals += self.origin[0][0]
        self.yvals = self.ypos/self.scale[0]
        self.yvals += self.origin[0][1]

        self.xpos0=np.empty(0,dtype='float32')
        self.ypos0=np.empty(0,dtype='float32')
        self.xpos1=np.empty(0,dtype='float32')
        self.ypos1=np.empty(0,dtype='float32')

        self.xpos0 = self.xpost0/self.scale[0]
        self.xpos0 += self.origin[0][0]
        self.ypos0 = self.ypost0/self.scale[0]
        self.ypos0 += self.origin[0][1]

        self.xpos1 = self.xpost1/self.scale[0]
        self.xpos1 += self.origin[0][0]
        self.ypos1 = self.ypost1/self.scale[0]
        self.ypos1 += self.origin[0][1]

        self.mask=(np.linspace(0,len(self.xvals)-1,len(self.xvals))).astype('int')
        self.ui.pixPlot.canvas.ax.clear()
        self.plot_grid()
        self.ui.pixPlot.canvas.ax.plot(self.xvals[self.mask],self.yvals[self.mask],'b+')
        self.ui.pixPlot.canvas.draw()
        
    def yoffle_pressed(self):
        self.yoff=float(self.ui.yoffle.text())
        self.origin = np.zeros((1,2))
        for i in range(1):
            self.origin[i][0] = self.xoff
            self.origin[i][1] = self.yoff

        self.xvals=np.empty(0,dtype='float32')
        self.yvals=np.empty(0,dtype='float32')

        # Can fix this later to pick out individual roaches
        self.xvals = self.xpos/self.scale[0]
        self.xvals += self.origin[0][0]
        self.yvals = self.ypos/self.scale[0]
        self.yvals += self.origin[0][1]

        self.xpos0=np.empty(0,dtype='float32')
        self.ypos0=np.empty(0,dtype='float32')
        self.xpos1=np.empty(0,dtype='float32')
        self.ypos1=np.empty(0,dtype='float32')

        self.xpos0 = self.xpost0/self.scale[0]
        self.xpos0 += self.origin[0][0]
        self.ypos0 = self.ypost0/self.scale[0]
        self.ypos0 += self.origin[0][1]

        self.xpos1 = self.xpost1/self.scale[0]
        self.xpos1 += self.origin[0][0]
        self.ypos1 = self.ypost1/self.scale[0]
        self.ypos1 += self.origin[0][1]

        self.mask=(np.linspace(0,len(self.xvals)-1,len(self.xvals))).astype('int')
        self.ui.pixPlot.canvas.ax.clear()
        self.plot_grid()
        self.ui.pixPlot.canvas.ax.plot(self.xvals[self.mask],self.yvals[self.mask],'b+')
        self.ui.pixPlot.canvas.draw()

    def dbl_process(self):

        for i in range(len(self.xpos0)):
            self.ui.pixPlot.canvas.ax.plot([self.xpos0[i],self.xpos1[i]],[self.ypos0[i],self.ypos1[i]],'D')
        self.ui.pixPlot.canvas.draw()

    def hide_process(self):
        self.xvals=np.empty(0,dtype='float32')
        self.yvals=np.empty(0,dtype='float32')

        # Can fix this later to pick out individual roaches
        self.xvals = self.xpos/self.scale[0]
        self.xvals += self.origin[0][0]
        self.yvals = self.ypos/self.scale[0]
        self.yvals += self.origin[0][1]

        self.mask=(np.linspace(0,len(self.xvals)-1,len(self.xvals))).astype('int')
        self.ui.pixPlot.canvas.ax.clear()
        self.plot_grid()
        self.ui.pixPlot.canvas.ax.plot(self.xvals[self.mask],self.yvals[self.mask],'b+')
        self.ui.pixPlot.canvas.draw()

        

    def plot_grid(self):
        # Vertical Lines
        xstart=self.xsize*np.linspace(0,1,self.xsize+1)
        ystart=np.zeros(self.xsize+1)
        xstop=xstart
        ystop=ystart+self.ysize

        xstart2=xstart*self.c - ystart*self.s
        xstop2=xstop*self.c - ystop*self.s
        ystart2=xstart*self.s + ystart*self.c
        ystop2=xstop*self.s + ystop*self.c

        for i in range(self.xsize+1):
            self.ui.pixPlot.canvas.ax.plot([xstart2[i],xstop2[i]],[ystart2[i],ystop2[i]],'g')

        # Horizontal Lines
        ystart=self.ysize*np.linspace(0,1,self.ysize+1)
        xstart=np.zeros(self.ysize+1)
        ystop=ystart
        xstop=xstart+self.xsize

        xstart2=xstart*self.c - ystart*self.s
        xstop2=xstop*self.c - ystop*self.s
        ystart2=xstart*self.s + ystart*self.c
        ystop2=xstop*self.s + ystop*self.c

        for i in range(self.ysize+1):
            self.ui.pixPlot.canvas.ax.plot([xstart2[i],xstop2[i]],[ystart2[i],ystop2[i]],'g')
    
    def dir_choose(self):
        #text =QFileDialog.getExistingDirectory()
        #self.path = str(text)
        #self.ui.dirle.setText(str(text))

        text =QFileDialog.getOpenFileName()
        self.configFileName=str(text)
        self.ui.dirle.setText(str(text))

        # Open config file
        self.configData = readDict()
        self.configData.read_from_file(self.configFileName)

        # Extract relevent file names from config file
        self.masterPositionList = str(self.configData['masterPositionList'])
        self.masterDoublesList = str(self.configData['masterDoublesList'])
        self.outputFilename = str(self.configData['outputFilename'])
        self.outputDoubleName = str(self.configData['outputDoubleName'])

        '''
        self.infile=[]       
        self.infile.append(self.path + '/r0.pos')
        self.infile.append(self.path + '/r1.pos')
        self.infile.append(self.path + '/r2.pos')
        self.infile.append(self.path + '/r3.pos')
        self.infile.append(self.path + '/r4.pos')
        self.infile.append(self.path + '/r5.pos')
        self.infile.append(self.path + '/r6.pos')
        self.infile.append(self.path + '/r7.pos')

        self.psfile=[]
        self.psfile.append(self.path + '/ps_freq0.txt')
        self.psfile.append(self.path + '/ps_freq1.txt')
        self.psfile.append(self.path + '/ps_freq2.txt')
        self.psfile.append(self.path + '/ps_freq3.txt')
        self.psfile.append(self.path + '/ps_freq4.txt')
        self.psfile.append(self.path + '/ps_freq5.txt')
        self.psfile.append(self.path + '/ps_freq6.txt')
        self.psfile.append(self.path + '/ps_freq7.txt')       
        '''
        
        self.origin = np.zeros((1,2))
        self.origin[0][0] = self.xoff
        self.origin[0][1] = self.yoff
        self.scale=[self.pscale]

        #self.origin = np.zeros((len(self.infile),2))
        #for i in range(len(self.infile)):
            #self.origin[i][0] = self.xoff
            #self.origin[i][1] = self.yoff
        #self.scale=[self.pscale]*len(self.infile)

        self.dblfile = self.masterDoublesList

        self.doublePixelNumber, self.xpost0, self.ypost0, self.xpost1, self.ypost1=np.loadtxt(self.dblfile,unpack='True', usecols = (0,1,2,3,4))

        self.pixelNumberAll, self.flag, self.xposAll, self.yposAll = np.loadtxt(self.masterPositionList,unpack='True')

        self.goodpix=np.where(self.flag == 0)[0]

        print len(self.xposAll[self.goodpix]), 'Good Pixels'

        self.pixelNumber = np.array(self.pixelNumberAll[self.goodpix], dtype=int)

        self.xpos=np.array(self.xposAll[self.goodpix], dtype=float)
        self.ypos=np.array(self.yposAll[self.goodpix], dtype=float)



        # Extract pixel number, flag, x position, y position from input file
        #pixeltag =''
        #for j in range(len(self.infile)):
            #print self.infile[j]
            #pixelNumber, flag, xpos, ypos = np.loadtxt(self.infile[j],unpack='True')

            #freq, xcenpos, ycenpos, atten=np.loadtxt(self.psfile[j],unpack='True',skiprows=1)       
        
            #self.goodpix=np.where(flag == 0)[0]
        
            #pixtag=np.empty(0,dtype='|S10')
            #for pixno in range(len(freq)):
                #pixtag = np.append(pixtag,'/r%i/p%i/' %(j,pixno))     
        
            #print len(xpos[self.goodpix]), 'Good Pixels'

            #Create a list of good pixel locations
            #self.xpos=np.append(self.xpos,xpos[self.goodpix])
            #self.ypos=np.append(self.ypos,ypos[self.goodpix])
            #self.freqvals=np.append(self.freqvals,freq[self.goodpix])
            #self.attenvals=np.append(self.attenvals,atten[self.goodpix])
            #self.goodpixtag=np.append(self.goodpixtag,pixtag[self.goodpix])

        # Can fix this later to pick out individual roaches
        self.xvals = self.xpos/self.scale[0]
        self.xvals += self.origin[0][0]
        self.yvals = self.ypos/self.scale[0]
        self.yvals += self.origin[0][1]

        self.xpos0 = self.xpost0/self.scale[0]
        self.xpos0 += self.origin[0][0]
        self.ypos0 = self.ypost0/self.scale[0]
        self.ypos0 += self.origin[0][1]

        self.xpos1 = self.xpost1/self.scale[0]
        self.xpos1 += self.origin[0][0]
        self.ypos1 = self.ypost1/self.scale[0]
        self.ypos1 += self.origin[0][1]

        self.mask=(np.linspace(0,len(self.xvals)-1,len(self.xvals))).astype('int')
        # Plot the locations of the good pixels
        self.ui.pixPlot.canvas.ax.clear()
        self.plot_grid()
        self.ui.pixPlot.canvas.ax.plot(self.xvals[self.mask],self.yvals[self.mask],'b+')
        self.ui.pixPlot.canvas.draw()
        
        
        
        

    def save_process(self):
        print len(self.xvals), 'Total Good Pixels'
        print 'xmin, xmax =', np.min(self.xvals), np.max(self.xvals)
        print 'ymin, ymax =', np.min(self.yvals), np.max(self.yvals)

        xpix=((self.xvals[self.mask])*self.c + (self.yvals[self.mask])*self.s)
        ypix=(-1.*(self.xvals[self.mask])*self.s + (self.yvals[self.mask])*self.c)

        xpix0=self.xpos0*self.c + self.ypos0*self.s
        ypix0=-1.*self.xpos0*self.s + self.ypos0*self.c

        xpix1=self.xpos1*self.c + self.ypos1*self.s
        ypix1=-1.*self.xpos1*self.s + self.ypos1*self.c
        
        self.xposAllTransformed = self.xposAll/self.scale[0]
        self.xposAllTransformed += self.origin[0][0]
        self.yposAllTransformed = self.yposAll/self.scale[0]
        self.yposAllTransformed += self.origin[0][1]
        
        self.xpixAll = ((self.xposAllTransformed*self.c) + (self.yposAllTransformed)*self.s)
        self.ypixAll = (-1.*(self.xposAllTransformed)*self.s + (self.yposAllTransformed)*self.c)
        
        f= open(self.outputFilename,'w')
        #for i in range(len(self.goodpix)):  
        for i in range(self.pixNumber):  
            f= open(self.outputFilename,'a')
            #f.write(str(self.pixelNumber[i]) +'\t' + str(xpix[i]) + '\t' + str(ypix[i]+25) +'\n')
            if int(self.flag[i]) == 0:
                f.write(str(int(self.pixelNumberAll[i])) + '\t' + str(int(self.flag[i])) + '\t' + str(self.xpixAll[i]) + '\t' + str(self.ypixAll[i]) + '\n')
            elif int(self.flag[i]) == 1:
                f.write(str(int(self.pixelNumberAll[i])) + '\t' + str(int(self.flag[i])) + '\t' + str(999) + '\t' + str(999) + '\n')
            elif int(self.flag[i]) == 2:
                f.write(str(int(self.pixelNumberAll[i])) + '\t' + str(int(self.flag[i])) + '\t' + str(self.xpixAll[i]) + '\t' + str(999) + '\n')
            elif int(self.flag[i]) == 3:
                f.write(str(int(self.pixelNumberAll[i])) + '\t' + str(int(self.flag[i])) + '\t' + str(999) + '\t' + str(self.ypixAll[i]) + '\n')
            f.close()

        print 'Number of Doubles:' + str(self.xpost0.size)
        #dbltag=np.loadtxt(self.dblfile,unpack='True', usecols = (4,), dtype='|S10' )
        f= open(self.outputDoubleName,'w')
        for i in range(self.xpos0.size):
            f= open(self.outputDoubleName,'a')
            f.write(str(self.doublePixelNumber[i]) + '\t' +str(xpix0[i]) + '\t' + str(ypix0[i]) + '\t' + str(xpix1[i]) + '\t' + str(ypix1[i]) + '\n')
            f.close()
        

# Start up main gui
if __name__ == "__main__":
	app = QApplication(sys.argv)
	myapp = StartQt4()
	myapp.show()
	app.exec_()

