"""
Author:    Alex Walter
Date:      Jul 3, 2016


This is a GUI class for real time control of the DARKNESS instrument. 
 - show realtime image
 - start/end observations
 - organize how data is saved to disk
 - pull telescope info
 - save header information
 """
 
 
import os, sys, time, struct
from functools import partial
import matplotlib
import matplotlib.cm
#import matplotlib.colors
import numpy as np
import ConfigParser
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
from PyQt4 import QtGui
from PyQt4.QtGui import *

class ImageSearcher(QtCore.QObject):     #Extends QObject for use with QThreads
    imageFound = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal()
    
    def __init__(self, path, nCols, nRows, parent=None):
        super(QtCore.QObject, self).__init__(parent)
        self.path = path
        self.nCols = nCols
        self.nRows= nRows
        self.search=True
        
    def checkDir(self):
        self.search=True
        while self.search:
            flist = []
            for f in os.listdir(self.path):
                if f.endswith(".img"):
                    flist.append(f)
            if len(flist)>0:
                flist.sort()        # if there's more than one, then deal with the oldest one first
                for f in flist:
                    image = self.readBinToList(self.path+f)
                    os.remove(self.path+f)
                    self.imageFound.emit(image)
                    time.sleep(.1) #Give image time to process before sending next one
        self.finished.emit()
    
    def readBinToList(self,fn):
        #fn = '/mnt/ramdisk/1467582711.img'
        fin = open(fn, "rb")
        data = fin.read()
        fmt = 'H'*(len(data)/2)   # 2 bytes per each unsigned 16 bit integer
        fin.close()
        image = np.asarray(struct.unpack(fmt,data), dtype=np.int)
        image=image.reshape((self.nCols,self.nRows))
        image=np.transpose(image)
        return image
        
class ConvertPhotonsToRGB(QtCore.QObject):
    convertedImage = QtCore.pyqtSignal(object)
    
    def __init__(self, image, minCountCutoff=0,maxCountCutoff=2000, logStretch=False,parent=None):
        super(QtCore.QObject, self).__init__(parent)
        self.image=np.asarray(image)
        print image.shape
        self.minCountCutoff=minCountCutoff
        self.maxCountCutoff=maxCountCutoff
        self.logStretch=logStretch
    
    def histEqualization(self):
        print "Running hist"
        imShape=self.image.shape
        
        self.image[np.where(self.image>self.maxCountCutoff)]=self.maxCountCutoff
        maxVal = np.amax(self.image)
        if self.logStretch: self.minCountCutoff=max(self.minCountCutoff, 1)
        self.image[np.where(self.image<self.minCountCutoff)]=self.minCountCutoff
        
        bins=256
        if self.logStretch:
            bins = np.logspace(np.log10(self.minCountCutoff),np.log10(maxVal),256)
        imhist, imbins = np.histogram(self.image.flatten(), bins, density=True)
        
        cdf = (imhist*(imbins[1:]-imbins[:-1])).cumsum()
        cdf*=255
        
        image2 = np.interp(self.image.flatten(),imbins[:-1],cdf)
        image2=image2.reshape(self.image.shape)
        print image2
        print image2.shape
        #image2=image2.astype(np.uint8)
        #print image2
        
        image2RGB=[]
        for i in range(len(image2)):
            row=[]
            for k in range(len(image2[0])):
                pixel_ik = [image2[i,k],image2[i,k],image2[i,k],1.]
                row.append(pixel_ik)
            image2RGB.append(row)
        image2RGB=np.asarray(image2RGB,dtype=np.uint8)
        print image2RGB.shape
        print image2RGB
        
        #self.makeQPixMap(image2RGB)
        self.makeQPixMap(image2)
    
    def makeQPixMap(self,image):
        
        #mappable = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('Greys'))
        #image = mappable.to_rgba(image)
        
        #print image
        #q_im = QtGui.QImage(image,len(image),len(image[0]))
        q_im = QtGui.QImage(image,len(image),len(image[0]),QImage.Format_Indexed8)
        q_im.setColorTable([QtGui.qRgb(i, i, i) for i in range(256)])
        #q_im = QtGui.QImage(image,len(image),len(image[0]),QImage.Format_Mono)
        #q_pixmap = QPixmap(q_im)
        self.convertedImage.emit(q_im)
        
        
        
        

class MkidDashboard(QMainWindow):
    def __init__(self, configPath=None, parent=None):
        self.config = ConfigParser.ConfigParser()
        if configPath is None:
            configPath = 'darkDash.cfg'
        self.config.read(configPath)
        
        
        
        #Setup GUI
        super(QMainWindow, self).__init__(parent)
        self.setWindowTitle(self.config.get('properties','instrument'))
        self.create_image_widget()
        self.obsTime = QtCore.QTime()
        self.create_obs_widget()
        self.create_menu()
        
        #Connect to ROACHES and initialize network port in firmware
        
        # Search for image files from cuber
        self.darkImageSearcher = ImageSearcher(self.config.get('properties','cuber_dir'), self.config.getint('properties','ncols'),self.config.getint('properties','nrows'),parent=None)
        self.thread = QtCore.QThread(parent=self)
        self.thread.setObjectName("DARKimageSearch")
        self.darkImageSearcher.moveToThread(self.thread)
        self.thread.started.connect(self.darkImageSearcher.checkDir)
        self.darkImageSearcher.imageFound.connect(self.convertImage)
        self.darkImageSearcher.finished.connect(self.thread.quit)
        self.darkImageSearcher.finished.connect(self.test)
        
        self.obsTime.start()
        #QtCore.QTimer.singleShot(2000,self.updateImage)
        
        self.convertThreads=[]
        self.converters=[]
    
    def test(self):
        print "Finished darkimage search"
    
    def convertImage(self,image):
        print "Found Image! Now converting..."
        
        converter=ConvertPhotonsToRGB(image)
        self.converters.append(converter)                       #Need local reference or else signal is lost!
        thread = QtCore.QThread(parent=self)
        thread_num=len(self.convertThreads)
        thread.setObjectName("convertImage_"+str(thread_num))
        self.convertThreads.append(thread)                      #Need to have local reference to thread or else it will get lost!
        converter.moveToThread(thread)
        thread.started.connect(converter.histEqualization)
        converter.convertedImage.connect(thread.quit)
        converter.convertedImage.connect(self.updateImage)
        thread.start()
        
        #print image.shape
    
    def updateImage(self, image):
        print 'Updating'
        #image=QImage("output1.png")
        #image=QImage(convertedImage,QImage.Format_Indexed8
        imageScale=self.config.getint('properties','image_scale')
        self.image=image.scaledToWidth(image.width()*imageScale)
        self.grPixMap.pixmap().convertFromImage(self.image)
        
        borderSize=24
        imgSize = self.grPixMap.pixmap().size()
        frameSize = QtCore.QSize(imgSize.width()+borderSize,imgSize.height()+borderSize)
        self.centralWidget().resize(frameSize)
        self.resize(self.childrenRect().size())
        
        self.grPixMap.update()
        #QtCore.QTimer.singleShot(1000,self.updateImage)

    
    def pixelClicked(self, event):
        position = QtCore.QPoint( event.pos().x(),  event.pos().y())
        color = QColor.fromRgb(self.image.pixel( position ) )
        x_pos = int(np.floor(event.pos().x()/self.config.getint('properties','image_scale')))
        y_pos = int(np.floor(event.pos().y()/self.config.getint('properties','image_scale')))
        if color.isValid():
            rgbColor = '('+str(color.red())+','+str(color.green())+','+str(color.blue())+','+str(color.alpha())+')'
            print 'Pixel position = (' + str( x_pos ) + ' , ' + str( y_pos )+ ') - Value (R,G,B,A)= ' + rgbColor
        else:
            print 'Pixel position = (' + str( event.pos().x() ) + ' , ' + str( event.pos().y() )+ ') - color not valid'
    
    def startObs(self):
        print "Starting Obs"
        #Need switch Firmware into photon collect mode
        #wait 1 ms, then write START file to RAM disk
        self.obsTime.restart()
        self.obsTime_updater.start()
        self.thread.start()
    
    def stopObs(self):
        print "Stop Obs"
        #Need to switch Firmware out of photon collect mode
        #wait 1 ms
        self.obsTime_updater.stop()
        self.darkImageSearcher.search=False
    
    def create_obs_widget(self):
        obs_dock_widget = QDockWidget(self)
        obs_dock_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)
        obs_widget=QWidget(obs_dock_widget)
        
        button_obs = QPushButton("Start Observing")
        font = button_obs.font()
        font.setPointSize(24)
        button_obs.setFont(font)
        button_obs.setEnabled(True)
        button_obs.clicked.connect(self.startObs)
        
        button_stop = QPushButton("Stop Observing")
        button_stop.setEnabled(True)
        button_stop.clicked.connect(self.stopObs)
        
        self.label_obsTime = QLabel("0.0 seconds")
        self.obsTime_updater = QtCore.QTimer(self)
        self.obsTime_updater.setInterval(1000)
        #self.obsTime_updater.timeout.connect(partial(self.label_obsTime.setText,str(self.obsTime.elapsed()/1000.)+' seconds'))
        self.obsTime_updater.timeout.connect(self.updateObsTime)
        
        
        
        vbox = QVBoxLayout()
        vbox.addWidget(button_obs)
        vbox.addWidget(button_stop)
        vbox.addWidget(self.label_obsTime)
        vbox.addStretch()
        obs_widget.setLayout(vbox)
        obs_dock_widget.setWidget(obs_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea,obs_dock_widget)
        
    def updateObsTime(self):
        #print 'hi'
        #print self.obsTime.elapsed()
        self.label_obsTime.setText(time.strftime("%H:%M:%S",time.gmtime(self.obsTime.elapsed()/1000.)))
        #self.label_obsTime.setText(str(self.obsTime.elapsed()/1000.)+' seconds')
    
    def create_image_widget(self):
        self.imageFrame = QtGui.QFrame(parent=self)
        self.imageFrame.setFrameShape(QtGui.QFrame.Box)
        self.imageFrame.setFrameShadow(QtGui.QFrame.Sunken)
        self.imageFrame.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        
        self.image=QImage(1,1,QImage.Format_Mono)
        self.image.fill(Qt.black)
        grview = QGraphicsView(self.imageFrame)
        scene = QGraphicsScene(parent=grview)
        self.grPixMap = QGraphicsPixmapItem(QPixmap(self.image), None, scene)
        self.grPixMap.mousePressEvent = self.pixelClicked
        grview.setScene(scene)
        grview.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        grview.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        layout=QHBoxLayout()
        layout.addWidget(grview)
        self.imageFrame.setLayout(layout)
        
        self.setCentralWidget(self.imageFrame)

    def create_menu(self):        
        self.file_menu = self.menuBar().addMenu("&File")
        quit_action = self.create_action("&Quit", slot=self.close,shortcut="Ctrl+Q", tip="Close the application")
        self.add_actions(self.file_menu, (None, quit_action))

        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.create_action("&About", shortcut='F1',slot=self.on_about, tip='About the demo')
        self.add_actions(self.help_menu, (about_action,))
    
    def on_about(self):
        msg = "MKID Dashboard\n"\
              "Click on pixels for timestream plot\n\n"\
              "Author: Alex Walter\n" \
              "Date: Jul 3, 2016"
        QMessageBox.about(self, "MKID-ROACH2 software", msg.strip())
    
    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def create_action(  self, text, slot=None, shortcut=None, 
                        icon=None, tip=None, checkable=False, 
                        signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, QtCore.SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action

        
    def closeEvent(self, event):
        self.stopObs()      # Make sure we send packetmaster the quit signal!
        del self.grPixMap   #Get segfault if we don't delete this. Something about signals in the queue trying to access deleted objects...
        QtCore.QCoreApplication.instance().quit()

def main():
    app = QApplication(sys.argv)
    form = MkidDashboard()
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
        
        
