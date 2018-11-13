"""
Author:    Alex Walter
Date:      Jul 3, 2016


This is a GUI class for real time control of the MEC and DARKNESS instruments. 
 - show realtime image
 - show realtime pixel timestreams (see guiwindows.PixelTimestreamWindow)
 - start/end observations
 - organize how data is saved to disk
 - pull telescope info
 - save header information
 
 CLASSES:
    MKIDDashboard - main GUI
    ImageSearcher - searches for new images on ramdisk
    ConvertPhotonsToRGB - converts a 2D list of photon counts to a QImage
 """
from __future__ import print_function
import argparse
import sys, traceback, os
import binascii
from functools import partial
import subprocess
import numpy as np
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
from PyQt4.QtGui import *

from astropy.io import fits

from mkidcore.fits import CalFactory, summarize, loadimg, combineHDU
import mkidcore.config
import time
from mkidcore.corelog import getLogger, setup_logging
from mkidreadout.readout.guiwindows import PixelTimestreamWindow, PixelHistogramWindow
from mkidreadout.readout.lasercontrol import LaserControl
from mkidreadout.readout.Telescope import *
from mkidreadout.channelizer.Roach2Controls import Roach2Controls
from mkidreadout.utils.utils import interpolateImage
from mkidreadout.configuration.beammap.beammap import Beammap
from mkidreadout.readout.packetmaster import Packetmaster
import mkidreadout.hardware.conex
import mkidreadout.hardware.hsfw
import threading


def add_actions(target, actions):
    for action in actions:
        if action is None:
            target.addSeparator()
        else:
            target.addAction(action)


class ImageSearcher(QtCore.QObject):  # Extends QObject for use with QThreads
    """
    This class looks for binary '.img' files spit out by the PacketMaster c program on the ramdisk
    
    When it finds an image, it grabs the data, parses it into an array, and emits an imageFound signal
    Optionally, it deletes the data on the ramdisk so it doesn't fill up
    
    SIGNALS
        imageFound - emits when an image is found
        finished - emits when self.search is set to False
    """
    imageFound = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal()

    def __init__(self, path, nCols, nRows, parent=None):
        """
        INPUTS:
            path - path to the ramdisk where PacketMaster places the image files
            nCols - Number of columns in image (for DARKNESS its 80)
            nRows - for DARKNESS its 125
            parent - Leave as None so that we can add to new thread
        """
        super(QtCore.QObject, self).__init__(parent)
        self.path = path
        self.nCols = nCols
        self.nRows = nRows
        self.search = True

    def checkDir(self, removeOldFiles=False):
        """
        Infinite loop that keeps checking directory for an image file
        When it finds an image, it parses it and emits imageFound signal
        It only returns files that have timestamps > than the last file's timestamp
        
        Loop will continue until you set self.search to False. Then it will emit finished signal
        
        INPUTS:
            removeOldFiles - remove .img and .png files after we read them
        """
        self.search = True
        latestTime = time.time() - .5
        while self.search:
            flist = []
            for f in os.listdir(self.path):
                if f.endswith(".img"):
                    if float(f.split('.')[0]) > latestTime:
                        flist.append(f)
                    elif removeOldFiles:
                        os.remove(os.path.join(self.path,f))
                elif removeOldFiles and f.endswith(".png"):
                    os.remove(os.path.join(self.path,f))

            flist.sort()
            for f in flist:
                latestTime = float(f.split('.')[0]) + .1
                file = os.path.join(self.path, f)
                try:
                    image = loadimg(file, self.nCols, self.nRows, returntype='hdu')
                    self.imageFound.emit(image)
                    # time.sleep(.01)  # Give image time to process before sending next one (not really needed)
                except Exception:
                    getLogger('Dashboard').error('Problem on file %s', os.path.join(self.path,f),
                                                 exc_info=True)
                if removeOldFiles:
                    os.remove(file)
        self.finished.emit()


class ConvertPhotonsToRGB(QtCore.QObject):
    """
    This class takes 2D arrays of photon counts and converts them into a QImage
    It needs to know how to map photon counts into an RGB color
    Usually just an 8bit grey color [0, 256) but also turns maxxed out pixel red
    
    SIGNALS
        convertedImage - emits when it's done converting an image
    """
    convertedImage = QtCore.pyqtSignal(object)

    def __init__(self, image, minCountCutoff=0, maxCountCutoff=450, stretchMode='log', interpolate=False, makeRed=True,
                 parent=None):
        """
        INPUTS:
            image - 2D numpy array of photon counts with np.nan where beammap failed
            minCountCutoff - anything <= this number of counts will be black
            maxCountCutoff - anything >= this number of counts will be red
            stretchMode - can be log, linear, hist
            interpolate - interpolate over np.nan pixels
        """
        super(QtCore.QObject, self).__init__(parent)
        self.image = np.copy(image)
        self.minCountCutoff = minCountCutoff
        self.maxCountCutoff = maxCountCutoff
        # self.logStretch=logStretch
        self.interpolate = interpolate
        self.makeRed = makeRed
        self.stretchMode = stretchMode

        # print '#red: ',len(self.redPixels[0])

    def stretchImage(self):
        """
        map photons to greyscale
        """
        # first interpolate and find hot pixels
        if self.interpolate:
            self.image = interpolateImage(self.image)
        self.image[np.where(np.logical_not(np.isfinite(self.image)))] = 0  # get rid of np.nan's
        if self.makeRed:
            self.redPixels = np.where(self.image >= self.maxCountCutoff)
        else:
            self.redPixels = []

        if self.stretchMode == 'logarithmic':
            imageGrey = self.logStretch()
        elif self.stretchMode == 'linear':
            imageGrey = self.linStretch()
        elif self.stretchMode == 'histogram equalization':
            imageGrey = self.histEqualization()
        else:
            raise ValueError

        self.makeQPixMap(imageGrey)

    def logStretch(self):
        """
        map photon counts to greyscale logarithmically
        """
        self.image[np.where(self.image > self.maxCountCutoff)] = self.maxCountCutoff
        self.image[np.where(self.image < self.minCountCutoff)] = self.minCountCutoff
        maxVal = np.amax(self.image)
        minVal = np.amin(self.image)
        maxVal = np.amax([minVal + 1, maxVal])

        image2 = 255. / (np.log10(1 + maxVal - minVal)) * np.log10(1 + self.image - minVal)
        return image2

    def linStretch(self):
        """
        map photon count to greyscale linearly (max 255 min 0)
        """
        self.image[np.where(self.image > self.maxCountCutoff)] = self.maxCountCutoff
        self.image[np.where(self.image < self.minCountCutoff)] = self.minCountCutoff

        # maxVal = np.amax(self.image)
        # minVal = np.amin(self.image)
        maxVal = self.maxCountCutoff
        minVal = self.minCountCutoff
        maxVal = np.amax([minVal + 1, maxVal])

        image2 = (self.image - minVal) / (1.0 * maxVal - minVal) * 255.
        return image2

    def histEqualization(self):
        """
        perform a histogram Equalization. This tends to make the contrast better
        
        if self.logStretch is True the histogram uses logarithmic spaced bins
        """
        imShape = self.image.shape

        self.image[np.where(self.image > self.maxCountCutoff)] = self.maxCountCutoff
        maxVal = np.amax(self.image)
        if self.logStretch: self.minCountCutoff = max(self.minCountCutoff, 1)
        self.image[np.where(self.image < self.minCountCutoff)] = self.minCountCutoff

        bins = 256
        if self.logStretch:
            bins = np.logspace(np.log10(self.minCountCutoff), np.log10(maxVal), 256)
        imhist, imbins = np.histogram(self.image.flatten(), bins, density=True)

        cdf = (imhist * (imbins[1:] - imbins[:-1])).cumsum()
        cdf *= 255

        image2 = np.interp(self.image.flatten(), imbins[:-1], cdf)
        image2 = image2.reshape(self.image.shape)
        image2[np.where(self.image <= self.minCountCutoff)] = 0

        return image2

    def makeQPixMap(self, image):
        """
        This function makes the QImage object
        
        INPUTS:
            image - 2D numpy array of [0,256) grey colors
        """
        image2 = image.astype(np.uint32)

        redMask = np.copy(image2)
        redMask[self.redPixels] = np.uint32(0)
        #           24-32 -> A  16-24 -> R     8-16 -> G      0-8 -> B
        imageRGB = (255 << 24 | image2 << 16 | redMask << 8 | redMask).flatten()  # pack into RGBA
        q_im = QtCore.QtGui.QImage(imageRGB, self.image.shape[1], self.image.shape[0], QImage.Format_RGB32)

        self.convertedImage.emit(q_im)


class MKIDDashboard(QMainWindow):
    """
    Dashboard for seeing realtime images
    
    SIGNALS:
        newImageProcessed() - emited after processing and plotting a new image. Also whenever the current pixel selection changes
    """
    newImageProcessed = QtCore.pyqtSignal()

    def __init__(self, roachNums, config='./dashboard.yml', observing=False, parent=None, offline=False):
        """
        INPUTS:
            roachNums - List of roach numbers to connect with
            config - the configuration file. See ConfigParser doc for making configuration file
            observing - indicates if packetmaster is currently writing data to disk
            parent -
        """
        super(QMainWindow, self).__init__(parent)
        self.config = mkidcore.config.load(config)
        self.offline = offline

        # important variables
        self.threadPool = []  # Holds all the threads so they don't get lost. Also, they're garbage collected if they're attributes of self
        self.workers = []  # Holds workder objects corresponding to threads
        self.imageList = []  # Holds photon count image data
        self.fitsList = []
        self.timeStreamWindows = []  # Holds PixelTimestreamWindow objects
        self.histogramWindows = []  # Holds PixelHistogramWindow objects
        self.selectedPixels = set()  # Holds the pixels currently selected
        self.observing = observing  # Indicates if packetmaster is currently writing data to disk
        self.darkField = None  # Holds a dark image for subtracting
        self.flatField = None  # Holds a flat image for normalizing

        self.roachList = []
        self.beammap = None
        self.beammapFailed = None
        self.flatFactory = None
        self.darkFactory = None
        self.sciFactory = None

        # Often overwritten variables
        self.clicking = False  # Flag to indicate we've pressed the left mouse button and haven't released it yet
        self.pixelClicked = None  # Holds the pixel clicked when mouse is pressed
        self.pixelCurrent = None  # Holds the pixel the mouse is currently hovering on
        self.takingDark = 0  # Flag for taking dark image. Indicates number of images we still need for darkField image.
        self.takingFlat = 0  # Flag for taking flat image. Indicates number of images we still need for flatField image

        # Initialize PacketMaster8
        getLogger('Dashboard').info('Initializing packetmaster...')
        self.packetmaster = Packetmaster(self.config.roaches, ramdisk=self.config.packetmaster.ramdisk,
                                         detinfo=(self.config.detector.ncols, self.config.detector.nrows),
                                         nuller=self.config.packetmaster.nuller,
                                         captureport=self.config.packetmaster.captureport,
                                         start=self.config.spawn_packetmaster and not self.offline)

        if not self.packetmaster.is_running:
            getLogger('Dashboard').info('Packetmaster not started. Start manually...')

        # Laser Controller
        getLogger('Dashboard').info('Setting up laser control...')
        self.laserController = LaserControl(self.config.lasercontrol.ip, self.config.lasercontrol.port,
                                            self.config.lasercontrol.receive_port)

        # telscope TCS connection
        getLogger('Dashboard').info('Setting up telescope connection...')
        self.telescopeController = Telescope(self.config.telescope.ip, self.config.telescope.port,
                                             self.config.telescope.receive_port)
        self.telescopeWindow = TelescopeWindow(self.telescopeController)

        # Setup GUI
        getLogger('Dashboard').info('Setting up GUI...')
        self.setWindowTitle(self.config.instrument + ' Dashboard')
        self.create_image_widget()
        self.create_dock_widget()
        self.contextMenu = QMenu(self)  # pops up on right click
        self.create_menu()  # file menu

        # Connect to ROACHES and initialize network port in firmware
        getLogger('Dashboard').info('Connecting roaches and loading beammap...')
        if not self.offline:
            for roachNum in roachNums:
                roach = Roach2Controls(self.config.roaches.get('r{}.ip'.format(roachNum)),
                                       self.config.roaches.fpgaparamfile, num=roachNum,
                                       verbose=False, debug=False)
                if not roach.connect() and not roach.issetup:
                    raise RuntimeError('Roach r{} has not been setup.'.format(roachNum))
                roach.loadCurTimestamp()
                roach.setPhotonCapturePort(self.packetmaster.captureport)
                self.roachList.append(roach)
            self.turnOnPhotonCapture()
            self.loadBeammap()

        # Setup search for image files from cuber
        getLogger('Dashboard').info('Setting up image searcher...')
        darkImageSearcher = ImageSearcher(self.config.packetmaster.ramdisk,
                                          self.config.detector.ncols,
                                          self.config.detector.nrows, parent=None)
        self.workers.append(darkImageSearcher)
        thread = QtCore.QThread(parent=self)
        self.threadPool.append(thread)
        thread.setObjectName("DARKimageSearch")
        darkImageSearcher.moveToThread(thread)
        thread.started.connect(darkImageSearcher.checkDir)
        darkImageSearcher.imageFound.connect(self.convertImage)
        darkImageSearcher.finished.connect(thread.quit)
        if not self.offline:
            QtCore.QTimer.singleShot(10, thread.start)  # start the thread after a second

    def turnOffPhotonCapture(self):
        """
        Tells roaches to stop photon capture
        """
        for roach in self.roachList:
            roach.stopSendingPhotons()
        getLogger('Dashboard').info('Roaches stopped sending photon packets')

    def turnOnPhotonCapture(self):
        """
        Tells roaches to start photon capture
        
        Have to be careful to set the registers in the correct order in case we are currently in phase capture mode
        """
        for roach in self.roachList:
            roach.startSendingPhotons(self.config.packetmaster.ip, self.config.packetmaster.captureport)
        getLogger('Dashboard').info('Roaches sending photon packets!')

    def loadBeammap(self):
        """
        This function loads the beammap into the roach firmware
        It uses the beammapFile property in the config file
        If it can't find the beammap file then it loads in the defualt beammap from default.txt
        
        We set self.beammapFailed here for later use. It's a 2D boolean array with the (row,col)=(y,x) 
        indicating if that pixel is in the beammap or not
        """
        try:
            self.beammap = Beammap(self.config.beammap)
        except IOError:
            getLogger('Dashboard').warning("Could not find beammap %s. Using default", self.config.beammap)
            self.beammap = Beammap(default=self.config.instrument)

        for roach in self.roachList:
            roach.loadBeammapCoords(self.beammap)

        self.beammapFailed = self.beammap.failmask
        getLogger('Dashboard').info('Loaded beammap: %s', self.beammap)

    def startDithering(self):
        self.button_dither.setEnabled(False)
        self.button_dither.clicked.disconnect()
        self.button_dither.setText('Stop Dithering')
        self.button_dither.clicked.connect(self.stopDithering)

        class DitherThread(QtCore.QThread):
            def __init__(self, url, pattern, timeout=np.inf, parent=None):
                super(QtCore.QThread, self).__init__(parent)
                self.url = url
                self.pattern = pattern
                self.status = None
                self.timeout = timeout

            def run(self):
                self.status = mkidreadout.hardware.conex.dither(id=self.pattern, address=self.url)
                if not self.status.running:
                    return
                # TODO self.status.state, pos, conexstatus need to get synced and logged with logstate
                while True:
                    time.sleep(.25)
                    self.timeout -= .25
                    self.status = mkidreadout.hardware.conex.status(self.url)

                    if not self.status.running:
                        break
                    elif self.timeout <= 0:
                        self.status = mkidreadout.hardware.conex.stop(self.url)
                        break

        dither = DitherThread(self.config.dither.url, self.config.dither.pattern, parent=self)

        def finish():
            if dither.status.offline or dither.status.haserrors:
                msg = 'Dither completed with errors: "{}", Conex Status="{}"'.format(
                    dither.status.state, dither.status.conexstatus)
                getLogger('Dashboard').error(msg)
            else:
                dither_result = 'Dither Path: {}\n'.format(str(dither.status.last_dither).replace('\n', '\n   '))
                getLogger('Dashboard').info(dither_result)
            self.button_dither.clicked.disconnect()
            self.button_dither.clicked.connect(self.startDithering)
            self.button_dither.setText('Start Dithering')

        dither.finished.connect(finish)
        dither.start()
        self.button_dither.setEnabled(True)

    def stopDithering(self):
        r = mkidreadout.hardware.conex.stop(address=self.config.dither.url)
        if not r.haserrors:
            getLogger('Dashboard').info('Dither halted by user.')
            self.button_dither.setText('Start Dithering')
            self.button_dither.clicked.disconnect()
            self.button_dither.clicked.connect(self.startDithering)
        else:
            getLogger('Dashboard').error('Stop dither error: {}'.format(r))
        self.button_dither.setEnabled(True)

    def addDarkImage(self, photonImage):
        self.spinbox_darkImage.setEnabled(False)
        if self.darkField is None or self.takingDark == self.spinbox_darkImage.value():
            self.darkFactory = CalFactory('dark', images=(photonImage,))
        else:
            self.darkFactory.add_image(photonImage)

        self.takingDark -= 1
        if self.takingDark == 0:
            self.takingDark = -1

            name = '{}_dark_{}'.format(self.config.dashboard.darkname, time.time())
            self.darkField = self.darkFactory.generate(fname=name, name=name, badmask=self.beammapFailed)
            self.darkField.writeto(os.path.join(self.cofig.paths.datadir, self.darkField.header.filename))

            getLogger('ObsLog').info('Finished dark {}:\n {}'.format(self.darkField.filename,
                                                                     summarize(self.darkField).replace('\n',
                                                                                                          '\n  ')))
            self.checkbox_darkImage.setChecked(True)
            self.spinbox_darkImage.setEnabled(True)

    def addFlatImage(self, photonImage, minFlat=1, maxFlat=2500):
        self.spinbox_flatImage.setEnabled(False)

        if self.flatField is None or self.takingFlat == self.spinbox_flatImage.value():
            self.flatFactory = CalFactory('flat', images=(photonImage,), min=minFlat, max=maxFlat,
                                          dark=self.darkField, colslice=slice(20, 60))
        else:
            self.flatFactory.add_image(photonImage)

        self.takingFlat -= 1
        if self.takingFlat == 0:
            self.takingFlat = -1
            name = '{}_flat_{}'.format(self.config.dashboard.flatname, time.time())
            self.flatField = self.flatFactory.generate(fname=name, name=name, badmask=self.beammapFailed)
            self.flatField.writeto(os.path.join(self.cofig.paths.datadir, self.flatField.header.filename))
            getLogger('ObsLog').info('Finished flat {}:\n {}'.format(self.flatField.header.filename,
                                                                     summarize(self.darkField).replace('\n',
                                                                                                          '\n  ')))
            self.checkbox_flatImage.setChecked(True)
            self.spinbox_flatImage.setEnabled(True)

    def convertImage(self, photonImage=None):
        """
        This function is automatically called when ImageSearcher object finds a new image file from cuber program
        We also call this function if we change the image processing controls like
            min/max count rate
            num_images_to_add
        
        Here we set up a converter object to parse the photon counts into an RBG QImage object
        We do this in a separate thread because it might take a while to process
        
        INPUTS:
            photonImage - 2D numpy array of photon counts (type np.uint16 if from ImageSearcher object)
        """
        # If there's new data, append it
        if photonImage is not None:
            photonImage.header['exptime'] = self.config.packetmaster.int_time
            self.imageList.append(photonImage)
            self.fitsList.append(photonImage)
            self.imageList = self.imageList[-self.config.dashboard.average:]  #trust the garbage collector

            if self.takingDark > 0:
                self.addDarkImage(photonImage)
            elif self.takingFlat > 0:
                self.addFlatImage(photonImage)
            elif self.observing:
                self.sciFactory.add_image(photonImage)
        elif not self.imageList:
            return

        # If we've got a full set of data package it as a fits file
        tstart = self.fitsList[0].header.utc if self.fitsList else time.time()
        tstamp = time.time()
        if ((sum([i.header.exptime for i in self.fitsList]) >= self.config.dashboard.fitstime) or
            (time.time() - tstart) >= self.config.dashboard.fitstime):
            combineHDU(self.fitsList, fname='stream{}.fits.gz'.format(tstamp),
                       name=str(tstamp), save=True, threaded=True)
            self.fitsList = []

        # Get the (average) photon count image
        bias = 0
        if self.checkbox_flatImage.isChecked() and self.flatField is None:
            try:
                self.flatField = fits.open(self.config.dashboard.flatfile)
            except IOError:
                getLogger('Dashboard').warning('Unable to load flat from {}'.format(self.config.dashboard.flatfile))

        #TODO this really could be moved into the ConvertPhotonsToRGB thread
        cf = CalFactory('avg', images=self.imageList[-self.config.dashboard.average:],
                        dark=self.darkField if self.checkbox_darkImage.isChecked() else None,
                        flat=self.flatField if self.checkbox_flatImage.isChecked() else None)
        image = cf.generate(bias=bias if self.checkbox_darkImage.isChecked() else 0)
        image.data[self.beammapFailed] = np.nan

        # Set up worker object and thread
        converter = ConvertPhotonsToRGB(image.data,
                                        self.config.dashboard.min_count_rate,
                                        self.config.dashboard.max_count_rate,
                                        self.combobox_stretch.currentText(),
                                        self.checkbox_interpolate.isChecked(),
                                        not self.checkbox_smooth.isChecked()) # if we're smoothing don't make pixels red
        self.workers.append(converter)  # Need local reference or else signal is lost!

        thread = QtCore.QThread(parent=self)
        thread_num = len(self.threadPool)
        thread.setObjectName("convertImage_" + str(thread_num))
        self.threadPool.append(thread)  # Need to have local reference to thread or else it will get lost!
        converter.moveToThread(thread)
        thread.started.connect(converter.stretchImage)
        converter.convertedImage.connect(lambda x: thread.quit())
        converter.convertedImage.connect(self.updateImage)
        thread.finished.connect(partial(self.threadPool.remove, thread))  # delete these when done so we don't
        #  have a memory leak
        thread.finished.connect(partial(self.workers.remove, converter))

        # When it's done converting the worker will emit a convertedImage Signal
        converter.convertedImage.connect(lambda x: self.label_numIntegrated.setText(
            str(self.config.dashboard.average) + '/' + self.label_numIntegrated.text().split('/')[-1]))
        thread.start()

    def updateImage(self, q_image):
        """
        This function is automatically called by a ConvertPhotonsToRGB object signal connection
        We don't call this function directly
        
        Add a new QPixMap to the graphicsViewItem and repaint
        It tells all the labels and timestream windows to update
        
        INPUTS:
            q_image - QImage that will be displayed on GUI
        """
        # Scale the bitmap so we can see individual pixels and add to the GraphicsViewItem
        imageScale = self.config.getint('properties', 'image_scale')
        q_image = q_image.scaledToWidth(q_image.width() * imageScale)
        self.grPixMap.pixmap().convertFromImage(q_image)

        # Possibly smooth image
        if self.checkbox_smooth.isChecked():
            self.grPixMap.graphicsEffect().setEnabled(True)
        else:
            self.grPixMap.graphicsEffect().setEnabled(False)

        # Dither image
        # if self.checkbox_dither.isChecked(): print 'dithering'

        # Resize the GUI to fit whole image
        borderSize = 0  # 24   # Not sure how to get the size of the frame's border so hardcoded this for now
        imgSize = self.grPixMap.pixmap().size()
        frameSize = QtCore.QSize(imgSize.width() + borderSize, imgSize.height() + borderSize)

        # self.centralWidget().resize(frameSize) #this automatically resizes window but causes array to move

        # Show image on screen!
        self.grPixMap.update()
        # Update the labels
        self.updateSelectedPixelLabels()
        self.updateCurrentPixelLabel()
        # Update any pixelTimestream windows that are listening
        self.newImageProcessed.emit()

    def mousePressed(self, event):
        """
        This function handles all mouse pressed events on the graphics scene through the QGraphicsPixMapItem
        
        It sets the following semi-important flags and variables:
            self.selectedPixels --> empty set       Unless we have the SHIFT key pressed, reset the list of selected pixels
            self.clicking --> True                  So we know we are in the midst of clicking and haven't released the mouse button
            self.pixelClicked --> clicked pixel     So we know which pixel we first clicked
            self.pixelCurrent --> clicked pixel     This is always the current pixel the mouse is hovering over. 
                                                    Also used for making the self.movingBox after the mouse moves
            self.movingBox --> QGraphicsRectItem    Temporary rectangle. We delete it and draw a new one everytime the mouse moves
            
        INPUTS:
            event - QGraphicsSceneMouseEvent
        """
        if event.button() == QtCore.LeftButton:
            self.clicking = True
            x_pos = int(np.floor(event.pos().x() / self.config.getint('properties', 'image_scale')))
            y_pos = int(np.floor(event.pos().y() / self.config.getint('properties', 'image_scale')))
            getLogger('Dashboard').info('Clicked (' + str(x_pos) + ' , ' + str(y_pos) + ')')
            if QtCore.QtGui.QApplication.keyboardModifiers() != QtCore.ShiftModifier:
                self.selectedPixels = set()
                self.removeAllPixelBoxLines()
            self.pixelClicked = [x_pos, y_pos]
            self.pixelCurrent = self.pixelClicked
            self.movingBox = self.drawPixelBox(self.pixelClicked)

    def mouseMoved(self, event):
        """
        All this function does is draw the little box when you're dragging the mouse
        
        This function is called everytime the mouse hovers or moves on the scene
        We use the self.clicking flag to make a box during a left click event
        
        It sets the following semi-important flags and variables:
            self.pixelCurrent --> current hover pixel         Used to make the self.movingBox
            self.movingBox --> QGraphicsRectItem        Box from the first clicked pixel (self.pixelClicked) to the current pixel (self.pixelCurrent
        
        INPUTS:
            event - QGraphicsSceneMouseEvent
        """
        x_pos = int(np.floor(event.pos().x() / self.config.getint('properties', 'image_scale')))
        y_pos = int(np.floor(event.pos().y() / self.config.getint('properties', 'image_scale')))
        if [x_pos, y_pos] != self.pixelCurrent and x_pos >= 0 and y_pos >= 0 and \
                x_pos < self.config.getint('properties', 'nCols') and \
                y_pos < self.config.getint('properties', 'nRows'):

            self.pixelCurrent = [x_pos, y_pos]
            self.updateCurrentPixelLabel()
            if self.clicking:
                try:
                    self.grPixMap.scene().removeItem(self.movingBox)
                except:
                    pass
                self.movingBox = self.drawPixelBox(pixel1=self.pixelClicked, pixel2=[x_pos, y_pos])

    def mouseReleased(self, event):
        """
        Executes when we release the mouse button. Adds the selected pixels to self.selectedPixels
        Tells everything that needs the current selected pixels to update
        
        It sets the following semi-important flags and variables:
            self.clicking --> False                     See mousePressed()
            self.movingBox --> None                     Removes the temporary box
            self.selectedPixels --> list of pixels      Used to draw selected pixel boxes
        
        INPUTS:
            event - QGraphicsSceneMouseEvent
        """
        if event.button() == QtCore.LeftButton and self.clicking:
            self.clicking = False
            x_pos = int(np.floor(event.pos().x() / self.config.getint('properties', 'image_scale')))
            y_pos = int(np.floor(event.pos().y() / self.config.getint('properties', 'image_scale')))
            getLogger('Dashboard').info('Released (' + str(x_pos) + ' , ' + str(y_pos) + ')')

            x_start, x_end = sorted([x_pos, self.pixelClicked[0]])
            y_start, y_end = sorted([y_pos, self.pixelClicked[1]])
            x_start = max(x_start, 0)  # make sure we're still inside the image
            y_start = max(y_start, 0)
            x_end = min(x_end, self.config.getint('properties', 'nCols') - 1)
            y_end = min(y_end, self.config.getint('properties', 'nRows') - 1)

            newPixels = set((x, y) for x in range(x_start, x_end + 1) for y in range(y_start, y_end + 1))
            self.selectedPixels = self.selectedPixels | newPixels  # Union of set so there are no repeats
            try:
                self.grPixMap.scene().removeItem(self.movingBox)
                self.movingBox = None
            except:
                pass
            self.updateSelectedPixelLabels()
            # Update any pixelTimestream windows that are listening
            self.newImageProcessed.emit()

            for pixel in newPixels:
                self.drawPixelBox(pixel, color='cyan', lineWidth=1)

    def drawPixelBox(self, pixel1, pixel2=None, color='blue', lineWidth=3):
        """
        This function draws a box around pixel1
        If pixel2 is given then it draws a box with pixel1 and pixel 2 at opposite corners
        
        INPUTS:
            pixel1 - an [x, y] coordinate pair
            pixel2 - an [x, y] coordinate pair or None
            color - color of the box
            lineWidth - linewidth of box border in pixels
        """
        # Get upper left and lower right coordinate on graphics scene
        if pixel2 is None:
            pixel2 = pixel1
        scale = self.config.dashboard.image_scale
        x_start, x_end = sorted([pixel1[0], pixel2[0]])
        y_start, y_end = sorted([pixel1[1], pixel2[1]])
        x_start = x_start * scale - 1  # start box one pixel over
        y_start = y_start * scale - 1
        x_end = x_end * scale + scale
        y_end = y_end * scale + scale
        x_start = max(x_start, 0)  # make sure we're still inside the image
        y_start = max(y_start, 0)
        x_end = min(x_end, self.config.detector.ncols * scale)
        y_end = min(y_end, self.config.detector.nrows * scale)
        width = x_end - x_start
        height = y_end - y_start

        # set up the QPen for drawing the box
        q_color = QColor()
        q_color.setNamedColor(color)
        q_pen = QPen(q_color)
        q_pen.setWidth(lineWidth)
        # Draw!
        pixelBox = self.grPixMap.scene().addRect(x_start, y_start, width, height, q_pen)
        return pixelBox

    def removeAllPixelBoxLines(self):
        """
        This function removes all QGraphicsItems (except the QGraphicsPixMapItem which holds our image) from the scene
        """
        for item in self.grPixMap.scene().items():
            if type(item) != QGraphicsPixmapItem:
                self.grPixMap.scene().removeItem(item)

    def getPixCountRate(self, pixelList, numImages2Sum=0, applyDark=False):
        """
        Get the count rate of a list of pixels.
        Can be slow :( 
        Might want to move this to a new thread in future...
        
        INPUTS:
            pixelList - a list or numpy array of pixels (not a set)
            numImages2Sum - average over this many of the last few images. If None, use the number specified on the GUI int Time box
        """
        if not len(pixelList):
            return 0
        if numImages2Sum < 1:
            numImages2Sum = self.config.dashboard.average

        cf=CalFactory('sum', images=self.imageList[-numImages2Sum:], dark=self.darkField if applyDark else None)
        im = cf.generate()
        pixelList = np.asarray(pixelList)
        return np.sum(np.asarray(im.data)[[pixelList[:, 1], pixelList[:, 0]]])

    def updateSelectedPixelLabels(self):
        """
        This updates the labels showing the selected pixel total count rate. Can be slow ...
        
        This function is called whenever the mouse releases and when new image data is processed
        """
        if len(self.selectedPixels) > 0:
            pixels = np.asarray([[p[0], p[1]] for p in self.selectedPixels])
            val = self.getPixCountRate(pixels)
            labelStr = 'Selected Pix : ' + str(np.round(val, 2)) + ' #/s'
            if self.checkbox_darkImage.isChecked():
                valDark = self.getPixCountRate(pixels, applyDark=True)
                labelStr = 'Selected Pix : ' + str(np.round(val, 2)) + ' --> ' + str(np.round(valDark, 2)) + ' #/s'
            self.label_selectedPixValue.setText(labelStr)

    def updateCurrentPixelLabel(self):
        """
        This updates the labels showing the current pixel we're hovering over
        This is usually fast enough as long as we aren't averaging over too many images
        
        This function is called when we hover to a new pixel or the image updates
        """
        if self.pixelCurrent is not None:
            val = self.getPixCountRate([self.pixelCurrent])
            self.label_pixelInfo.setText(
                '(' + str(self.pixelCurrent[0]) + ' , ' + str(self.pixelCurrent[1]) + ') : ' + str(
                    np.round(val, 2)) + ' #/second')

            beammapData = np.loadtxt(self.beammap.file)
            resID = 0
            freq = 0
            feedline = 0
            board = 'a'
            freqCh = 0
            try:
                indx = \
                np.where((beammapData[:, 2] == self.pixelCurrent[0]) & (beammapData[:, 3] == self.pixelCurrent[1]))[0][
                    0]
                resID = beammapData[indx, 0]
                for roach in self.roachList:
                    freqFN = self.config.get('Roach ' + str(roach.num), 'freqList')
                    resIDs, freqs = np.loadtxt(freqFN, unpack=True, usecols=(0, 1))
                    try:
                        freqCh = int(np.where(resIDs == resID)[0][0])
                        freq = freqs[freqCh]
                        feedline = self.config.getint('Roach ' + str(roach.num), 'feedline')
                        board = self.config.get('Roach ' + str(roach.num), 'boardRange')
                        break
                    except IndexError:
                        pass
            except IndexError:
                pass
            self.label_pixelID.setText('ResID: ' + str(resID) + '\nFreq: ' +
                                       str(freq / 10 ** 9.) + ' GHz\nFeedline: ' +
                                       str(feedline) + '\nBoard: ' + board + '\nCh: ' +
                                       str(freqCh))

    def showContextMenu(self, point):
        """
        This function is called on a right click
        
        We don't need to clear and add the action everytime but, eh
        """
        self.contextMenu.clear()
        self.contextMenu.addAction('Plot Timestream', self.plotTimestream)
        self.contextMenu.addAction('Plot Histogram', self.plotHistogram)
        self.contextMenu.exec_(self.sender().mapToGlobal(point))  # Need to reference to global coordinate system

    def plotTimestream(self):
        """
        This function is called when the user clicks on the Plot Timestream action in the context menu
        
        It pops up a window showing the selected pixel's timestream
        """
        if len(self.selectedPixels):
            pixels = np.asarray([[p[0], p[1]] for p in self.selectedPixels])
        else:
            pixels = [self.pixelCurrent]

        window = PixelTimestreamWindow(pixels, parent=self)
        self.timeStreamWindows.append(window)
        window.closeWindow.connect( partial(self.timeStreamWindows.remove, window))  # remove from list if closed
        window.show()

    def plotHistogram(self):
        """
        This function is called when the user clicks on the Plot Histogram action in the context menu
        
        It pops up a window showing a histogram of count rates for the selected pixels
        """
        if len(self.selectedPixels):
            pixels = np.asarray([[p[0], p[1]] for p in self.selectedPixels])
        else:
            pixels = [self.pixelCurrent]
        window = PixelHistogramWindow(pixels, parent=self)
        self.histogramWindows.append(window)
        window.closeWindow.connect(partial(self.histogramWindows.remove, window))  # remove from list if closed
        window.show()

    def startObs(self):
        """
        When we start to observe we have to:
            - switch Firmware into photon collect mode
            - write START file to RAM disk for PacketMaster
        """
        if not self.observing:
            self.observing = True
            self.button_obs.setEnabled(False)
            self.button_obs.clicked.disconnect()
            self.textbox_target.setReadOnly(True)
            self.turnOnPhotonCapture()
            if self.takingDark<0 and self.takingFlat < 0:
                self.sciFactory = CalFactory('sum', dark=self.darkField, flat=self.flatField)
            self.packetmaster.startobs(self.config.paths.data)
            self.button_obs.setText('Stop Observing')
            self.button_obs.clicked.connect(self.stopObs)
            self.button_obs.setEnabled(True)

    def stopObs(self):
        """
        When we stop observing we need to:
            - Write QUIT file to RAM disk for PacketMaster
            - switch Firmware out of photon collect mode
            - Move any log files in the ram disk to the hard disk
        """
        if self.observing:
            self.observing = False
            self.button_obs.setEnabled(False)
            self.button_obs.clicked.disconnect()
            self.packetmaster.stopobs()
            getLogger('Dashboard').info("Stop Obs")
            if self.sciFactory is not None:
                #TODO get fram info and file name
                self.sciFactory.generate(threaded=True,
                                         fname=os.path.join(self.paths.data, 't{}.fits'.format(time.time())),
                                         name=self.textbox_target.text(),
                                         save=True, header=self.state())
            self.textbox_target.setReadOnly(False)
            self.button_obs.setText('Start Observing')
            self.button_obs.clicked.connect(self.startObs)
            self.button_obs.setEnabled(True)

    def toggleFlipper(self):
        getLogger('Dashboard').info("Toggling flipper!")
        laserStr = str(int(self.checkbox_flipper.isChecked())) + '0' * len(self.checkbox_laser_list)

        def flipperoff():
            getLogger('Dashboard').info("Toggling flipper back!")
            if self.laserController.laserOff():
                self.checkbox_flipper.setCheckState(False)
                self.checkbox_flipper.setText(str(self.checkbox_flipper.text()).rstrip(' ERROR'))
            else:
                getLogger('Dashboard').error("Toggling flipper back failed.")
                self.checkbox_flipper.setText(str(self.checkbox_flipper.text()).rstrip(' ERROR') + ' ERROR')
            self.logstate()

        if not self.laserController.toggleLaser(laserStr):
            getLogger('Dashboard').error("Toggling flipper failed.")
            self.checkbox_flipper.stateChanged.disconnect()
            self.checkbox_flipper.setText(str(self.checkbox_flipper.text()).rstrip(' ERROR')+' ERROR')
            self.checkbox_flipper.setCheckState(not self.checkbox_flipper.isChecked())
            self.checkbox_flipper.stateChanged.connect(self.toggleFlipper)
        else:
            self.checkbox_flipper.setText(str(self.checkbox_flipper.text()).rstrip(' ERROR'))
            if self.checkbox_flipper.isChecked():
                QtCore.QTimer.singleShot(500 * 1000, flipperoff)
        self.logstate()

    def setFilter(self, filter):
        self.combobox_filter.removeItem(mkidreadout.hardware.hsfw.NFILTERS)
        result = mkidreadout.hardware.hsfw.setfilter(int(filter), home=False,
                                                     host=self.config.filter.ip)
        if not result:
            self.combobox_filter.insertItem(mkidreadout.hardware.hsfw.NFILTERS, 'ERROR')
            self.combobox_filter.model().item(mkidreadout.hardware.hsfw.NFILTERS).setEnabled(False)
            self.combobox_filter.setCurrentIndex(mkidreadout.hardware.hsfw.NFILTERS)
        self.logstate()

    def laserCalClicked(self, _, laserCalStyle = "simultaneous"):
        laserTime = self.spinbox_laserTime.value()

        def lcaldone():
            if not self.laserController.laserOff():
                getLogger('Dashboard').info("Laser cal done. Lasers off.")
            else:
                getLogger('Dashboard').error("Unable to turn lasers off.")
            self.checkbox_flipper.setEnabled(True)
            self.logstate()

        # simultaneous is classic laser cal style, with all desired lasers at once
        if laserCalStyle == "simultaneous":
            # turn off flipper control until laser cal is done
            self.checkbox_flipper.setEnabled(False)
            laserStr = '0'+''.join([str(int(cb.isChecked())) for cb in self.checkbox_laser_list])

            if self.laserController.toggleLaser(laserStr):
                getLogger('ObsLog').info('Starting a {} laser cal for {} s.'.format(laserCalStyle, laserTime))
                QtCore.QTimer.singleShot(laserTime * 1000, lcaldone)
            else:
                getLogger('ObsLog').error('Failed to start  a {} laser cal. Lasers may be on.'.format(laserCalStyle))
                self.checkbox_flipper.setEnabled(True)
            self.logstate()

    def state(self):
        #TODO this is the crap that populates the headers and the log, it needs to be prompt enough that
        #it won't cause the GUI to be sluggish so polls should be used with caution
        from datetime import datetime
        return dict(target=self.textbox_target.text(), ditherx=np.nan, dithery=np.nan,
                    flipper='image', filter='1', ra='00:00:00.00', dec='00:00:00.00',
                    utc=datetime.utcnow(), roaches='raoch.yml')

    def logstate(self):
        targ, cmt = self.textbox_target.text(), self.textbox_log.toPlainText()
        # state = InstrumentState(target=targ, comment=cmt, flipper=None, laser)
        # targetname, telescope params, filter, dither x y ts state, roach info if first log
        state = self.state()
        getLogger('ObsLog').info('foo')

    def create_dock_widget(self):
        """
        Add buttons and controls
        """
        obs_dock_widget = QDockWidget(self)
        obs_dock_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)
        obs_widget = QWidget(obs_dock_widget)

        # Current time label
        label_currentTime = QLabel("UTC: ")
        getCurrentTime = QtCore.QTimer(self)
        getCurrentTime.setInterval(997)  # prime number :)
        updateCurrentTime = lambda: label_currentTime.setText("UTC: " + str(time.time()))
        getCurrentTime.timeout.connect(updateCurrentTime)
        getCurrentTime.start()

        # Mkid data directory
        label_dataDir = QLabel('Data Dir:')
        dataDir = self.config.get('properties', 'data_dir')
        textbox_dataDir = QLineEdit()
        textbox_dataDir.setText(dataDir)
        textbox_dataDir.textChanged.connect(partial(self.config.update, 'paths.data'))
        textbox_dataDir.setEnabled(False)

        # Start observing button
        self.button_obs = QPushButton("Start Observing")
        font = self.button_obs.font()
        font.setPointSize(24)
        self.button_obs.setFont(font)
        if self.observing:
            self.button_obs.setText('Stop Observing')
            self.button_obs.clicked.connect(self.stopObs)
        else:
            self.button_obs.setText('Start Observing')
            self.button_obs.clicked.connect(self.startObs)

        # dithering
        self.button_dither = QPushButton("Start Dithering")
        self.button_dither.clicked.connect(self.startDithering)

        # Filter
        self.combobox_filter = combobox_filter = QComboBox()
        self.combobox_filter.setMaxVisibleItems(mkidreadout.hardware.hsfw.NFILTERS + 1)
        label_filter = QLabel("Filter:")
        combobox_filter.addItems(map(str, range(1, mkidreadout.hardware.hsfw.NFILTERS + 1)))
        result = mkidreadout.hardware.hsfw.getfilter(self.config.filter.ip)
        if 'error' in str(result).lower():
            self.combobox_filter.insertItem(mkidreadout.hardware.hsfw.NFILTERS, 'ERROR')
            self.combobox_filter.model().item(mkidreadout.hardware.hsfw.NFILTERS).setEnabled(False)
            self.combobox_filter.setCurrentIndex(mkidreadout.hardware.hsfw.NFILTERS)
        else:
            combobox_filter.setCurrentIndex(int(result) - 1)
        combobox_filter.setToolTip("Select a filter")
        combobox_filter.activated[str].connect(self.setFilter)

        # log file
        label_target = QLabel("Target: ")
        self.textbox_target = QLineEdit()
        self.textbox_log = QTextEdit()
        autoLogTimer = QtCore.QTimer(self)
        autoLogTimer.setInterval(5 * 1000 * 60)
        autoLogTimer.timeout.connect(self.logstate)
        autoLogTimer.start()

        # ==================================
        # Image settings!
        # integration Time
        integrationTime = self.config.dashboard.average
        image_int_time = self.config.packetmaster.int_time
        max_int_time = self.config.dashboard.num_images_to_save
        label_integrationTime = QLabel('int time:')
        spinbox_integrationTime = QSpinBox()
        spinbox_integrationTime.setRange(1, max_int_time)
        spinbox_integrationTime.setValue(integrationTime)
        spinbox_integrationTime.setSuffix(' * {} s'.format(image_int_time))
        spinbox_integrationTime.setWrapping(False)
        spinbox_integrationTime.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_integrationTime.valueChanged.connect(partial(self.config.update, 'dashboard.average'))  # change in
        # config file

        # current num images integrated
        self.label_numIntegrated = QLabel('0/' + str(integrationTime))
        spinbox_integrationTime.valueChanged.connect(
            lambda x: self.label_numIntegrated.setText(self.label_numIntegrated.text().split('/')[0] + '/' + str(x)))
        spinbox_integrationTime.valueChanged.connect(lambda x: QtCore.QTimer.singleShot(10, self.convertImage))  #
        # remake current image after 10 ms

        # dark Image
        darkIntTime = self.config.dashboard.n_darks
        self.spinbox_darkImage = QSpinBox()
        self.spinbox_darkImage.setRange(1, max_int_time)
        self.spinbox_darkImage.setValue(darkIntTime)
        self.spinbox_darkImage.setSuffix(' * ' + str(image_int_time) + ' s')
        self.spinbox_darkImage.setWrapping(False)
        self.spinbox_darkImage.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        self.spinbox_darkImage.valueChanged.connect(partial(self.config.update, 'dashboard.n_darks'))  # change in
        # config file
        button_darkImage = QPushButton('Take Dark')

        def takeDark():
            self.darkField = None
            self.takingDark = self.spinbox_darkImage.value()

        button_darkImage.clicked.connect(takeDark)
        self.checkbox_darkImage = QCheckBox()
        self.checkbox_darkImage.setChecked(False)

        # flat Image
        flatIntTime = self.config.dashboard.n_flats
        self.spinbox_flatImage = QSpinBox()
        self.spinbox_flatImage.setRange(1, max_int_time)
        self.spinbox_flatImage.setValue(flatIntTime)
        self.spinbox_flatImage.setSuffix(' * ' + str(image_int_time) + ' s')
        self.spinbox_flatImage.setWrapping(False)
        self.spinbox_flatImage.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        self.spinbox_flatImage.valueChanged.connect(partial(self.config.update, 'dashboard.n_flats'))  # change in
        # config file
        button_flatImage = QPushButton('Take Flat')

        def takeFlat():
            self.flatField = None
            self.takingFlat = self.spinbox_flatImage.value()

        button_flatImage.clicked.connect(takeFlat)
        self.checkbox_flatImage = QCheckBox()
        self.checkbox_flatImage.setChecked(False)

        # maxCountRate
        maxCountRate = self.config.dashboard.max_count_rate
        minCountRate = self.config.dashboard.min_count_rate
        label_maxCountRate = QLabel('max:')
        spinbox_maxCountRate = QSpinBox()
        spinbox_maxCountRate.setRange(minCountRate, 2500)
        spinbox_maxCountRate.setValue(maxCountRate)
        spinbox_maxCountRate.setSuffix(' #/s')
        spinbox_maxCountRate.setWrapping(False)
        spinbox_maxCountRate.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        # minCountRate
        label_minCountRate = QLabel('min:')
        spinbox_minCountRate = QSpinBox()
        spinbox_minCountRate.setRange(0, maxCountRate)
        spinbox_minCountRate.setValue(minCountRate)
        spinbox_minCountRate.setSuffix(' #/s')
        spinbox_minCountRate.setWrapping(False)
        spinbox_minCountRate.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        # connections for max and min count rates
        spinbox_minCountRate.valueChanged.connect(partial(self.config.update, 'dashboard.min_count_rate'))  # change
        #  in config file
        spinbox_maxCountRate.valueChanged.connect(partial(self.config.update, 'dashboard.max_count_rate'))  # change
        #  in config file
        spinbox_minCountRate.valueChanged.connect(
            spinbox_maxCountRate.setMinimum)  # make sure min is always less than max
        spinbox_maxCountRate.valueChanged.connect(spinbox_minCountRate.setMaximum)
        convertSS = lambda x: QtCore.QTimer.singleShot(10, self.convertImage)
        spinbox_maxCountRate.valueChanged.connect(convertSS)  # remake current image after 10 ms
        spinbox_minCountRate.valueChanged.connect(convertSS)

        # Drop down menu for choosing image stretch
        label_stretch = QLabel("Image Stretch:")
        self.combobox_stretch = QComboBox()
        self.combobox_stretch.addItems(['linear', 'logarithmic', 'histogram equalization'])

        # Checkbox for showing unbeammaped pixels
        self.checkbox_showAllPix = QCheckBox('Show All Pixels')
        self.checkbox_showAllPix.setChecked(False)

        # Checkbox for interpolating dead (unbeammapped) pixels
        self.checkbox_interpolate = QCheckBox('Interpolate Dead Pixels')
        self.checkbox_interpolate.setChecked(False)
        # Checkbox for Smoothing image
        self.checkbox_smooth = QCheckBox('Smooth Image')
        self.checkbox_smooth.setChecked(False)

        # Checkbox for dithering image
        # self.checkbox_dither = QCheckBox('Dither Image')
        # self.checkbox_dither.setChecked(False)

        # Pixel info labels
        self.label_pixelInfo = QLabel('(, ) - (, ) : 0 #/s')
        self.label_pixelInfo.setMaximumWidth(250)
        self.label_selectedPixValue = QLabel('Selected Pix : 0 #/s')
        self.label_selectedPixValue.setMaximumWidth(250)
        self.label_pixelID = QLabel('ResID: -1\nFreq: 0 GHz\nFeedline: 1\nBoard: a\nCh: 0')

        # =============================================
        # Laser control!
        self.spinbox_laserTime = QDoubleSpinBox()
        self.spinbox_laserTime.setRange(1, 1800)
        self.spinbox_laserTime.setValue(300)
        self.spinbox_laserTime.setSuffix(' s')
        self.spinbox_laserTime.setSingleStep(1.)
        self.spinbox_laserTime.setWrapping(False)
        self.spinbox_laserTime.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        self.spinbox_laserTime.setButtonSymbols(QAbstractSpinBox.NoButtons)
        button_laserCal = QPushButton("Start Laser Cal")

        self.checkbox_laser_list = []
        for lname in self.config.lasercontrol.lasers:
            checkbox_laser = QCheckBox(lname)
            checkbox_laser.setChecked(False)
            self.checkbox_laser_list.append(checkbox_laser)

        button_laserCal.clicked.connect(self.laserCalClicked)

        # Also have the pupil imager flipper controlled with laser box arduino
        self.checkbox_flipper = QCheckBox('SBIG Flipper to Image')
        self.checkbox_flipper.setChecked(False)
        self.checkbox_flipper.stateChanged.connect(self.toggleFlipper)

        # ================================================
        # Layout on GUI

        vbox = QVBoxLayout()
        vbox.addWidget(label_currentTime)

        hbox_dataDir = QHBoxLayout()
        hbox_dataDir.addWidget(label_dataDir)
        hbox_dataDir.addWidget(textbox_dataDir)
        vbox.addLayout(hbox_dataDir)

        vbox.addWidget(self.button_obs)

        hbox_target = QHBoxLayout()
        hbox_target.addWidget(label_target)
        hbox_target.addWidget(self.textbox_target)
        vbox.addLayout(hbox_target)

        vbox.addWidget(self.textbox_log)

        hbox_filter = QHBoxLayout()
        hbox_filter.addWidget(self.button_dither)
        hbox_filter.addStretch()
        hbox_filter.addWidget(label_filter)
        hbox_filter.addWidget(combobox_filter)
        vbox.addLayout(hbox_filter)

        vbox.addStretch()

        hbox_intTime = QHBoxLayout()
        hbox_intTime.addWidget(label_integrationTime)
        hbox_intTime.addWidget(spinbox_integrationTime)
        hbox_intTime.addWidget(self.label_numIntegrated)
        hbox_intTime.addStretch()
        vbox.addLayout(hbox_intTime)

        hbox_darkImage = QHBoxLayout()
        hbox_darkImage.addWidget(self.spinbox_darkImage)
        hbox_darkImage.addWidget(button_darkImage)
        hbox_darkImage.addWidget(self.checkbox_darkImage)
        hbox_darkImage.addStretch()
        vbox.addLayout(hbox_darkImage)

        hbox_flatImage = QHBoxLayout()
        hbox_flatImage.addWidget(self.spinbox_flatImage)
        hbox_flatImage.addWidget(button_flatImage)
        hbox_flatImage.addWidget(self.checkbox_flatImage)
        hbox_flatImage.addStretch()
        vbox.addLayout(hbox_flatImage)

        hbox_CountRate = QHBoxLayout()
        hbox_CountRate.addWidget(label_minCountRate)
        hbox_CountRate.addWidget(spinbox_minCountRate)
        hbox_CountRate.addSpacing(20)
        hbox_CountRate.addWidget(label_maxCountRate)
        hbox_CountRate.addWidget(spinbox_maxCountRate)
        hbox_CountRate.addStretch()
        vbox.addLayout(hbox_CountRate)

        hbox_stretch = QHBoxLayout()
        hbox_stretch.addWidget(label_stretch)
        hbox_stretch.addWidget(self.combobox_stretch)
        hbox_stretch.addStretch()
        vbox.addLayout(hbox_stretch)

        vbox.addWidget(self.checkbox_showAllPix)
        vbox.addWidget(self.checkbox_interpolate)
        vbox.addWidget(self.checkbox_smooth)
        vbox.addWidget(self.label_selectedPixValue)
        vbox.addWidget(self.label_pixelInfo)
        vbox.addWidget(self.label_pixelID)

        vbox.addStretch()

        hbox_laser = QHBoxLayout()
        hbox_laser.addWidget(self.spinbox_laserTime)
        hbox_laser.addWidget(button_laserCal)
        vbox.addLayout(hbox_laser)
        for checkbox_laser in self.checkbox_laser_list:
            vbox.addWidget(checkbox_laser)
        vbox.addWidget(self.checkbox_flipper)

        obs_widget.setLayout(vbox)
        obs_dock_widget.setWidget(obs_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, obs_dock_widget)

    def create_image_widget(self):
        """
        Make image part of the GUI
        
        Heirarchy is:
                  QMainWindow
                       |
                    QFrame
                       |
                 QGraphicsView   QGraphicsPixmapItem
                        \          /         |      \
                         \        /       QPixmap   QGraphicsBlurEffect
                       QGraphicsScene
        
        To update image:
            - Replace the QPixmap in the QGraphicsPixmapItem
            - Call QGraphicsPixmapItem.graphicsEffect().setEnabled(True) to turn on blurring
            - Call update() on the QGraphicsPixmapItem to repaint the QGraphicsScene
            
        """
        # self.imageFrame = QtGui.QFrame(parent=self)
        # self.imageFrame.setFrameShape(QtGui.QFrame.Box)
        # self.imageFrame.setFrameShadow(QtGui.QFrame.Sunken)
        # self.imageFrame.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)

        q_image = QImage(1, 1, QImage.Format_Mono)
        q_image.fill(Qt.black)
        # grview = QGraphicsView(self.imageFrame)
        grview = QGraphicsView()
        scene = QGraphicsScene(parent=grview)
        self.grPixMap = QGraphicsPixmapItem(QPixmap(q_image), None, scene)
        self.grPixMap.mousePressEvent = self.mousePressed
        self.grPixMap.mouseMoveEvent = self.mouseMoved
        self.grPixMap.mouseReleaseEvent = self.mouseReleased
        self.grPixMap.setAcceptHoverEvents(True)
        self.grPixMap.hoverMoveEvent = self.mouseMoved
        blurEffect = QGraphicsBlurEffect()
        blurEffect.setBlurRadius(1.5 * self.config.dashboard.image_scale)
        self.grPixMap.setGraphicsEffect(blurEffect)
        self.grPixMap.graphicsEffect().setEnabled(False)
        grview.setScene(scene)
        grview.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        grview.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        grview.setContextMenuPolicy(Qt.CustomContextMenu)
        grview.customContextMenuRequested.connect(self.showContextMenu)
        grview.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # layout=QVBoxLayout()
        # layout.addWidget(grview)
        # layout.addStretch()
        # self.imageFrame.setLayout(layout)

        # self.setCentralWidget(self.imageFrame)
        self.setCentralWidget(grview)

    def create_menu(self):
        self.file_menu = self.menuBar().addMenu("&File")
        telescope_action = self.create_action("&Telescope Info", slot=self.telescopeWindow.show, shortcut="Ctrl+T",
                                              tip="Show Telescope Info")

        # TODO this shortcircuts the logic in starObs/stopObs and will desync the observing button
        photonCapOn_action = self.create_action("Start &Photon Capture", slot=self.turnOnPhotonCapture,
                                                shortcut="Ctrl+P", tip="Tell Roaches to send photon packets")
        photonCapOff_action = self.create_action("Stop &Photon Capture", slot=self.turnOffPhotonCapture,
                                                 shortcut="Ctrl+Shift+P",
                                                 tip="Tell Roaches to stop sending photon packets")
        quit_action = self.create_action("&Quit", slot=self.close, shortcut="Ctrl+Q", tip="Close the application")

        add_actions(self.file_menu,
                    (telescope_action, photonCapOn_action, photonCapOff_action, viewLogs_action, None, quit_action))

        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.create_action("&About", shortcut='F1', slot=self.on_about, tip='About the demo')
        add_actions(self.help_menu, (about_action,))

    def on_about(self):
        msg = ("{} Dashboard!!\n"
               "Click and drag on Pixels to select them.\n"
               "You can select non-contiguous pixels using SHIFT.\n"
               "Right click to plot the timestream of your selected pixels!\n\n"
               "Author: Alex Walter\n"
               "Date: Jul 3, 2016").format(self.config.instrument)
        QMessageBox.about(self, "MKID-ROACH2 software", msg.strip())

    def create_action(self, text, slot=None, shortcut=None, icon=None, tip=None, checkable=False,
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
        """
        Clean up before closing
        """
        if self.observing:
            self.stopObs()

        self.workers[0].search = False  # stop searching for new images
        del self.grPixMap  # Get segfault if we don't delete this. Something about signals in the queue trying to access deleted objects...
        for thread in self.threadPool:  # They should all be done at this point, but just in case
            thread.quit()
        for window in self.timeStreamWindows:
            window.close()
        for window in self.histogramWindows:
            window.close()
        self.turnOffPhotonCapture()  # stop sending photon packets
        self.telescopeWindow._want_to_close = True
        self.telescopeWindow.close()

        self.hide()
        time.sleep(1)
        self.packetmaster.quit()

        QtCore.QCoreApplication.instance().quit()


DEFAULT_CFG_FILE = os.path.join(os.path.dirname(__file__), 'dashboard.yml')

if __name__ == "__main__":
    setup_logging()
    app = QApplication(sys.argv)

    parser = argparse.ArgumentParser(description='MKID Dashboard')
    parser.add_argument('roaches', nargs='+', type=int, help='Roach numbers')
    parser.add_argument('-c', '--config', default=DEFAULT_CFG_FILE, dest='config',
                        type=str, help='The config file')
    parser.add_argument('-o', '--offline', default=False, dest='offline', action='store_true', help='Run offline')
    args = parser.parse_args()

    form = MKIDDashboard(args.roaches, config=args.config, offline=args.offline)
    form.show()
    app.exec_()
