#!/bin/env python
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

============Dashboard config
config.paths
logs: directory where logs will be written
data: directory where bin files, fits files are saved and freq files are sought

config.beammap omitting will load the default beammap for instrument
config.beammap.default: str will load default for
presence of config.beammap.freqfiles will result in freqs from those files being attached to the loaded .bmap file
but these are not used for seting the LO freq. the frequences will be pulled from whatever was specified in the roach config
those files are assumed relative to config.paths.data (even if they look to be FQPs)


 """
from __future__ import print_function

import argparse
import json
import os
import sys
import time
from datetime import datetime
from functools import partial

import numpy as np
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
from PyQt4.QtGui import *
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import wcs
import astropy.units as units

import mkidcore.corelog
import mkidcore.instruments
from mkidcore.instruments import compute_wcs_ref_pixel
import mkidreadout.config
import mkidreadout.configuration.sweepdata as sweepdata
import mkidreadout.hardware.hsfw
from mkidcore.corelog import create_log, getLogger
from mkidcore.fits import CalFactory, combineHDU, summarize
from mkidcore.objects import Beammap
from mkidreadout.channelizer.Roach2Controls import Roach2Controls
from mkidreadout.hardware.lasercontrol import LaserControl
from mkidreadout.hardware.telescope import Palomar, Subaru, NoScope
from mkidreadout.readout.guiwindows import DitherWindow, PixelHistogramWindow, PixelTimestreamWindow, TelescopeWindow
from mkidreadout.readout.packetmaster import Packetmaster
from mkidreadout.utils.utils import interpolateImage


def add_actions(target, actions):
    for action in actions:
        if action is None:
            target.addSeparator()
        else:
            target.addAction(action)


def build_hbox(things, stretch=True):
    h = QHBoxLayout()
    for t in things:
        h.addWidget(t)
    if stretch:
        h.addStretch()
    return h


class LiveImageFetcher(QtCore.QObject):  # Extends QObject for use with QThreads
    """
    This class fetches images from PacketMaster for the live view and fits files

    SIGNALS
        newImage - emits when an image is avaialble
        finished - emits when self.search is set to False
    """
    newImage = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal()

    def __init__(self, sharedim, inttime=0.1, parent=None):
        """
        INPUTS:
            sharedim - ImageCube
            inttime - the integration time interval
            parent - Leave as None so that we can add to new thread
        """
        super(QtCore.QObject, self).__init__(parent)
        self.imagebuffer = sharedim
        self.inttime = inttime
        self.search = True

    def update_inttime(self, it):
        self.inttime = float(it)

    def run(self):
        """
        Infinite loop that keeps checking directory for an image file
        When it finds an image, it parses it and emits imageComplete signal
        It only returns files that have timestamps > than the last file's timestamp
        
        Loop will continue until you set self.search to False. Then it will emit finished signal
        
        INPUTS:
            removeOldFiles - remove .img and .png files after we read them
        """
        self.search = True
        while self.search:
            try:
                utc = datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                self.imagebuffer.startIntegration(startTime=time.time(), integrationTime=self.inttime)
                data = self.imagebuffer.receiveImage()
                if not data.sum():
                    getLogger('Dashboard').warning('Received a frame of zeros from packetmaster!')
                ret = fits.ImageHDU(data=data)
                ret.header['utcstart'] = utc
                ret.header['exptime'] = self.inttime
                ret.header['wavecal'] = self.imagebuffer.wavecalID.decode('UTF-8', "backslashreplace")
                if ret.header['wavecal']:
                    ret.header['wmin'] = self.imagebuffer.wvlStart
                    ret.header['wmax'] = self.imagebuffer.wvlStop
                else:
                    ret.header['wmin'] = 'NaN'
                    ret.header['wmax'] = 'NaN'
                self.newImage.emit(ret)
            except RuntimeError as e:
                getLogger('Dashboard').debug('Image stream unavailable: {}'.format(e))
            except Exception:
                getLogger('Dashboard').error('Problem', exc_info=True)
        self.finished.emit()


_stretchtime = [0,0]


class ConvertPhotonsToRGB(QtCore.QObject):
    """
    This class takes 2D arrays of photon counts and converts them into a QImage
    It needs to know how to map photon counts into an RGB color
    Usually just an 8bit grey color [0, 256) but also turns maxxed out pixel red
    
    SIGNALS
        convertedImage - emits when it's done converting an image
    """
    convertedImage = QtCore.pyqtSignal(object)

    def __init__(self, cal_factory=None, image=None, minCountCutoff=0, maxCountCutoff=450, stretchMode='log',
                 interpolate=False, makeRed=True, parent=None):
        """
        INPUTS:
            image - 2D numpy array of photon counts with np.nan where beammap failed
            minCountCutoff - anything <= this number of counts will be black
            maxCountCutoff - anything >= this number of counts will be red
            stretchMode - can be log, linear, hist
            interpolate - interpolate over np.nan pixels
        """
        super(QtCore.QObject, self).__init__(parent)
        self.cal_factory = cal_factory
        self.image = np.copy(image) if image is not None else None
        if cal_factory is None and image is None:
            raise ValueError('Must specify cal_factory or image')
        self.minCountCutoff = minCountCutoff
        self.maxCountCutoff = maxCountCutoff
        self.interpolate = interpolate
        self.makeRed = makeRed
        self.redPixels = []
        self.stretchMode = stretchMode

    def stretchImage(self):
        """
        map photons to greyscale
        """
        global _stretchtime
        tic = time.time()

        if self.image is None:
            self.image = self.cal_factory.generate(name='LiveImage', bias=0, maskvalue=np.nan).data

        # first interpolate and find hot pixels
        if self.interpolate:
            self.image = interpolateImage(self.image)

        self.image[~np.isfinite(self.image)] = 0  # get rid of np.nan's

        self.redPixels = self.image >= self.maxCountCutoff if self.makeRed else []

        if self.stretchMode == 'logarithmic':
            imageGrey = self.logStretch()
        elif self.stretchMode == 'linear':
            imageGrey = self.linStretch()
        elif self.stretchMode == 'histogram equalization':
            imageGrey = self.histEqualization()
        else:
            raise ValueError('Unknown stretch mode')

        self.makeQPixMap(imageGrey)
        _stretchtime = (_stretchtime[0] + time.time() - tic, _stretchtime[1]+1)

        if _stretchtime[1] > 60:
            msg = 'ConvertPhotonsToRGB.stretchImage took {:.3} ms/frame for the last {} frames.'
            getLogger('Dashboard').debug(msg.format(1000*_stretchtime[0]/_stretchtime[1], _stretchtime[1]))
            _stretchtime = (0, 0)

    def logStretch(self):
        """
        map photon counts to greyscale logarithmically
        """
        self.image.clip(self.minCountCutoff, self.maxCountCutoff, self.image)
        maxVal = np.amax(self.image)
        minVal = np.amin(self.image)
        maxVal = np.amax([minVal + 1, maxVal])

        a=10.
        image2 = 255.*np.log10(a*(self.image-minVal) / (maxVal - minVal) + 1.)/np.log10(a+1.)
        #image2 = 255. / (np.log10(1 + maxVal - minVal)) * np.log10(1 + self.image - minVal)
        return image2

    def linStretch(self):
        """
        map photon count to greyscale linearly (max 255 min 0)
        """
        self.image.clip(self.minCountCutoff, self.maxCountCutoff, self.image)

        maxVal = self.maxCountCutoff
        minVal = self.minCountCutoff
        maxVal = np.amax([minVal + 1, maxVal])

        image2 = (self.image - minVal) / (1.0 * maxVal - minVal) * 255.
        return image2

    def histEqualization(self, bins = 256):
        """
        perform a histogram Equalization. This tends to make the contrast better
        
        if self.logStretch is True the histogram uses logarithmic spaced bins
        """
        if self.logStretch:
            self.minCountCutoff = max(self.minCountCutoff, 1)

        self.image.clip(self.minCountCutoff, self.maxCountCutoff, self.image)
        maxVal = np.amax(self.image)

        if self.logStretch:
            bins = np.logspace(np.log10(self.minCountCutoff), np.log10(maxVal), bins)
        imhist, imbins = np.histogram(self.image.flatten(), bins, density=True)

        cdf = (imhist * np.diff(imbins)).cumsum() * 256

        image2 = np.interp(self.image.flatten(), imbins[:-1], cdf)
        image2 = image2.reshape(self.image.shape)
        image2[self.image <= self.minCountCutoff] = 0

        return image2

    def makeQPixMap(self, image):
        """
        This function makes the QImage object
        
        INPUTS:
            image - 2D numpy array of [0,256) grey colors
        """
        image2 = image.astype(np.uint32)

        redMask = np.copy(image2)
        redMask[self.redPixels] = 0
        #           24-32 -> A  16-24 -> R     8-16 -> G      0-8 -> B
        imageRGB = (255 << 24 | image2 << 16 | redMask << 8 | redMask).flatten()  # pack into RGBA
        q_im = QImage(imageRGB, self.image.shape[1], self.image.shape[0], QImage.Format_RGB32)

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
        self.config = mkidreadout.config.load(config)
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
        self.takingDark = False  # Flag for taking dark image. Indicates number of images we still need for darkField
        # image.
        self.takingFlat = False  # Flag for taking flat image. Indicates number of images we still need for flatField
        # image

        # Initialize PacketMaster8
        getLogger('Dashboard').info('Initializing packetmaster...')
        imgcfg = dict(self.config.dashboard)
        imgcfg['n_wave_bins'] = 1
        if 'forwarding' in self.config.packetmaster:
            forwarding = dict(self.config.packetmaster.forwarding)
        else:
            forwarding = None
        self.packetmaster = Packetmaster(len(roachNums), self.config.packetmaster.captureport,
                                         useWriter=not self.offline, sharedImageCfg={'dashboard': imgcfg},
                                         beammap=self.config.beammap, forwarding=forwarding, recreate_images=True)
        self.liveimage = self.packetmaster.sharedImages['dashboard']

        self.liveimage.startIntegration(startTime=time.time(), integrationTime=1)
        data = self.liveimage.receiveImage()
        getLogger('Dahsboard').debug(data)

        # Laser Controller
        getLogger('Dashboard').info('Setting up laser control...')
        self.laserController = LaserControl(self.config.lasercontrol.ip, self.config.lasercontrol.port,
                                            self.config.lasercontrol.receive_port)

        # telscope TCS connection
        getLogger('Dashboard').info('Setting up telescope connection...')
        if self.config.telescope.ip is None:
            self.telescopeController = NoScope()
        else:
            if self.config.instrument.lower() == 'mec':
                self.telescopeController = Subaru(ip=self.config.telescope.ip, user=self.config.telescope.user,
                                                  password=self.config.telescope.password)
            elif self.config.instrument.lower() == 'dark':
                self.telescopeController = Palomar(ip=self.config.telescope.ip, port=self.config.telescope.port,
                                                   receivePort=self.config.telescope.receive_port)
            elif self.config.instrument.lower() == 'bluefors':
                self.telescopeController = NoScope()

        self.telescopeWindow = TelescopeWindow(self.telescopeController)
        
        # This polling loop is to help ensure that queries to the state don't lag
        self.last_tcs_poll = self.telescopeController.get_header()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_tcs)
        timer.setInterval(1000)
        timer.start()

        if self.config.instrument.lower() == 'bluefors':
            self.dither_dialog = None
        else:
            #Dither window
            self.dither_dialog = DitherWindow(self.config.dither.url, parent=self)

            #TODO sort out the two dither logs
            def logdither(status):
                if status.offline or status.haserrors:
                    msg = 'Dither completed with errors: "{}", Conex Status="{}"'.format(
                        status.state, status.conexstatus)
                    getLogger('Dashboard').error(msg)
                else:
                    dither_result = 'Dither Path: {}\n'.format(str(status.last_dither).replace('\n', '\n   '))
                    getLogger('Dashboard').info(dither_result)
                    getLogger('dither').info(dither_result)

            def logdither(d):
                state = d['status']['state'][1]
                if state == 'Stopped':
                    getLogger('Dashboard').error("Dither aborted early by user STOP. Conex Status="+str(d['status']['conexstatus']))
                elif state.startswith('Error'):
                    getLogger('Dashboard').error("Dither aborted from error. Conex State="+state+" Conex Status="+str(d['status']['conexstatus']))
                dither_dict = d['dither']
                msg="Dither Path: ({}, {}) --> ({}, {}), {} steps {} seconds".format(
                        dither_dict['startx'], dither_dict['starty'],
                        dither_dict['endx'], dither_dict['endy'],
                        dither_dict['n'], dither_dict['t'])
                if 'subStep' in dither_dict.keys() and dither_dict['subStep']>0 and \
                   'subT' in dither_dict.keys() and dither_dict['subT']>0:
                    msg = msg+" +/-{} for {} seconds".format(dither_dict['subStep'], dither_dict['subT'])
                msg = msg+"\n\tstarts={}\n\tends={}\n\tpath={}\n".format(dither_dict['startTimes'],
                        dither_dict['endTimes'], zip(dither_dict['xlocs'], dither_dict['ylocs']))
                getLogger('dither').info(msg)
                getLogger('Dashboard').info(msg)

            self.dither_dialog.complete.connect(logdither)
            self.dither_dialog.statusupdate.connect(self.logstate)
            self.dither_dialog.hide()

        # Connect to ROACHES and initialize network port in firmware
        getLogger('Dashboard').info('Connecting roaches and loading beammap...')
        if not self.offline:
            for roachNum in roachNums:
                roach = Roach2Controls(self.config.roaches.get('r{}.ip'.format(roachNum)),
                                       self.config.roaches.fpgaparamfile, num=roachNum,
                                       feedline=self.config.roaches.get('r{}.feedline'.format(roachNum)),
                                       range=self.config.roaches.get('r{}.range'.format(roachNum)),
                                       verbose=False, debug=False)
                if not roach.connect() and not roach.issetup:
                    raise RuntimeError('Roach r{} has not been setup.'.format(roachNum))
                roach.loadCurTimestamp()
                roach.setPhotonCapturePort(self.config.packetmaster.captureport)
                self.roachList.append(roach)
            self.turnOnPhotonCapture()
        self.loadBeammap()

        if self.config.dashboard.get('wavecal',''):
            try:
                self.packetmaster.applyWvlSol(self.config.dashboard.wavecal, self.beammap)
            except IOError:
                getLogger('Dashboard').critical('Unable to load wavecal {}.'.format(self.config.dashboard.wavecal))
                exit(1)

        # Setup search for image files from cuber
        getLogger('Dashboard').info('Setting up image searcher...')
        self.imageFetcher = LiveImageFetcher(self.liveimage, self.config.dashboard.inttime, parent=None)
        fetcherthread = self.startworker(self.imageFetcher, 'imageFetcher')
        fetcherthread.started.connect(self.imageFetcher.run)
        self.imageFetcher.newImage.connect(self.convertImage)
        self.imageFetcher.finished.connect(fetcherthread.quit)

        # Setup GUI
        getLogger('Dashboard').info('Setting up GUI...')
        self.setWindowTitle(self.config.instrument + ' Dashboard')
        self.create_image_widget()
        self.create_dock_widget()
        self.contextMenu = QMenu(self)  # pops up on right click
        self.create_menu()  # file menu

        # Connect to Filter wheel
        self.setFilter(None)

        QtCore.QTimer.singleShot(10, fetcherthread.start)  # start the thread after a second

    def update_tcs(self):
        self.last_tcs_poll = self.telescopeController.get_header()
        # getLogger('Dashboard').debug(self.last_tcs_poll)

    def startworker(self, obj, name):
        self.workers.append(obj)
        thread = QtCore.QThread(parent=self)
        self.threadPool.append(thread)
        thread.setObjectName(name)
        obj.moveToThread(thread)
        return thread

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
            roach.setMaxCountRate(self.config.dashboard.max_count_rate)
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
            self.beammap = self.config.beammap
        except KeyError:
            getLogger('Dashboard').warning("No beammap specified in config, using default")
            self.beammap = Beammap(default=self.config.instrument)

        self.beammapFailed = self.beammap.failmask

        getLogger('Dashboard').info('Loaded beammap: %s', self.beammap)

        if self.offline:
            return

        for roach in self.roachList:
            ffile = roach.tagfile(self.config.roaches.get('r{}.freqfileroot'.format(roach.num)),
                                  dir=self.config.paths.setup)
            roach.setLOFreq(self.config.roaches.get('r{}.lo_freq'.format(roach.num)))
            roach.loadBeammapCoords(self.beammap, freqListFile=ffile)

        getLogger('Dashboard').info('Loaded beammap into roaches')

    def addDarkImage(self, photonImage):
        self.darkFactory = CalFactory('dark', images=(photonImage,))
        self.takingDark = False
        self.darkField = self.darkFactory.generate(fname=self.darkfile, badmask=self.beammapFailed, save=True,
                                                   name=os.path.splitext(os.path.basename(self.darkfile))[0],
                                                   overwrite=True)
        getLogger('Dashboard').info('Finished dark:\n {}'.format(summarize(self.darkField).replace('\n', '\n  ')))
        self.checkbox_darkImage.setChecked(True)
        self.spinbox_minLambda.setEnabled(True)
        self.spinbox_maxLambda.setEnabled(True)
        self.spinbox_integrationTime.setEnabled(True)

    def addFlatImage(self, photonImage):
        self.flatFactory = CalFactory('flat', images=(photonImage,), dark=self.darkField)
        self.takingFlat = False
        self.flatField = self.flatFactory.generate(fname=self.flatfile, badmask=self.beammapFailed, save=True,
                                                   name=os.path.splitext(os.path.basename(self.flatfile))[0],
                                                   overwrite=True)
        getLogger('Dashboard').info('Finished flat:\n {}'.format(summarize(self.flatField).replace('\n', '\n  ')))
        self.checkbox_flatImage.setChecked(True)
        self.spinbox_minLambda.setEnabled(True)
        self.spinbox_maxLambda.setEnabled(True)
        self.spinbox_integrationTime.setEnabled(True)

    def convertImage(self, photonImage=None):
        """
        This function is automatically called when ImageSearcher object finds a new image file from cuber program
        We also call this function if we change the image processing controls like
            min/max count rate
        
        Here we set up a converter object to parse the photon counts into an RBG QImage object
        We do this in a separate thread because it might take a while to process
        
        INPUTS:
            photonImage - 2D numpy array of photon counts (type np.uint16 if from ImageSearcher object)
        """
        # If there's new data, append it
        if photonImage is not None:

            state = self.state()
            for k in state:
                try:
                    if len(state[k]) > 1 and not type(state[k]) in (str, unicode):
                        state[k] = json.dumps(state[k])  # header values must be scalar
                except TypeError:
                    pass
            photonImage.header.update(state)

            if self.config.instrument.lower() != 'bluefors':
                w = wcs.WCS(naxis=2)
                w.wcs.ctype = ["RA--TAN", "DEC-TAN"]
                w._naxis1, w._naxis2 = photonImage.shape
                c = SkyCoord(photonImage.header['ra'], photonImage.header['dec'], unit=(units.hourangle, units.deg),
                             obstime='J' + str(photonImage.header['equinox']))
                w.wcs.crval = np.array([c.ra.deg, c.dec.deg])
                w.wcs.crpix = compute_wcs_ref_pixel(json.loads(photonImage.header['dither_pos']),
                                                    self.config.dashboard.dither_home,
                                                    self.config.dashboard.dither_ref)
                do_rad = np.deg2rad(self.config.dashboard.device_orientation)
                w.wcs.pc = np.array([[np.cos(do_rad), -np.sin(do_rad)],
                                     [np.sin(do_rad), np.cos(do_rad)]])
                w.wcs.cdelt = [self.config.dashboard.platescale/3600.0, self.config.dashboard.platescale/3600.0]
                w.wcs.cunit = ["deg", "deg"]
                photonImage.header.update(w.to_header())

            self.imageList.append(photonImage)
            self.fitsList.append(photonImage)  # for the stream

            self.imageList = self.imageList[-max(self.config.dashboard.timestream_samples, 1):]

            if self.takingDark:
                self.addDarkImage(photonImage)
            elif self.takingFlat:
                self.addFlatImage(photonImage)
            elif self.observing:
                self.sciFactory.add_image(photonImage)
        elif not self.imageList:
            return

        # If we've got a full set of data package it as a fits file
        tconv = lambda x: (datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1, 1)).total_seconds()
        tstart = tconv(self.fitsList[0].header['utcstart']) if self.fitsList else time.time()
        tstamp = int(time.time())
        if ((sum([i.header['exptime'] for i in self.fitsList]) >= self.config.dashboard.fitstime) or
            (tstamp - tstart) >= self.config.dashboard.fitstime):
            combineHDU(self.fitsList, fname=os.path.join(self.config.paths.data, 'stream{}.fits.gz'.format(tstamp)),
                       name=str(tstamp), save=True, threaded=True)
            self.fitsList = []

        # Get the (average) photon count image
        if self.checkbox_flatImage.isChecked() and self.flatField is None:
            try:
                self.flatField = fits.open(self.flatfile)
            except IOError:
                self.flatField = None
                self.checkbox_flatImage.setChecked(False)
                getLogger('Dashboard').warning('Unable to load flat from {}'.format(self.flatfile))

        if self.checkbox_darkImage.isChecked() and self.darkField is None:
            try:
                self.darkField = fits.open(self.darkfile)
            except IOError:
                self.darkField = None
                self.checkbox_darkImage.setChecked(False)
                getLogger('Dashboard').warning('Unable to load flat from {}'.format(self.darkfile))

        # Set up worker object and thread for the display.
        #  All of this code could be axed if the live image was broken out into a separate program
        cf = CalFactory('avg', images=self.imageList[-1:],
                        dark=self.darkField if self.checkbox_darkImage.isChecked() else None,
                        flat=self.flatField if self.checkbox_flatImage.isChecked() else None,
                        mask=self.beammapFailed)

        converter = ConvertPhotonsToRGB(cal_factory=cf, minCountCutoff=self.config.dashboard.min_count_rate,
                                        maxCountCutoff=self.config.dashboard.max_count_rate,
                                        stretchMode=self.combobox_stretch.currentText(),
                                        interpolate=self.checkbox_interpolate.isChecked(),
                                        makeRed=not self.checkbox_smooth.isChecked())  # no red pixels if smoothing

        thread = self.startworker(converter, "convertImage_{}".format(len(self.threadPool)))
        thread.started.connect(converter.stretchImage)
        converter.convertedImage.connect(lambda x: thread.quit())
        converter.convertedImage.connect(self.updateImage)
        thread.finished.connect(partial(self.threadPool.remove, thread))  # delete these when done to avoid mem leak
        thread.finished.connect(partial(self.workers.remove, converter))
        thread.start()  # When it's done converting the worker will emit a convertedImage Signal

    @property
    def flatfile(self):
        file = mkidreadout.config.tagstr(self.config.dashboard.get('flatname', ''))
        if file:
            return file if 'fit' in os.path.splitext(file)[1].lower() else file + '.fits'
        wvstr = '_{:.0f}-{:.0f}nm'.format(self.spinbox_minLambda.value(), self.spinbox_maxLambda.value())
        name = mkidreadout.config.tagstr('flat' + wvstr if self.checkbox_usewave else '' + '_{time}')
        return os.path.join(self.config.paths.data, name + '.fits')

    @property
    def darkfile(self):
        file = mkidreadout.config.tagstr(self.config.dashboard.get('darkname', ''))
        if file:
            return file if 'fit' in os.path.splitext(file)[1].lower() else file + '.fits'
        wvstr = '_{:.0f}-{:.0f}nm'.format(self.spinbox_minLambda.value(), self.spinbox_maxLambda.value())
        name = mkidreadout.config.tagstr('dark' + wvstr if self.checkbox_usewave else '' + '_{time}')
        return os.path.join(self.config.paths.data, name+'.fits')

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
        imageScale = self.config.dashboard.image_scale

        #TODO scale the image based on the size of the gui window!

        q_image = q_image.scaledToWidth(q_image.width() * imageScale)
        self.grPixMap.pixmap().convertFromImage(q_image)

        # Possibly smooth image
        self.grPixMap.graphicsEffect().setEnabled(self.checkbox_smooth.isChecked())

        # Dither image
        # if self.checkbox_dither.isChecked(): print 'dithering'

        # Resize the GUI to fit whole image
        # borderSize = 0  # 24   # Not sure how to get the size of the frame's border so hardcoded this for now
        # imgSize = self.grPixMap.pixmap().size()
        # frameSize = QtCore.QSize(imgSize.width() + borderSize, imgSize.height() + borderSize)
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
        if event.button() == QtCore.Qt.LeftButton:
            self.clicking = True
            x_pos = int(np.floor(event.pos().x() / self.config.dashboard.image_scale))
            y_pos = int(np.floor(event.pos().y() / self.config.dashboard.image_scale))
            getLogger('Dashboard').info('Clicked (' + str(x_pos) + ' , ' + str(y_pos) + ')')
            if QApplication.keyboardModifiers() != QtCore.Qt.ShiftModifier:
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
        x_pos = int(np.floor(event.pos().x() / self.config.dashboard.image_scale))
        y_pos = int(np.floor(event.pos().y() / self.config.dashboard.image_scale))
        if ((x_pos, y_pos) != self.pixelCurrent and
            x_pos >= 0 and y_pos >= 0 and
            x_pos < self.beammap.ncols and
            y_pos < self.beammap.nrows):

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
        if event.button() == QtCore.Qt.LeftButton and self.clicking:
            self.clicking = False
            x_pos = int(np.floor(event.pos().x() / self.config.dashboard.image_scale))
            y_pos = int(np.floor(event.pos().y() / self.config.dashboard.image_scale))
            getLogger('Dashboard').info('Released (' + str(x_pos) + ' , ' + str(y_pos) + ')')

            x_start, x_end = sorted([x_pos, self.pixelClicked[0]])
            y_start, y_end = sorted([y_pos, self.pixelClicked[1]])
            x_start = max(x_start, 0)  # make sure we're still inside the image
            y_start = max(y_start, 0)
            x_end = min(x_end, self.beammap.ncols - 1)
            y_end = min(y_end, self.beammap.nrows - 1)

            newPixels = set((x, y) for x in range(x_start, x_end + 1) for y in range(y_start, y_end + 1))
            self.selectedPixels = self.selectedPixels | newPixels  # Union of set so there are no repeats
            try:
                self.grPixMap.scene().removeItem(self.movingBox)
                self.movingBox = None
            except Exception:
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
        x_end = min(x_end, self.beammap.ncols * scale)
        y_end = min(y_end, self.beammap.nrows * scale)
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
            numImages2Sum = 1

        cf=CalFactory('sum', images=self.imageList[-numImages2Sum:], dark=self.darkField if applyDark else None)
        im = cf.generate(name='pixelcount')
        pixelList = np.asarray(pixelList)
        return im.data[(pixelList[:, 1], pixelList[:, 0])].sum()

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
            self.label_pixelInfo.setText('({p[0]}, {p[1]}) : {v:.2f} #/s'.format(p=self.pixelCurrent, v=val))

            bm = self.beammap
            resID = 0
            freq = 0
            feedline = 0
            board = 'a'
            freqCh = 0
            try:
                resIDs = bm.resIDat(*self.pixelCurrent)
                if len(resIDs) > 1:
                    getLogger('Dashboard').warning('Multiple ResIDs for pixel {},{}. Using first.'.format(*self.pixelCurrent))
                resID = resIDs[0]
                for roach in self.roachList:
                    freqFN = roach.tagfile(self.config.roaches.get('r{}.freqfileroot'.format(roach.num)),
                                           dir=self.config.paths.setup)
                    sd = sweepdata.SweepMetadata(file=freqFN)
                    resIDs, freqs, _, _, _ = sd.templar_data(self.config.roaches.get('r{}.lo_freq'.format(roach.num)))
                    try:
                        freqCh = np.where(resIDs == resID)[0][0]
                        freq = freqs[freqCh]
                        feedline = self.config.roaches.get('r{}.feedline'.format(roach.num))
                        board = self.config.roaches.get('r{}.range'.format(roach.num))
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
        window.closeWindow.connect(partial(self.timeStreamWindows.remove, window))  # remove from list if closed
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
            self.logstate()
            self.observing = True
            self.button_obs.setEnabled(False)
            self.button_obs.clicked.disconnect()
            self.textbox_target.setReadOnly(True)
            self.turnOnPhotonCapture()  # NB this does NOT also need to be in stop obs, roaches still send
            if not (self.takingDark or self.takingFlat):
                self.sciFactory = CalFactory('sum', dark=self.darkField, flat=self.flatField)
            self.packetmaster.startWriting(self.config.paths.data)
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
            self.logstate()
            self.observing = False
            self.button_obs.setEnabled(False)
            self.button_obs.clicked.disconnect()
            self.packetmaster.stopWriting()
            getLogger('Dashboard').info("Stop Obs")
            if self.sciFactory is not None:
                self.sciFactory.generate(threaded=True,
                                         fname=os.path.join(self.config.paths.data,
                                                            't{}.fits'.format(int(time.time()))),
                                         name=str(self.textbox_target.text()), save=True)
            else:
                getLogger('Dashboard').critical('sciFactory is None')
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

    def setFilter(self, filter_index=None, instrument='mec'):
        error = False
        if self.config.instrument.lower() != 'bluefors':
            if filter_index is None:
                result = mkidreadout.hardware.hsfw.getfilter(self.config.filter.ip)
                if str(result).lower().startswith('error'):
                    error = True
                else:
                    self.filter=int(result)
                    filternames = mkidreadout.hardware.hsfw.getfilternames(self.config.filter.ip)
                    filternames = filternames.split(', ')
                    self.combobox_filter.clear()
                    self.combobox_filter.addItems(filternames)
                    self.combobox_filter.setCurrentIndex(self.filter - 1)
            else:
                if str(self.combobox_filter.itemText(filter_index)).startswith('Connect'):
                    return self.setFilter(None)
                elif str(self.combobox_filter.itemText(filter_index)).startswith('Error'):
                    return
                else:
                    result = mkidreadout.hardware.hsfw.setfilter(filter_index+1, home=False,host=self.config.filter.ip)
                    if str(result).lower().startswith('error'):
                        error = True
                    else:
                        self.filter = int(result)
            if error:
                self.filter = 'UNKNOWN'
                self.combobox_filter.clear()
                self.combobox_filter.addItems(['Connect', 'Error'])
                self.combobox_filter.setCurrentIndex(1)
        else:
            self.filter = 0

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
                self.logstate()
                getLogger('Dashboard').info('Starting a {} laser cal for {} s.'.format(laserCalStyle, laserTime))
                QtCore.QTimer.singleShot(laserTime * 1000, lcaldone)
            else:
                getLogger('Dashboard').error('Failed to start  a {} laser cal. Lasers may be on.'.format(laserCalStyle))
                self.checkbox_flipper.setEnabled(True)

    def state(self):
        """this is the function that populates the headers and the log, it needs to be prompt enough that it won't
        cause slowdowns

        Do not use
        utcstart, exptime, wmin, wmax they would lead to overwriting photonimage keys

        """
        targ, cmt = str(self.textbox_target.text()), str(self.textbox_log.toPlainText())
        telescope_state = self.telescopeController.get_header()
        now = datetime.utcnow()
        state = dict(target=targ, laser=self.laserController.status,
                     flipper='image', filter=self.filter, observatory=self.telescopeController.observatory,
                     instrument=self.config.instrument,
                     dither_home=tuple(self.config.dashboard.dither_home),
                     dither_ref=tuple(self.config.dashboard.dither_ref),
                     dither_pos=self.dither_dialog.status['pos'] if self.dither_dialog is not None else None,
                     platescale=self.config.dashboard.platescale,
                     device_orientation=self.config.dashboard.device_orientation,
                     utc_readable=now.strftime("%Y%m%d%H%M%S"), utc=now.strftime("%Y%m%d%H%M%S"), comment=cmt)
        for key in telescope_state:
            k = key.lower()
            state[k if k not in state else 'tel_'+k] = telescope_state[key]
        return state

    def logstate(self):
        state = self.state()
        state.pop('utc_readable')
        getLogger('ObsLog').info(json.dumps(state))

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
        getCurrentTime.timeout.connect(lambda: label_currentTime.setText("UTC: " + str(time.time())))
        getCurrentTime.start()

        # Mkid data directory
        label_dataDir = QLabel('Data Dir:')
        dataDir = self.config.paths.data
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
        if self.dither_dialog is not None:
            button_dither = QPushButton("Dithers")
            button_dither.clicked.connect(lambda: self.dither_dialog.show())
        else:
            button_dither = QPushButton("Dithers")

        # Filter
        label_filter = QLabel("Filter:")
        self.combobox_filter = combobox_filter = QComboBox()
        combobox_filter.setToolTip("Select a filter!")
        combobox_filter.activated[int].connect(self.setFilter)

        # log file
        label_target = QLabel("Target: ")
        self.textbox_target = QLineEdit()
        self.textbox_log = QTextEdit()
        # autoLogTimer = QtCore.QTimer(self)
        # autoLogTimer.setInterval(5 * 1000)  # milliseconds
        # autoLogTimer.timeout.connect(self.logstate)
        # autoLogTimer.start()

        # ==================================
        # Image settings!
        # integration Time
        integrationTime = self.config.dashboard.inttime
        label_integrationTime = QLabel('Integrate ')
        self.spinbox_integrationTime = QDoubleSpinBox()
        self.spinbox_integrationTime.setRange(self.config.dashboard.mininttime, self.config.dashboard.maxinttime)
        self.spinbox_integrationTime.setSingleStep(self.config.dashboard.mininttime)
        self.spinbox_integrationTime.setDecimals(1)
        self.spinbox_integrationTime.setValue(integrationTime)
        self.spinbox_integrationTime.setSuffix(' s')
        self.spinbox_integrationTime.setWrapping(False)
        self.spinbox_integrationTime.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        self.spinbox_integrationTime.valueChanged.connect(partial(self.config.update, 'dashboard.inttime'))
        #NB this is a hack and I've not clue why it is needed
        self.spinbox_integrationTime.valueChanged.connect(lambda x: self.imageFetcher.update_inttime(x))

        # dark Image
        button_darkImage = QPushButton('Take Dark')

        def takeDark():
            self.darkField = None
            # self.spinbox_minLambda.setValue(0)
            # self.spinbox_maxLambda.setValue(10000)
            self.spinbox_minLambda.setDisabled(True)
            self.spinbox_maxLambda.setDisabled(True)
            self.spinbox_integrationTime.setDisabled(True)
            self.takingDark = True

        button_darkImage.clicked.connect(takeDark)
        self.checkbox_darkImage = QCheckBox()
        self.checkbox_darkImage.setChecked(False)

        # config file
        button_flatImage = QPushButton('Take Flat')

        def takeFlat():
            self.flatField = None
            # self.spinbox_minLambda.setValue(0)
            # self.spinbox_maxLambda.setValue(10000)
            self.spinbox_minLambda.setDisabled(True)
            self.spinbox_maxLambda.setDisabled(True)
            self.spinbox_integrationTime.setDisabled(True)
            self.takingFlat = True

        button_flatImage.clicked.connect(takeFlat)
        self.checkbox_flatImage = QCheckBox()
        self.checkbox_flatImage.setChecked(False)

        # maxCountRate
        maxCountRate = self.config.dashboard.max_count_rate
        minCountRate = self.config.dashboard.min_count_rate
        label_maxCountRate = QLabel('max:')
        spinbox_maxCountRate = QSpinBox()
        spinbox_maxCountRate.setRange(minCountRate, self.config.dashboard.max_count_rate)
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
        spinbox_minCountRate.valueChanged.connect(partial(self.config.update, 'dashboard.min_count_rate'))  # change cfg
        spinbox_maxCountRate.valueChanged.connect(partial(self.config.update, 'dashboard.max_count_rate'))  # change cfg
        # make sure min is always less than max
        spinbox_minCountRate.valueChanged.connect(spinbox_maxCountRate.setMinimum)
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

        self.checkbox_usewave = QCheckBox('Apply wavecal')
        if not os.path.exists(self.config.dashboard.wavecal):
            self.config.dashboard.update('use_wave', False)
            self.checkbox_usewave.setDisabled(True)
            self.checkbox_usewave.setChecked(False)
        else:
            self.checkbox_usewave.setChecked(self.config.dashboard.use_wave)
        self.liveimage.useWvl = self.config.dashboard.use_wave
        self.checkbox_usewave.stateChanged.connect(lambda: self.liveimage.set_useWvl(not self.liveimage.useWvl))

        #wavelength bounds
        label_lambdaRange = QLabel('<font face="Symbol"><font size="+1">l</font></font><sub>-</sub> - '
                                   '<font size="+1">l</font></font><sub>+</sub>') #QChar(0xBB, 0x03))
        self.spinbox_maxLambda = spinbox_maxLambda = QSpinBox()
        spinbox_maxLambda.setRange(0, 10000)
        spinbox_maxLambda.setValue(self.config.dashboard.wave_stop)
        spinbox_maxLambda.setSuffix(' nm')
        spinbox_maxLambda.setWrapping(False)
        spinbox_maxLambda.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        spinbox_maxLambda.valueChanged.connect(lambda x: self.checkbox_darkImage.setChecked(False))
        spinbox_maxLambda.valueChanged.connect(lambda x: self.checkbox_flatImage.setChecked(False))

        self.spinbox_minLambda = spinbox_minLambda = QSpinBox()
        spinbox_minLambda.setRange(0, 10000)
        spinbox_minLambda.setValue(self.config.dashboard.wave_start)
        spinbox_minLambda.setSuffix(' nm')
        spinbox_minLambda.setWrapping(False)
        spinbox_minLambda.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)

        # make sure min is always less than max
        spinbox_minLambda.valueChanged.connect(spinbox_maxLambda.setMinimum)
        spinbox_maxLambda.valueChanged.connect(spinbox_minLambda.setMaximum)
        spinbox_minLambda.valueChanged.connect(self.liveimage.set_wvlStart)
        spinbox_maxLambda.valueChanged.connect(self.liveimage.set_wvlStop)
        spinbox_minLambda.valueChanged.connect(lambda x: self.checkbox_darkImage.setChecked(False))
        spinbox_minLambda.valueChanged.connect(lambda x: self.checkbox_flatImage.setChecked(False))
        # don't bother remaking the current image because it requires taking a new one for a
        # difference to be seen

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
        vbox.addLayout(build_hbox((label_dataDir, textbox_dataDir)))
        vbox.addWidget(self.button_obs)
        vbox.addLayout(build_hbox((label_target, self.textbox_target)))
        vbox.addWidget(self.textbox_log)

        hbox_filter = QHBoxLayout()
        hbox_filter.addWidget(button_dither)
        hbox_filter.addStretch()
        hbox_filter.addWidget(label_filter)
        hbox_filter.addWidget(combobox_filter)
        vbox.addLayout(hbox_filter)

        vbox.addStretch()

        vbox.addLayout(build_hbox((label_integrationTime, self.spinbox_integrationTime)))
        vbox.addLayout(build_hbox((button_darkImage, QLabel('Use'), self.checkbox_darkImage)))
        vbox.addLayout(build_hbox((button_flatImage, QLabel('Use'), self.checkbox_flatImage)))

        hbox_CountRate = QHBoxLayout()
        hbox_CountRate.addWidget(label_minCountRate)
        hbox_CountRate.addWidget(spinbox_minCountRate)
        hbox_CountRate.addSpacing(20)
        hbox_CountRate.addWidget(label_maxCountRate)
        hbox_CountRate.addWidget(spinbox_maxCountRate)
        hbox_CountRate.addStretch()
        vbox.addLayout(hbox_CountRate)

        vbox.addLayout(build_hbox((label_stretch, self.combobox_stretch)))

        vbox.addWidget(self.checkbox_usewave)
        vbox.addLayout(build_hbox((label_lambdaRange, self.spinbox_minLambda, self.spinbox_maxLambda)))

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
        quit_action = self.create_action("&Quit", slot=self.close, shortcut="Ctrl+Q", tip="Close the application")
        add_actions(self.file_menu, (telescope_action, None, quit_action))
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

        self.turnOffPhotonCapture()  # stop sending photon packets

        self.workers[0].search = False  # stop searching for new images
        del self.grPixMap  # Get segfault if we don't delete this. Something about signals in the queue trying to access deleted objects...
        for thread in self.threadPool:  # They should all be done at this point, but just in case
            thread.quit()
        for window in self.timeStreamWindows:
            window.close()
        for window in self.histogramWindows:
            window.close()
        self.telescopeWindow._want_to_close = True
        self.telescopeWindow.close()

        if self.dither_dialog is not None:
            self.dither_dialog._want_to_close=True
            self.dither_dialog.close()

        self.hide()
        time.sleep(0.2) #wait a bit so everything finishes exiting nicely
        self.packetmaster.quit()

        QtCore.QCoreApplication.instance().quit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MKID Dashboard')
    parser.add_argument('-a', action='store_true', default=False, dest='all_roaches',
                        help='Run with all roaches for instrument in cfg')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--alla', action='store_true', help='Run with all range A roaches in config')
    group.add_argument('--allb', action='store_true', help='Run with all range B roaches in config')

    parser.add_argument('-r', nargs='+', type=int, help='Roach numbers', dest='roaches')
    parser.add_argument('-c', '--config', default=mkidreadout.config.DEFAULT_DASHBOARD_CFGFILE, dest='config',
                        type=str, help='The config file')
    parser.add_argument('-o', '--offline', default=False, dest='offline', action='store_true', help='Run offline')
    parser.add_argument('--gencfg', default=False, dest='genconfig', action='store_true',
                        help='generate configs in CWD')

    args = parser.parse_args()

    if args.genconfig:
        mkidreadout.config.generate_default_configs(dashboard=True)
        exit(0)

    config = mkidreadout.config.load(args.config)

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
    create_log('ObsLog',
               logfile=os.path.join(config.paths.logs, 'obslog_{}.json'.format(timestamp)),
               console=False, mpsafe=True, propagate=False,
               fmt='%(message)s',
               level=mkidcore.corelog.DEBUG)
    create_log('dither',
               logfile=os.path.join(config.paths.logs, 'dither_{}.log'.format(timestamp)),
               console=False, mpsafe=True, propagate=False,
               fmt='%(asctime)s %(message)s',
               level=mkidcore.corelog.DEBUG)
    create_log('Dashboard',
               logfile=os.path.join(config.paths.logs, 'dashboard_{}.log'.format(timestamp)),
               console=True, mpsafe=True, propagate=False,
               fmt='%(asctime)s Dashboard %(levelname)s: %(message)s',
               level=mkidcore.corelog.DEBUG)
    create_log('mkidreadout',
               console=True, mpsafe=True, propagate=False,
               fmt='%(asctime)s %(funcName)s: %(levelname)s %(message)s',
               level=mkidcore.corelog.DEBUG)
    create_log('mkidcore',
               console=True, mpsafe=True, propagate=False,
               fmt='%(asctime)s mkidcore.x.%(funcName)s: %(levelname)s %(message)s',
               level='INFO')
    create_log('packetmaster',
               logfile=os.path.join(config.paths.logs, 'packetmaster_{}.log'.format(timestamp)),
               console=True, mpsafe=True, propagate=False,
               fmt='%(asctime)s Packetmaster: %(levelname)s %(message)s',
               level=mkidcore.corelog.DEBUG)

    app = QApplication(sys.argv)
    if args.alla:
        roaches = mkidcore.instruments.ROACHESA[config.instrument]
    elif args.allb:
        roaches = mkidcore.instruments.ROACHESB[config.instrument]
    elif args.all_roaches:
        roaches = mkidcore.instruments.ROACHES[config.instrument]
    else:
        roaches = args.roaches

    if not roaches:
        getLogger('Dashboard').error('No roaches specified')
        exit()
    form = MKIDDashboard(roaches, config=config, offline=args.offline)
    form.show()
    app.exec_()
