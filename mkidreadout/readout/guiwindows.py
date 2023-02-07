"""
Author:    Alex Walter
Date:      Jul 6, 2016

GUI window  classes

"""
import threading
import time

import numpy as np
from PyQt4 import QtCore
from PyQt4.QtGui import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import mkidreadout.hardware.conex as conex
from mkidcore.corelog import getLogger

try:
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
except ImportError: #Named changed in some newer matplotlib versions
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

class TelescopeWindow(QMainWindow):
    def __init__(self, telescope, parent=None):
        """
        INPUTES:
            telescope - Telescope object
        """
        super(QMainWindow, self).__init__(parent)
        self.setWindowTitle(telescope.observatory)
        self._want_to_close = False
        self.telescope = telescope
        self.create_main_frame()
        updater = QtCore.QTimer(self)
        updater.setInterval(1003)
        updater.timeout.connect(self.updateTelescopeInfo)
        updater.start()

    def updateTelescopeInfo(self, target='sky'):
        if not self.isVisible():
            return
        tel_dict = self.telescope.get_telescope_position()
        for key in tel_dict.keys():
            try:
                self.label_dict[key].setText(str(tel_dict[key]))
            except:
                layout = self.main_frame.layout()
                label = QLabel(key)
                label_val = QLabel(str(tel_dict[key]))
                hbox = QHBoxLayout()
                hbox.addWidget(label)
                hbox.addWidget(label_val)
                layout.addLayout(hbox)
                self.main_frame.setLayout(layout)
                self.label_dict[key] = label_val

    def create_main_frame(self):
        self.main_frame = QWidget()
        vbox = QVBoxLayout()

        def add2layout(vbox, *args):
            hbox = QHBoxLayout()
            for arg in args:
                hbox.addWidget(arg)
            vbox.addLayout(hbox)

        label_telescopeStatus = QLabel('Telescope Status')
        font = label_telescopeStatus.font()
        font.setPointSize(24)
        label_telescopeStatus.setFont(font)
        vbox.addWidget(label_telescopeStatus)

        tel_dict = self.telescope.get_telescope_position()
        self.label_dict = {}
        for key in tel_dict.keys():
            label = QLabel(key)
            label.setMaximumWidth(150)
            label_val = QLabel(str(tel_dict[key]))
            label_val.setMaximumWidth(150)
            add2layout(vbox, label, label_val)
            self.label_dict[key] = label_val

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def closeEvent(self, event):
        if self._want_to_close:
            self.close()
        else:
            self.hide()


class ValidDoubleTuple(QValidator):
    def __init__(self, parent=None):
        QValidator.__init__(self, parent=parent)
        # super(ValidDoubleTuple, self).__init__(self, parent=parent)

    def validate(self, s, pos):
        try:
            assert len(map(float, s.split(','))) == 2
        except (AssertionError, ValueError):
            return QValidator.Invalid, pos
        return QValidator.Acceptable, pos


class DitherWindow(QMainWindow):
    complete = QtCore.pyqtSignal(object)
    statusupdate = QtCore.pyqtSignal()

    def __init__(self, conexaddress, parent=None):
        """
        Window for gathing dither info
        """
        QMainWindow.__init__(self, parent=parent)
        self.address = conexaddress
        self.status = conex.status(self.address)
        self._want_to_close = False
        self.thread_pool=np.asarray([])
        self._rlock =threading.RLock()

        self.create_main_frame()

    def create_main_frame(self):
        self.setWindowTitle('Dither')

        main_frame = QWidget()
        vbox = QVBoxLayout()
        doubletuple_validator = ValidDoubleTuple()
        def add2layout(*args):
            hbox = QHBoxLayout()
            for arg in args:
                hbox.addWidget(arg)
            vbox.addLayout(hbox)

        label_dither = QLabel('Dither Control')
        font = label_dither.font()
        font.setPointSize(18)
        label_dither.setFont(font)
        add2layout(label_dither)

        label_start= QLabel('Start Position (x,y):')
        self.textbox_start = QLineEdit('0.0, 0.0')
        self.textbox_start.setValidator(doubletuple_validator)
        add2layout(label_start, self.textbox_start)

        label_end = QLabel('End Position (x,y):')
        self.textbox_end = QLineEdit('0.0, 0.0')
        self.textbox_end.setValidator(doubletuple_validator)
        add2layout(label_end, self.textbox_end)

        label_nsteps = QLabel('N Steps:')
        self.textbox_nsteps = QLineEdit('5')
        self.textbox_nsteps.setValidator(QIntValidator(bottom=1))
        add2layout(label_nsteps, self.textbox_nsteps)

        label_dwell = QLabel('Dwell Time (s):')
        self.textbox_dwell = QLineEdit('30')
        self.textbox_dwell.setValidator(QDoubleValidator(bottom=0))
        add2layout(label_dwell, self.textbox_dwell)

        label_sub = QLabel('Sub Dither:  jog:')
        self.textbox_sub = QLineEdit('0')
        self.textbox_sub.setValidator(QDoubleValidator(bottom=0))
        label_subT = QLabel('dwell (s):')
        self.textbox_subT = QLineEdit('0')
        self.textbox_subT.setValidator(QDoubleValidator(bottom=0))
        add2layout(label_sub, self.textbox_sub,label_subT, self.textbox_subT)

        button_dither = QPushButton('Dither')
        button_dither.clicked.connect(self.do_dither)
        add2layout(button_dither)

        label_pos = QLabel('Position (x,y):')
        self.textbox_pos = QLineEdit('0.0, 0.0')
        self.textbox_pos.setValidator(doubletuple_validator)
        try: self.textbox_pos.setText('{}, {}'.format(self.status['pos'][0], self.status['pos'][1]))
        except: pass
        button_goto = QPushButton('Go')
        button_goto.clicked.connect(self.do_goto)
        add2layout(label_pos, self.textbox_pos, button_goto)

        self.status_label = QLabel('Status')
        self.status_label.setText(self.status['state'][0]+'-->'+self.status['state'][1])
        self.statusupdate.connect(lambda: self.status_label.setText(self.status['state'][0]+'-->'+self.status['state'][1]))
        add2layout(self.status_label)

        button_halt = QPushButton('STOP')
        button_halt.clicked.connect(self.do_halt)
        add2layout(button_halt)

        main_frame.setLayout(vbox)
        self.setCentralWidget(main_frame)

    def do_dither(self):
        start = map(float, self.textbox_start.text().split(','))
        end = map(float, self.textbox_end.text().split(','))
        ns = int(self.textbox_nsteps.text())
        dt = float(self.textbox_dwell.text())
        jog = float(self.textbox_sub.text())
        subT = float(self.textbox_subT.text())
        getLogger('Dashboard').info('Starting dither')

        dither_dict = {'startx': start[0],
                       'endx': end[0],
                       'starty': start[1],
                       'endy': end[1],
                       'n':ns,
                       't':dt,
                       'subStep':jog,
                       'subT':subT}
        started=conex.dither(dither_dict, address=self.address)
        if started:
            self.thread_pool = self.thread_pool[[t.is_alive() for t in self.thread_pool]]
            thread = threading.Thread(target=self._wait4dither, name="Dithering wait thread")
            thread.daemon = True
            self.thread_pool=np.append(self.thread_pool,thread)
            thread.start()

    def _wait4dither(self):
        d = conex.queryDither(address=self.address)
        with self._rlock:
            self.status = d['status']
        self.statusupdate.emit()
        pos_tolerance = 0.003
        while not d['completed']:
            time.sleep(0.0001)
            try:
                d = conex.queryDither(address=self.address)

                oldPos = self.status['pos']
                newPos = d['status']['pos']
                posNear = (np.abs(newPos[0] - oldPos[0]) <= pos_tolerance) and (np.abs(newPos[1] - oldPos[1]) <=pos_tolerance)
                with self._rlock:
                    self.status = d['status']
                if not posNear:  #If the position changed
                    self.statusupdate.emit()
            except:
                d = {'completed': False}
        self.complete.emit(d)
        getLogger('Dashboard').info('Finished dither')

    def do_halt(self):
        getLogger('Dashboard').info('Conex Movement Stopped by user.')
        s=conex.stop(address=self.address, timeout=None)   #blocking
        with self._rlock:
            self.status = s
        self.statusupdate.emit()

    def do_goto(self, pos=None):
        try:
            x=pos[0]
            y=pos[1]
            self.textbox_pos.setText('{}, {}'.format(x,y))
        except:
            x, y = map(float, self.textbox_pos.text().split(','))
        getLogger('Dashboard').info('Starting move to {:.2f}, {:.2f}'.format(x,y))
        started = conex.move(x, y, address=self.address)
        if started:
            self.thread_pool = self.thread_pool[[t.is_alive() for t in self.thread_pool]]
            thread = threading.Thread(target=self._wait4move, name="Move wait thread")
            thread.daemon = True
            self.thread_pool=np.append(self.thread_pool,thread)
            thread.start()

    def _wait4move(self):
        d= conex.queryMove(address=self.address)
        self.status = d['status']
        pos_tolerance = 0.003
        while not d['completed']:
            time.sleep(0.0001)
            try:
                d= conex.queryMove(address=self.address)
                self.status = d['status']
            except:
                d={'completed':False}
        self.statusupdate.emit()
        getLogger('Dashboard').info('Finished conex GOTO')

    def closeEvent(self, event):
        if self._want_to_close:
            conex.stop(address=self.address, timeout=1.)
            for t in self.thread_pool:
                t.join(timeout=1.)
            event.accept()
            self.close()
        else:
            event.ignore()
            self.hide()


class PixelTimestreamWindow(QMainWindow):
    """ We use this for realtime plotting with MKIDDashboard"""
    closeWindow = QtCore.pyqtSignal()

    def __init__(self, pixelList, parent):
        super(QMainWindow, self).__init__(parent)
        self.setWindowTitle("Timestream")
        self.parent = parent
        self.pixelList = np.asarray(pixelList, dtype=np.int)
        self.create_main_frame()
        self.cur_line = None
        self.pix_line = None
        self.plotData()
        parent.newImageProcessed.connect(self.plotData)

    def plotData(self, **kwargs):
        self.setCheckboxText(currentPix=True)

        oldx_lim = self.ax.get_xlim()
        oldy_lim = self.ax.get_ylim()

        if self.checkbox_plotPix.isChecked():
            times1, countRate = self.getCountRate()
            if self.pix_line is None:
                self.pix_line = self.ax.plot(times1, countRate, 'g-')[0]
            else:
                self.pix_line.set_data(times1, countRate)
        elif self.checkbox_plotPix is not None:
            self.pix_line.set_data([], [])
        if self.checkbox_plotCurrentPix.isChecked():
            times2, countRate_cur = self.getCountRate(forCurrentPix=True)
            if self.cur_line is None:
                self.cur_line = self.ax.plot(times2, countRate_cur, 'c-')[0]
            else:
                self.cur_line.set_data(times2, countRate_cur)
        elif self.cur_line is not None:
            self.cur_line.set_data([], [])

        # self.ax.cla()

        if self.mpl_toolbar._active is None:
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
        else:
            self.ax.set_xlim(oldx_lim)
            self.ax.set_ylim(oldy_lim)
        self.ax.set_xlabel('Time (s)')

        ytitle = '{}Count Rate ({})'.format('Avg. ' if self.checkbox_normalize.isChecked() else '',
                                            '#/s' if self.checkbox_persec.isChecked() else '#')
        self.ax.set_ylabel(ytitle)
        self.draw()

    def getCountRate(self, forCurrentPix=False):
        imageList = list(self.parent.imageList)  #a list of fits hdu
        pixList = np.asarray([[p[0], p[1]] for p in self.parent.selectedPixels]) if forCurrentPix else self.pixelList

        if len(imageList) == 0 or len(pixList) == 0:
            return []

        x = pixList[:, 0]
        y = pixList[:, 1]
        # c = np.asarray([i.data for i in imageList])[:, y, x]
        c = np.asarray([i.data[y, x] for i in imageList], dtype=float)

        dt = np.array([i.header['exptime'] for i in imageList], dtype=float)
        times = np.cumsum(dt)

        countRate = np.sum(c, axis=1)

        if self.checkbox_normalize.isChecked():
            numZeroPix = (np.sum(c, axis=0) == 0).sum()
            countRate /= max(len(pixList) - numZeroPix, 1)

        if self.checkbox_persec.isChecked():
            countRate /= dt

        return times, countRate

    def addData(self, imageList):
        # countRate = np.sum(np.asarray(image)[self.pixelList[:,1],self.pixelList[:,0]])
        # self.countTimestream = np.append(self.countTimestream,countRate)
        self.plotData()

    def draw(self):
        self.canvas.draw()
        self.canvas.flush_events()

    def setCheckboxText(self, currentPix=False):
        pixList = self.pixelList
        checkbox = self.checkbox_plotPix
        label = "Pixels: "
        if currentPix:
            pixList = np.asarray([[p[0], p[1]] for p in self.parent.selectedPixels])
            checkbox = self.checkbox_plotCurrentPix
            label = "Current Pixels: "

        label = label + ('{}, ' * len(pixList)).format(*pixList)[:-2]
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
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Count Rate (#/s)')

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

        self.checkbox_persec = QCheckBox('Use units of counts per second')
        self.checkbox_persec.setChecked(True)
        self.checkbox_persec.stateChanged.connect(lambda x: self.plotData())

        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        vbox_plot.addWidget(self.mpl_toolbar)

        vbox_plot.addWidget(self.checkbox_plotPix)
        vbox_plot.addWidget(self.checkbox_plotCurrentPix)
        vbox_plot.addWidget(self.checkbox_normalize)
        vbox_plot.addWidget(self.checkbox_persec)

        self.main_frame.setLayout(vbox_plot)
        self.setCentralWidget(self.main_frame)

    def closeEvent(self, event):
        self.parent.newImageProcessed.disconnect(self.plotData)
        self.closeWindow.emit()
        event.accept()


class PixelHistogramWindow(QMainWindow):
    """
    We use this for realtime plotting with MKIDDashboard
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

    # def plotData(self, **kwargs):
    #     image = self.parent.imageList[-1]
    #     countsX = np.sum(image,axis=0)
    #     countsY = np.sum(image,axis=1)
    #     self.line.set_data(image.shape[0],countsX)
    #     self.line2.set_data(image.shape[1],countsY)

    def getCountRateHist(self, forCurrentPix=False):
        imageList = self.parent.imageList
        pixList = self.pixelList
        if forCurrentPix:
            pixList = np.asarray([[p[0], p[1]] for p in self.parent.selectedPixels])
        if len(imageList) and len(pixList):
            x = pixList[:, 0]
            y = pixList[:, 1]
            c = np.asarray(imageList[-1].data)[y,x]
            #countRates = np.sum(c,axis=0)
            #if self.checkbox_normalize.isChecked():
            #    countRates/=len(pixList)
            countRateHist, bin_edges = np.histogram(c, bins=50, range=(0, 2500))
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
        self.parent.newImageProcessed.disconnect(self.plotData)
        self.closeWindow.emit()
        event.accept()
