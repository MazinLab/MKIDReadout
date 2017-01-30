

"""
Author: Alex Walter
Date: May 18, 2016

A class for the settings window for HighTemplar.py. 
"""
import numpy as np
from functools import partial
from PyQt4.QtGui import *
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
import ConfigParser
from InitStateMachine import InitStateMachine


class InitSettingsWindow(QTabWidget):
    """
    
    SIGNALS
        resetRoach - Signal emmited when we change a setting so we can reload it into the ROACH2
        initTemplar - Signal emmited when we want to trick templar into initializing into a specific state
        nBitsRemovedInFFT
        
        
        The first parameter is the roach number
        The second parameter is data
    """
    
    resetRoach = QtCore.pyqtSignal(int,int)     
    initTemplar = QtCore.pyqtSignal(int,object)
    # nBitsRemovedInFFT = QtCore.pyqtSignal(int)
    
    def __init__(self,roachNums,config,parent=None):
        """
        Creates settings window GUI
        
        INPUTS:
            roachNums - list of roaches and their numbers
            defaultSettings - ConfigParser object that contains all the default values
            parent - the HighTemplar QMainWindow
        """
        super(InitSettingsWindow, self).__init__(parent)
        self.setWindowTitle('Settings')
        #self.setTabBar(HorizontalTabWidget())
        #self.setTabBar(FingerTabWidget(parent=self,width=100,height=25))
        self.setUsesScrollButtons(False)
        self._want_to_close = False
        self.roachNums = roachNums
        self.config = config
        
        for roachNum in self.roachNums:
            tab = InitSettingsTab(roachNum, self.config)
            tab.resetRoach.connect(partial(self.resetRoach.emit,roachNum))
            tab.initTemplar.connect(partial(self.initTemplar.emit,roachNum))
            #tab.nBitsRemovedInFFT.connect(partial(self.nBitsRemovedInFFT.emit,roachNum))
            self.addTab(tab, ''+str(roachNum))
        
        #self.setMovable(True)
        self.setTabsClosable(False)
        #self.resize(self.minimumSizeHint())
        #self.resize(300,1000)
        self.setTabPosition(QTabWidget.West)
        #self.setStyleSheet('QTabBar::tab {color: red;}')
        #self.setStyleSheet('QTabBar::tab {selection-color: red;}')
        
        #self.printColors()

    
    def finishedInitV7(self, roachNum):
        tabArg = np.where(np.asarray(self.roachNums) == roachNum)[0][0]
        self.widget(tabArg).checkbox_waitV7.setChecked(False)
        #self.config.set('Roach '+str(self.num),'waitForV7Ready',False)
    
    def closeEvent(self, event):
        if self._want_to_close:
            self.close()
        else:
            self.hide()
            


class InitSettingsTab(QMainWindow):
    resetRoach = QtCore.pyqtSignal(int)     #Signal emmited when we change a setting so we can reload it into the ROACH2
    initTemplar = QtCore.pyqtSignal(object)
    # nBitsRemovedInFFT = QtCore.pyqtSignal() #signal emitted when we change the number of bits removed before the FFT

    def __init__(self,roachNum,config):
        super(InitSettingsTab, self).__init__() #parent=None. This is the correct way according to the QTabWidget documentation
        self.roachNum = roachNum
        self.config = config
        
        self.create_main_frame()
    
    # def changedNBitsRemoved(self, nBitsRemoved):
    #     self.changedSetting('nBitsRemovedInFFT',nBitsRemoved)
    #     #self.nBitsRemovedInFFT.emit()

    def changedSetting(self,settingID,setting):
        """
        When a setting is changed, reflect the change in the config object which is shared across all GUI elements.
        
        INPUTS:
            settingID - the key in the configparser
            setting - the value
        """
        self.config.set('Roach '+str(self.roachNum),settingID,str(setting))
        #If we don't force the setting value to be a string then the configparser has trouble grabbing the value later on for some unknown reason
        
    def create_main_frame(self):
        """
        Makes everything on the tab page
        
        
        """
        self.main_frame = QWidget()
        vbox = QVBoxLayout()
       
        def add2layout(vbox, *args):
            hbox = QHBoxLayout()
            for arg in args:
                hbox.addWidget(arg)
            vbox.addLayout(hbox)
        
        label_roachNum = QLabel('Settings for roach '+str(self.roachNum))
        add2layout(vbox,label_roachNum)
        
        ipAddress = self.config.get('Roach '+str(self.roachNum),'ipAddress')
        self.label_ipAddress = QLabel('ip address: ')
        self.label_ipAddress.setMinimumWidth(110)
        self.textbox_ipAddress = QLineEdit(ipAddress)
        self.textbox_ipAddress.setMinimumWidth(70)
        self.textbox_ipAddress.textChanged.connect(partial(self.changedSetting,'ipAddress'))
        self.textbox_ipAddress.textChanged.connect(lambda x: self.resetRoach.emit(-1))      # reset roach state if ipAddress changes
        add2layout(vbox,self.label_ipAddress,self.textbox_ipAddress)


        fpgPath = self.config.get('Roach '+str(self.roachNum),'fpgPath')
        label_fpgPath = QLabel('fpgPath: ')
        label_fpgPath.setMinimumWidth(110)
        textbox_fpgPath = QLineEdit(fpgPath)
        textbox_fpgPath.setMinimumWidth(70)
        textbox_fpgPath.textChanged.connect(partial(self.changedSetting,'fpgPath'))
        textbox_fpgPath.textChanged.connect(lambda x: self.resetRoach.emit(InitStateMachine.PROGRAM_V6))      # reset roach state if ipAddress changes
        add2layout(vbox, label_fpgPath, textbox_fpgPath)

        self.checkbox_waitV7 = QCheckBox('waitForV7Ready')
        self.checkbox_waitV7.setChecked(True)
        self.checkbox_waitV7.stateChanged.connect(lambda x: self.changedSetting('waitForV7Ready', self.checkbox_waitV7.isChecked()))
        add2layout(vbox,self.checkbox_waitV7)

        FPGAParamFile = self.config.get('Roach '+str(self.roachNum),'FPGAParamFile')
        label_FPGAParamFile = QLabel('FPGAParamFile: ')
        label_FPGAParamFile.setMinimumWidth(110)
        textbox_FPGAParamFile = QLineEdit(FPGAParamFile)
        textbox_FPGAParamFile.setMinimumWidth(70)
        textbox_FPGAParamFile.textChanged.connect(partial(self.changedSetting,'FPGAParamFile'))
        textbox_FPGAParamFile.textChanged.connect(lambda x: self.resetRoach.emit(-1))      # reset roach state if ipAddress changes
        add2layout(vbox, label_FPGAParamFile, textbox_FPGAParamFile)
        
        # nBitsRemovedInFFT = self.config.getint('Roach '+str(self.roachNum),'nBitsRemovedInFFT')
        # label_nBitsRemovedInFFT = QLabel('nBitsRemovedInFFT:')
        # label_nBitsRemovedInFFT.setMinimumWidth(110)
        # spinbox_nBitsRemovedInFFT = QSpinBox()
        # spinbox_nBitsRemovedInFFT.setRange(0,12)
        # spinbox_nBitsRemovedInFFT.setValue(nBitsRemovedInFFT)
        # spinbox_nBitsRemovedInFFT.valueChanged.connect(self.changedNBitsRemoved)
        # add2layout(vbox, label_nBitsRemovedInFFT, spinbox_nBitsRemovedInFFT)
        
        

        label_note = QLabel("NOTE: Changing the ip address won't take effect until you re-connect.")
        label_note.setWordWrap(True)
        
        vbox.addWidget(label_note)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)
        
        





#class HorizontalTabWidget(QtGui.QTabBar):
#    def paintEvent(self, event):
#        for index in range(self.count()):
#            painter = QtGui.QPainter()
#            painter.begin(self)
#            painter.setPen(QtCore.Qt.blue);
#            painter.setFont(QtGui.QFont("Arial", 10));
#            tabRect = self.tabRect(index)
#            painter.drawText(tabRect, QtCore.Qt.AlignVCenter | QtCore.Qt.TextDontClip, self.tabText(index));
#            painter.end()
#    
#    def sizeHint(self):
#        return QtCore.QSize(60, 130)
class FingerTabWidget(QtGui.QTabBar):
    """
    This class should extend QTabBar and overload the paintEvent() so that the tabs are on the left and text is horizontal
    
    I can't get the text color to change though...
    """
    def __init__(self, *args, **kwargs):
        self.tabSize = QtCore.QSize(kwargs.pop('width'), kwargs.pop('height'))
        super(FingerTabWidget, self).__init__(*args, **kwargs)
        self.setStyleSheet('QTabBar::tab {color: red;}')
        self.setStyleSheet('QTabBar::tab {selection-color: red;}')

    def paintEvent(self, event):
        painter = QtGui.QStylePainter(self)
        painter.setPen(Qt.red)
        option = QtGui.QStyleOptionTab()

        #painter.begin(self)
        for index in range(self.count()):
            #print ''+str(index)+str(option.palette.color(QPalette.Text).name())
            
            #print self.tabText(index)
            
            
            self.initStyleOption(option, index)
            self.setTabTextColor(index,Qt.red)
            #print option.palette.text().color().name()
            #print painter.brush().color().name()
            #print ''+str(index)+str(option.palette.color(QPalette.Text).name())
            option.palette.setColor(QPalette.Text, Qt.red)
            #print ''+str(index)+str(option.palette.color(QPalette.Text).name())
            option.palette.setColor(QPalette.HighlightedText, Qt.red)
            option.palette.setColor(QPalette.Foreground, Qt.red)
            option.palette.setColor(QPalette.WindowText, Qt.red)
            tabRect = self.tabRect(index)
            tabRect.moveLeft(10)
            painter.drawControl(QtGui.QStyle.CE_TabBarTabShape, option)
            painter.drawControl(QtGui.QStyle.CE_TabBarTabLabel, option)
            painter.drawText(tabRect, QtCore.Qt.AlignVCenter |\
                             QtCore.Qt.TextDontClip, \
                             self.tabText(index));
            #print tabRect
            #painter.drawText(tabRect,0,self.tabText(index))
            print self.tabTextColor(index).name()
            
        #painter.end()
    def tabSizeHint(self,index):
        return self.tabSize








