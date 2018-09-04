"""
Author:    Alex Walter
Date:      May 15, 2016


This is a GUI class to setup the MKID resonators for readout. It replaces the old Channelizer and Templar software from the ARCONS readout.
This file only contains the GUI and control code. The roach commands are implemented in the RoachStateMachine class and communication to the electronic boards is done in Roach2Controls.py.

Command Flow:
 - Create non blocking thread for each roach that just sits in memory
 - When button clicked, load command into command queue in RoachStateMachine object
 - start thread (by calling start()) which executes all commands in queue
 - When thread is done executing the command the RoachStateMachine object emits a singal notifying the GUI and the thread exits its event loop

NOTES:
 - do not add commands to the RoachStateMachine object while the thread is executing commands

Features to add:
 - color buttons yellow on warnings
 - keep log of errors and warnings in txt file
    - add to file menu (help) a viewer for log file
"""
import sys, time, traceback, logging, re
from functools import partial
import numpy as np
import ConfigParser
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
from PyQt4 import QtGui
from PyQt4.QtGui import *

from mkidreadout.channelizer.RoachStateMachine import RoachStateMachine
from mkidreadout.channelizer.RoachSettingsWindow import RoachSettingsWindow
from mkidreadout.channelizer.RoachPlotWindow import RoachPhaseStreamWindow, RoachSweepWindow


class TemplarConfig(object):
    def __init__(self, file=''):
        self.log=logging.getLogger('hightemplar.config')
        self.file = file if file else 'hightemplar.cfg'
        self.log.debug('Loading {}', self.file)
        self.cp = ConfigParser.ConfigParser()
        self.cp.read(self.file)

    @property
    def roaches(self):
        return [s for s in self.cp.sections() if 'roach' in s.lower()]

    def guessRoachFeedlines(self):
        roachmap = {}
        for roach in self.roaches:
            settings='   '.join([x[1] for x in self.cp.items(roach)])
            fl = set(map(lambda x: x.replace('_', '').lower(),
                         re.findall('[fF][lL][0-9]{1,2}_?[ab]', settings, flags=re.IGNORECASE)))
            roachmap[roach] = list(fl)[0] if len(fl) != 1 else ''
            self.log.debug('{} may be for feedline(s) {}, adopting "{}"', roach, fl, roachmap[roach])
        return roachmap


class HighTemplar(QMainWindow):
    def __init__(self, roachNums=None, defaultValues=None):
        """
        Create HighTemplar GUI
        
        INPUTS:
            roachNums - list of roach numbers. ie. [0,2,3,7]
            defaultValues - path to config file. See documentation on ConfigParser
        """
        if roachNums is None or len(roachNums) ==0:
            roachNums = range(10)
        self.roachNums = np.unique(roachNums)       # sorts and removes duplicates
        self.numRoaches = len(self.roachNums)       # (int) number of roaches connected
        self.config = ConfigParser.ConfigParser()
        if defaultValues is None:
            defaultValues = 'hightemplar.cfg'
        self.defaultValues=defaultValues
        self.config.read(defaultValues)
        
        
        #Setup GUI
        super(HighTemplar, self).__init__()
        self.create_main_frame()
        self.setWindowTitle('High Templar Resonator Setup')
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('plastique'))
        #self.create_status_bar()
        
        #Setup Settings window
        self.settingsWindow = RoachSettingsWindow(self.roachNums, self.config, parent=None) # keep parent None for now
        self.settingsWindow.resetRoach.connect(self.resetRoachState)
        self.settingsWindow.setDdsShift.connect(self.setDdsShift)
        self.settingsWindow.initTemplar.connect(self.initTemplar)
        
        #Setup RoachStateMachine and threads for each roach
        self.roaches = []
        self.roachThreads=[]
        for i in self.roachNums:
            roach=RoachStateMachine(i,self.config)
            thread = QtCore.QThread(parent=self)                                        # if parent isn't specified then need to be careful to destroy thread
            thread.setObjectName("Roach_"+str(i))                                       # process name
            roach.finishedCommand_Signal.connect(partial(self.catchRoachSignal,i))      # call catchRoachSignal when roach finishes a command
            roach.commandError_Signal.connect(partial(self.catchRoachError,i))          # call catchRoachError when roach errors out on a command
            thread.started.connect(roach.executeCommands)                               # When the thread is started, automatically call RoachStateMachine.executeCommands()
            roach.finished.connect(thread.quit)                                         # When all the commands are done executing stop the thread. Can be restarted with thread.start()
            roach.moveToThread(thread)                                                  # The roach functions run on the seperate thread
            self.roaches.append(roach)
            self.roachThreads.append(thread)
            #self.destroyed.connect(self.thread.deleteLater)
            #self.destroyed.connect(self.roach.deleteLater)
            roach.reset.connect(partial(self.colorCommandButtons,i))
            
            roach.ddsShift.connect(partial(self.settingsWindow.ddsShiftLoaded,i))
            

        
        #Create plot windows
        self.sweepWindows=[]
        for roach_i in self.roaches:
            window = RoachSweepWindow(roach_i,self.config)
            window.sweepClicked.connect(partial(self.commandButtonClicked, [roach_i.num] , RoachStateMachine.SWEEP))
            #window.fitClicked.connect(partial(self.commandButtonClicked, [roach_i.num] , RoachStateMachine.FIT))
            window.rotateClicked.connect(partial(self.commandButtonClicked, [roach_i.num] , RoachStateMachine.ROTATE))
            window.translateClicked.connect(partial(self.commandButtonClicked, [roach_i.num] , RoachStateMachine.TRANSLATE))
            window.resetRoach.connect(partial(self.resetRoachState, roach_i.num))
            window.adcAttenChanged.connect(partial(self.commandButtonClicked, [roach_i.num]))
            window.dacAttenChanged.connect(partial(self.commandButtonClicked, [roach_i.num]))
            self.sweepWindows.append(window)
        self.phaseWindows=[]
        for roach_i in self.roaches:
            window=RoachPhaseStreamWindow(roach_i,self.config)
            window.thresholdClicked.connect(partial(self.commandButtonClicked, [roach_i.num] , RoachStateMachine.LOADTHRESHOLD))
            #window.phaseSnapClicked.connect(partial(self.commandButtonClicked, [roach_i.num] , RoachStateMachine.LOADTHRESHOLD+1))
            #window.phaseTimestreamClicked.connect(partial(self.commandButtonClicked, [roach_i.num] , RoachStateMachine.LOADTHRESHOLD+1))
            window.phaseSnapClicked.connect(partial(self.commandButtonClicked, [roach_i.num]))          # we don't need to loadthresholds before taking a snapshot
            window.phaseTimestreamClicked.connect(partial(self.commandButtonClicked, [roach_i.num]))
            window.resetRoach.connect(partial(self.resetRoachState, roach_i.num))
            self.phaseWindows.append(window)
            
        self.create_menu()
        
        
        #Initialize by connecting to roaches over ethernet and loading ddsshift
        for i in range(self.numRoaches):
            colorStatus = self.roaches[i].addCommands(RoachStateMachine.CONNECT)        # add command to roach queue
            self.colorCommandButtons(self.roachNums[i],colorStatus)                              # color the command buttons appropriately
            #QtCore.QMetaObject.invokeMethod(roach, 'executeCommands', Qt.QueuedConnection)
            self.roachThreads[i].start()                                                # starting the thread automatically invokes the roach's executeCommand function
        
        # Also auto set DDS shift
        #for i in range(self.numRoaches):
        #    self.setDdsShift(self.roachNums[i])
        
    def test(self,roachNum,state):
        print "Roach "+str(roachNum)+' - '+str(state)

    def catchRoachError(self,roachNum, command, exc_info=None):
        """
        This function is executed when the GUI sees the commandError signal from a RoachThread
        
        INPUTS:
            command - (int) the command that finished
            roachNum - (int) the roach number
        """
        #print e
        traceback.print_exception(*exc_info)
        roachArg = np.where(np.asarray(self.roachNums) == roachNum)[0][0]
        print 'Roach ',roachNum,' errored out: ',RoachStateMachine.parseCommand(command)
        #self.roachBusy[roachArg]=-1
        #self.colorCommandButtons(roaches=[roachNum],commands=[command],color='error')
        colorStatus = [None]*RoachStateMachine.NUMCOMMANDS
        colorStatus[command]='error'
        self.colorCommandButtons(roachNum,colorStatus)

    #def catchRoachSignal(self,command,roachNum,**kwargs):
    def catchRoachSignal(self,roachNum,command,commandData):
        """
        This function is executed when the GUI sees the finishedCommand signal from a RoachThread
        
        INPUTS:
            roachNum - (int) the roach number
            command - (int) the command that finished
            commandData - Data from the command. For Example, after sweep it returns a dictionary of I and Q values
        """
        print "Finished r"+str(roachNum)+' '+RoachStateMachine.parseCommand(command)
        #self.colorCommandButtons(roaches=[roachNum],commands=[command],color='green')
        colorStatus = [None]*RoachStateMachine.NUMCOMMANDS
        colorStatus[command]='green'
        self.colorCommandButtons(roachNum,colorStatus)
        
        roachArg = np.where(np.asarray(self.roachNums) == roachNum)[0][0]
        if command == RoachStateMachine.LOADFREQ:
            self.sweepWindows[roachArg].initFreqs()
            self.phaseWindows[roachArg].initFreqs()
            
        if command == RoachStateMachine.DEFINEROACHLUT:
            self.sweepWindows[roachArg].initFreqs() # initialize LO freq
        if command == RoachStateMachine.DEFINEDACLUT:
            self.sweepWindows[roachArg].initFreqs() # if modified pixel attenuation then need this to remove 'MODIFIED' tag on plot window
            self.sweepWindows[roachArg].updateDACAttenSpinBox(commandData[0])
            self.sweepWindows[roachArg].updateADCAttenSpinBox(commandData[1])
                                                    # Also shows dac quantized freqs
        if command == RoachStateMachine.SWEEP:
            self.sweepWindows[roachArg].plotData(commandData)
        #if command == RoachStateMachine.FIT:
        #    self.sweepWindows[roachArg].plotData(commandData,fit=True)
        if command == RoachStateMachine.ROTATE:
            self.sweepWindows[roachArg].plotData(commandData,rotated=True)
        if command == RoachStateMachine.TRANSLATE:
            self.sweepWindows[roachArg].plotData(commandData)
        if command == RoachStateMachine.LOADTHRESHOLD:
            self.phaseWindows[roachArg].appendThresh(commandData)

    def resetRoachState(self,roachNum,command):
        print "Templar told to reset r"+str(roachNum)+' to '+RoachStateMachine.parseCommand(command)
        roachArg = np.where(np.asarray(self.roachNums) == roachNum)[0][0]
        QtCore.QMetaObject.invokeMethod(self.roaches[roachArg], 'resetStateTo', Qt.QueuedConnection,
                                        QtCore.Q_ARG(int, command))
        self.roachThreads[roachArg].start()
    
    def setDdsShift(self, roachNum, ddsShift=None):
        roachArg = np.where(np.asarray(self.roachNums) == roachNum)[0][0]
        QtCore.QMetaObject.invokeMethod(self.roaches[roachArg], 'loadDdsShift', Qt.QueuedConnection,
                                        QtCore.Q_ARG(int, ddsShift))
        self.roachThreads[roachArg].start()
    
    def initTemplar(self, roachNum, state):
        roachArg = np.where(np.asarray(self.roachNums) == roachNum)[0][0]
        QtCore.QMetaObject.invokeMethod(self.roaches[roachArg], 'initializeToState', Qt.QueuedConnection,
                                        QtCore.Q_ARG(object, state))
        self.roachThreads[roachArg].start()
    
    def commandButtonClicked(self, roachNums, command=None):
        """
        This function is executed when a command button is clicked. 
        
        INPUTS:
            roachNums - a list of roach numbers
            command - the command clicked
        """

        #source = self.sender()
        #print 'Roach: ',source.roach,'Command: ',RoachStateMachine.parseState(source.command)
        for roach_i in roachNums:
            roachArg = np.where(np.asarray(self.roachNums) == roach_i)[0][0]
            #if self.threadPool[roachArg].isRunning():
            if self.roachThreads[roachArg].isRunning():
                print 'Roach '+str(roach_i)+' is busy'
            elif command is None:
                self.roachThreads[roachArg].start()
            else: 
                #colorStatus = self.threadPool[roachArg].roach.addCommands(source.command)        # add command to roach queue
                colorStatus = self.roaches[roachArg].addCommands(command)        # add command to roach queue
                self.colorCommandButtons(roach_i,colorStatus)                               #Change color of command buttons
                #self.threadPool[roachArg].start()
                self.roachThreads[roachArg].start()                                     

    
    def colorCommandButtons(self,roachNum,colorList):
        """
        Changes the color of the command buttons
        
        Color Scheme:
            green - completed
            darkRed - undefined
            cyan - command in progress
            red - error
            blue - color for 'all roach' buttons
            gray/grey - unused
            
        INPUTS:
            roachNum - specify roach
            colorList - list of colors. One color for each possible command. 
                      - colors can be named or numbers corresponding to RoachStateMachine states
                      - if color is None then don't change anything
        """
        roachArg = np.where(np.asarray(self.roachNums) == roachNum)[0][0]
        for com in range(len(colorList)):
            color = colorList[com]
            if color is None:
                continue
            if color=='red' or color==RoachStateMachine.UNDEFINED:
                self.commandButtons[roachArg][com].setPalette(self.redButtonPalette)
            elif color=='cyan' or color==RoachStateMachine.INPROGRESS:
                self.commandButtons[roachArg][com].setPalette(self.cyanButtonPalette)
            elif color=='green' or color==RoachStateMachine.COMPLETED:
                self.commandButtons[roachArg][com].setPalette(self.greenButtonPalette)
            elif color=='error' or color==RoachStateMachine.ERROR:
                self.commandButtons[roachArg][com].setPalette(self.errorButtonPalette)
            elif color=='blue':
                self.commandButtons[roachArg][com].setPalette(self.blueButtonPalette)
            elif color=='grey' or color=='gray':
                self.commandButtons[roachArg][com].setPalette(self.grayButtonPalette)

    def commandButtonRightClicked(self, roachNums, command, source, point):
        """
        This function is called when a button is right clicked
        
        INPUTS:
            roachNums - list of roach numbers
            command - 
            source - the button object clicked
            point - the customContextMenuRequested() SIGNAL passes a QPoint argument specifying the location in the button that was clicked
        """
        #print 'here: ',point
        #source = self.sender()
        print 'openMenu for roach: ',roachNums,' Command: ',RoachStateMachine.parseCommand(command)
        
        self.contextMenu.clear()    # remove any actions added during previous right click
        
        if command == RoachStateMachine.DEFINEDACLUT:
            self.contextMenu.addAction('Auto ADC Atten',partial(self.onContextAutoADCatten,roachNums))
            self.contextMenu.addSeparator()

        if command == RoachStateMachine.SWEEP:
            self.contextMenu.addAction('Plot Sweep',partial(self.onContextPlotSweepClick,roachNums))
            self.contextMenu.addSeparator()
            self.contextMenu.addAction('Auto ADC Atten',partial(self.onContextAutoADCatten,roachNums))
            self.contextMenu.addSeparator()
        
        if command == RoachStateMachine.LOADTHRESHOLD:
            self.contextMenu.addAction('Plot Phase Timestream',partial(self.onContextPlotThreshClick,roachNums))
            self.contextMenu.addSeparator()
        
        settingsAction = self.contextMenu.addAction('Settings',partial(self.onContextSettingsClick,roachNums))
        self.contextMenu.exec_(source.mapToGlobal(point))   # point is referenced to local coordinates in the button. Need to reference to global coordinate system

    
    def onContextSettingsClick(self, roachNum):
        index = np.where(np.asarray(self.settingsWindow.roachNums) == roachNum)[0][0]
        self.settingsWindow.setCurrentIndex(index)
        self.settingsWindow.show()
    
    def onContextAutoADCatten(self, roachNums):
        for roachNum in roachNums:
            roachArg = np.where(np.asarray(self.roachNums) == roachNum)[0][0]
            if self.roachThreads[roachArg].isRunning():
                print 'Roach '+str(roachNum)+' is busy'
            else:
                adcAtten = self.roaches[roachArg].config.getfloat('Roach '+str(roachNum),'adcatten')
                newAdcAtten=self.roaches[roachArg].roachController.getOptimalADCAtten(adcAtten)
                self.sweepWindows[roachArg].updateADCAttenSpinBox(newAdcAtten)

    def onContextPlotSweepClick(self, roachNums):
        for roachNum in roachNums:
            roachArg = np.where(np.asarray(self.roachNums) == roachNum)[0][0]
            self.sweepWindows[roachArg].show()
    
    def onContextPlotThreshClick(self, roachNums):
        for roachNum in roachNums:
            roachArg = np.where(np.asarray(self.roachNums) == roachNum)[0][0]
            self.phaseWindows[roachArg].show()
            

    def create_main_frame(self):
        """
        Makes GUI. 
        
        Creates array of QPushButtons for each roach and command. Each button has the following important attributes:
         - roach: list of roaches to which the command applies (either 1 number or all roaches)
         - command: integer representing the command the button is for. (see RoachStateMachine class for how these are defined)
        """
        self.main_frame = QWidget()
        
        #button color palettes        
        self.greenButtonPalette = QPalette()
        self.greenButtonPalette.setColor(QPalette.Button,Qt.green)
        self.redButtonPalette = QPalette()
        self.redButtonPalette.setColor(QPalette.Button,Qt.darkRed)
        self.grayButtonPalette = QPalette()
        self.grayButtonPalette.setColor(QPalette.Button,Qt.gray)
        self.blueButtonPalette = QPalette()
        self.blueButtonPalette.setColor(QPalette.Button,Qt.blue)
        self.errorButtonPalette = QPalette()
        self.errorButtonPalette.setColor(QPalette.Button,Qt.red)
        self.cyanButtonPalette = QPalette()
        self.cyanButtonPalette.setColor(QPalette.Button,Qt.darkCyan)


        button_size = 25
        label_length = 110        
        label_height = 10
        
        #Command Labels:
        commandLabels = []
        for com in range(RoachStateMachine.NUMCOMMANDS):
            label = QLabel(RoachStateMachine.parseCommand(com))
            label.setMaximumWidth(label_length)
            label.setMinimumWidth(label_length)
            label.setMaximumHeight(button_size)
            label.setMinimumHeight(button_size)
            commandLabels.append(label)

        #Roach Labels
        label_roachNum = QLabel('Roach:')
        label_roachNum.setMaximumWidth(label_length)
        label_roachNum.setMinimumWidth(label_length)
        label_roachNum.setMaximumHeight(label_height)
        label_roachNum.setMinimumHeight(label_height)
        roachLabels=[]
        for i in self.roachNums:
            roachLabels.append(QLabel(str(i)))
        label_allRoaches = QLabel('All')
        roachLabels.append(label_allRoaches)
        for label in roachLabels:
            label.setMaximumWidth(button_size)
            label.setMinimumWidth(button_size)
            label.setMaximumHeight(label_height)
            label.setMinimumHeight(label_height)
        
        #Command buttons
        self.commandButtons=[]
        for i in range(self.numRoaches+1):
            roach_i_commandButtons=[]
            for j in range(len(commandLabels)):
                button = QPushButton()
                button.setMaximumWidth(button_size)
                button.setMinimumWidth(button_size)
                button.setMaximumHeight(button_size)
                button.setMinimumHeight(button_size)
                button.setEnabled(True)
                button.setPalette(self.redButtonPalette)
                try:
                    button.roach = [self.roachNums[i]]
                except IndexError:
                    button.roach=self.roachNums #all roach button
                    button.setPalette(self.blueButtonPalette)
                button.command = j
                button.clicked.connect(partial(self.commandButtonClicked, button.roach, button.command))
                #self.connect(button,QtCore.SIGNAL('clicked()'),partial(self.commandButtonClicked, button.roach, button.command))
                button.setContextMenuPolicy(Qt.CustomContextMenu)
                #self.connect(button,QtCore.SIGNAL('customContextMenuRequested()'),self.openMenu)
                #button.customContextMenuRequested.connect(self.openMenu)
                #self.connect(button,QtCore.SIGNAL('released()'),self.commandButtonClicked)
                button.customContextMenuRequested.connect(partial(self.commandButtonRightClicked, button.roach, button.command, button))
                #self.connect(button, QtCore.SIGNAL('customContextMenuRequested(const QPoint&)'), self.commandButtonRightClicked)
                roach_i_commandButtons.append(button)
            self.commandButtons.append(roach_i_commandButtons)

        self.contextMenu = QtGui.QMenu(self)

        #command buttons/labels Layout
        hbox = QHBoxLayout()
        vbox0 = QVBoxLayout(spacing = 5)
        vbox0.addWidget(label_roachNum)
        for label in commandLabels:
            vbox0.addWidget(label)
        hbox.addLayout(vbox0)
        for i in range(self.numRoaches+1):
            vbox_i = QVBoxLayout(spacing = 5)
            vbox_i.addWidget(roachLabels[i])
            for j in range(len(commandLabels)):
                vbox_i.addWidget(self.commandButtons[i][j])
            hbox.addLayout(vbox_i)


        self.main_frame.setLayout(hbox)
        self.setCentralWidget(self.main_frame)
    
    def create_status_bar(self):
        self.status_text = QLabel("Awaiting orders.")
        self.statusBar().addWidget(self.status_text, 1)
        
    def create_menu(self):        
        self.file_menu = self.menuBar().addMenu("&File")
        quit_action = self.create_action("&Quit", slot=self.close,shortcut="Ctrl+Q", tip="Close the application")
        settings_action = self.create_action("&Settings",shortcut="Ctrl+S",slot=self.settingsWindow.show, tip="Open Settings Window")
        self.add_actions(self.file_menu, (settings_action, None, quit_action))
        
        #self.otherCommands_menu = self.menuBar().addMenu("Other &Commands")
        #powerSweep_action = self.create_action("&Power Sweep", shortcut='Ctrl+P',slot=self.on_powerSweep, tip='Run Power Sweep')
        #snapShot_action = self.create_action("Collect Pulse &Template Data", shortcut='Ctrl+T',slot=self.on_snapShot, tip='Do a long snapshot to collect data for pulse templates')
        #self.add_actions(self.otherCommands_menu,(powerSweep_action,snapShot_action))

        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.create_action("&About", shortcut='F1',slot=self.on_about, tip='About the demo')
        self.add_actions(self.help_menu, (about_action,))
    
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
    
    def on_powerSweep(self):
        print 'Pop up power sweep window'
    
    def on_snapShot(self):
        print 'Pop up snap shot window'
    
    #def show_Settings(self):
    #    self.settingsWindow.show()

    def on_about(self):
        msg = "Click the buttons to execute the commands\n"\
              "Right click for more options\n"\
              "\n"\
              "Color Scheme:\n" \
              "  green - \tcommand completed\n" \
              "  darkRed - \tcommand not completed\n" \
              "  cyan - \t\tcommand in progress\n" \
              "  red - \t\terror\n" \
              "  blue - \t\tcolor for 'all roach' buttons\n" \
              "  gray/grey - \tunused" \
              "\n\n" \
              "Author: Alex Walter\n" \
              "Date: May 15, 2016"
        QMessageBox.about(self, "MKID-ROACH2 software", msg.strip())

    def closeEvent(self, event):
        self.settingsWindow._want_to_close=True
        self.settingsWindow.close()
        for window in self.sweepWindows:
            window._want_to_close=True
            window.close()
        for window in self.phaseWindows:
            window._want_to_close=True
            window.close()
        #for thread in self.threadPool:
        #    thread.quit()
        #    del thread
        
        newConfigFN = self.defaultValues.rsplit('.',1)[0]+'_NEW.'+ self.defaultValues.rsplit('.',1)[1]
        with open(newConfigFN, 'wb') as configfile:
            self.config.write(configfile)
        
        time.sleep(.1)
        QtCore.QCoreApplication.instance().quit


if __name__ == "__main__":
    app = QApplication(sys.argv)
    args = sys.argv[1:]
    defaultValues=None
    if '-c' in args:
        indx = args.index('-c')
        defaultValues=args[indx+1]
        try: args = args[:indx]+args[indx+2:]
        except IndexError:args = args[:indx]
    roachNums = np.asarray(args, dtype=np.int)
    print defaultValues,roachNums

    '''
    try: roachNums = np.asarray(sys.argv[1:],dtype=np.int)
    except: pass
    if len(sys.argv[1:]) == 2:
        if sys.argv[1] == '-a' or sys.argv[1] == '-all':
            roachNums = np.arange(int(sys.argv[2]),dtype=np.int)
        elif sys.argv[2] == '-a' or sys.argv[2] == '-all':
            roachNums = np.arange(int(sys.argv[1]),dtype=np.int)
    '''
    form = HighTemplar(roachNums,defaultValues)
    form.show()
    app.exec_()
