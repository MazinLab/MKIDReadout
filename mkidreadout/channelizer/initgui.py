#!/bin/env python
"""
Author:    Matt

This is a GUI class to setup the readout boards. 
This file only contains the GUI and control code. The roach commands are implemented in the InitStateMachine class.
Based on HighTemplar.py

Command Flow:
 - Create non blocking thread for each roach that just sits in memory
 - When button clicked, load command into command queue in RoachStateMachine object
 - start thread (by calling start()) which executes all commands in queue
 - When thread is done executing the command the RoachStateMachine object emits a singal notifying the GUI and the thread exits its event loop

NOTES:
 - do not add commands to the RoachStateMachine object while the thread is executing commands


"""
import sys, time, traceback, os, argparse
from datetime import datetime
from functools import partial
import numpy as np
from shutil import copy2
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
from PyQt4.QtGui import *
from pkg_resources import resource_filename

import mkidcore.corelog
import mkidcore.config
from mkidcore.corelog import getLogger, create_log
from mkidreadout.channelizer.InitStateMachine import InitStateMachine
from mkidreadout.channelizer.InitSettingsWindow import InitSettingsWindow


class InitGui(QMainWindow):
    def __init__(self, roachNums, config='roach.yml'):
        """
        Create GUI
        
        INPUTS:
            roachNums - list of roach numbers in the  config
            defaultValues - path to config file. See documentation on ConfigParser
        """
        self.roachNums = list(np.unique(roachNums))  # sorts and removes duplicates
        self.config = mkidcore.config.load(config)

        # Setup GUI
        super(InitGui, self).__init__()
        self.create_main_frame()
        self.setWindowTitle('Digital Readout Initialization')
        # self.create_status_bar()

        # Setup Settings window
        self.settingsWindow = InitSettingsWindow(self.roachNums, self.config, parent=None)  # keep parent None
        #  for now
        self.settingsWindow.resetRoach.connect(self.resetRoachState)
        self.settingsWindow.initTemplar.connect(self.initTemplar)
        QApplication.setStyle(QStyleFactory.create('plastique'))

        # Setup RoachStateMachine and threads for each roach
        self.roaches = []
        self.roachThreads = []
        for i in self.roachNums:
            roach = InitStateMachine(i, self.config)
            thread = QtCore.QThread(parent=self)  # if parent isn't specified then need to be careful to destroy thread
            thread.setObjectName("Roach_" + str(i))  # process name
            roach.finishedCommand_Signal.connect(
                partial(self.catchRoachSignal, i))  # call catchRoachSignal when roach finishes a command
            roach.commandError_Signal.connect(
                partial(self.catchRoachError, i))  # call catchRoachError when roach errors out on a command
            thread.started.connect(
                roach.executeCommands)  # When the thread is started, automatically call RoachStateMachine.executeCommands()
            roach.finished.connect(
                thread.quit)  # When all the commands are done executing stop the thread. Can be restarted with thread.start()
            roach.moveToThread(thread)  # The roach functions run on the seperate thread
            self.roaches.append(roach)
            self.roachThreads.append(thread)
            # self.destroyed.connect(self.thread.deleteLater)
            # self.destroyed.connect(self.roach.deleteLater)
            roach.reset.connect(partial(self.colorCommandButtons, i))

        # Create file menu
        self.create_menu()

        # Initialize by connecting to roaches over ethernet
        for i, num in enumerate(self.roachNums):
            colorStatus = self.roaches[i].addCommands(InitStateMachine.CONNECT)  # add command to roach queue
            self.colorCommandButtons(num, colorStatus)  # color the command buttons appropriately
            # QtCore.QMetaObject.invokeMethod(roach, 'executeCommands', Qt.QueuedConnection)
            self.roachThreads[i].start()  # starting the thread automatically invokes the roach's executeCommand function

    def test(self, roachNum, state):
        getLogger('Init').info("Roach " + str(roachNum) + ' - ' + str(state))

    def catchRoachError(self, roachNum, command, exc_info=None):
        """
        This function is executed when the GUI sees the commandError signal from a RoachThread
        
        INPUTS:
            command - (int) the command that finished
            roachNum - (int) the roach number
        """
        traceback.print_exception(*exc_info)
        getLogger('Init').info('Roach {} errored out: {}'.format(roachNum,
                                                                 InitStateMachine.parseCommand(command)))
        # self.colorCommandButtons(roaches=[roachNum],commands=[command],color='error')
        colorStatus = [None] * InitStateMachine.NUMCOMMANDS
        colorStatus[command] = "error"
        self.colorCommandButtons(roachNum, colorStatus)

    def catchRoachSignal(self, roachNum, command, commandData):
        """
        This function is executed when the GUI sees the finishedCommand signal from a RoachThread
        
        INPUTS:
            roachNum - (int) the roach number
            command - (int) the command that finished
            commandData - Data from the command. For Example, after sweep it returns a dictionary of I and Q values
        """
        getLogger("Init").info("Finished r{} {}".format(roachNum, InitStateMachine.parseCommand(command)))
        colorStatus = [None] * InitStateMachine.NUMCOMMANDS
        colorStatus[command] = 'green'
        self.colorCommandButtons(roachNum, colorStatus)

        if command == InitStateMachine.INIT_V7:
            self.settingsWindow.finishedInitV7(roachNum)

    def resetRoachState(self, roachNum, command):
        roachArg = self.roachNums.index(roachNum)
        QtCore.QMetaObject.invokeMethod(self.roaches[roachArg], 'resetStateTo', Qt.QueuedConnection,
                                        QtCore.Q_ARG(int, command))
        self.roachThreads[roachArg].start()

    def initTemplar(self, roachNum, state):
        roachArg = self.roachNums.index(roachNum)
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
        for roach_i in roachNums:
            roachArg = self.roachNums.index(roach_i)
            if self.roachThreads[roachArg].isRunning():
                getLogger('Init').info('Roach {} is busy'.format(roachArg))
            elif command is None:
                self.roachThreads[roachArg].start()
            else:
                colorStatus = self.roaches[roachArg].addCommands(command)  # add command to roach queue
                self.colorCommandButtons(roach_i, colorStatus)  # Change color of command buttons
                self.roachThreads[roachArg].start()

    def colorCommandButtons(self, roachNum, colorList):
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
        roachArg = self.roachNums.index(roachNum)
        for com in range(len(colorList)):
            color = colorList[com]
            if color is None:
                continue
            if color == 'red' or color == InitStateMachine.UNDEFINED:
                self.commandButtons[roachArg][com].setPalette(self.redButtonPalette)
            elif color == 'cyan' or color == InitStateMachine.INPROGRESS:
                self.commandButtons[roachArg][com].setPalette(self.cyanButtonPalette)
            elif color == 'green' or color == InitStateMachine.COMPLETED:
                self.commandButtons[roachArg][com].setPalette(self.greenButtonPalette)
            elif color == 'error' or color == InitStateMachine.ERROR:
                self.commandButtons[roachArg][com].setPalette(self.errorButtonPalette)
            elif color == 'blue':
                self.commandButtons[roachArg][com].setPalette(self.blueButtonPalette)
            elif color == 'grey' or color == 'gray':
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
        getLogger('Init').debug('openMenu for roach {} Command: {}'.format(roachNums,
                                InitStateMachine.parseCommand(command)))
        self.contextMenu.clear()  # remove any actions added during previous right click
        self.contextMenu.addAction('Settings', partial(self.onContextSettingsClick, roachNums))
        self.contextMenu.exec_(source.mapToGlobal(point))  # point is referenced to local coordinates in the
        # button. Need to reference to global coordinate system

    def onContextSettingsClick(self, roachNum):
        index = self.settingsWindow.roachNums.index(roachNum)
        self.settingsWindow.setCurrentIndex(index)
        self.settingsWindow.show()

    def create_main_frame(self):
        """
        Makes GUI. 
        
        Creates array of QPushButtons for each roach and command. Each button has the following important attributes:
         - roach: list of roaches to which the command applies (either 1 number or all roaches)
         - command: integer representing the command the button is for. (see RoachStateMachine class for how these are defined)
        """
        self.main_frame = QWidget()

        # button color palettes
        self.greenButtonPalette = QPalette()
        self.greenButtonPalette.setColor(QPalette.Button, Qt.green)
        self.redButtonPalette = QPalette()
        self.redButtonPalette.setColor(QPalette.Button, Qt.darkRed)
        self.grayButtonPalette = QPalette()
        self.grayButtonPalette.setColor(QPalette.Button, Qt.gray)
        self.blueButtonPalette = QPalette()
        self.blueButtonPalette.setColor(QPalette.Button, Qt.blue)
        self.errorButtonPalette = QPalette()
        self.errorButtonPalette.setColor(QPalette.Button, Qt.red)
        self.cyanButtonPalette = QPalette()
        self.cyanButtonPalette.setColor(QPalette.Button, Qt.darkCyan)

        button_size = 25
        label_length = 110
        label_height = 10

        # Command Labels:
        commandLabels = []
        for com in range(InitStateMachine.NUMCOMMANDS):
            label = QLabel(InitStateMachine.parseCommand(com))
            label.setMaximumWidth(label_length)
            label.setMinimumWidth(label_length)
            label.setMaximumHeight(button_size)
            label.setMinimumHeight(button_size)
            commandLabels.append(label)

        # Roach Labels
        label_roachNum = QLabel('Roach:')
        label_roachNum.setMaximumWidth(label_length)
        label_roachNum.setMinimumWidth(label_length)
        label_roachNum.setMaximumHeight(label_height)
        label_roachNum.setMinimumHeight(label_height)
        roachLabels = [QLabel(str(i)) for i in self.roachNums]
        roachLabels.append(QLabel('All'))
        for label in roachLabels:
            label.setMaximumWidth(button_size)
            label.setMinimumWidth(button_size)
            label.setMaximumHeight(label_height)
            label.setMinimumHeight(label_height)

        # Command buttons
        self.commandButtons = []
        for i in range(len(self.roachNums) + 1):
            roach_i_commandButtons = []
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
                    button.roach = self.roachNums  # all roach button
                    button.setPalette(self.blueButtonPalette)
                button.command = j
                button.clicked.connect(partial(self.commandButtonClicked, button.roach, button.command))
                # self.connect(button,QtCore.SIGNAL('clicked()'),partial(self.commandButtonClicked, button.roach, button.command))
                button.setContextMenuPolicy(Qt.CustomContextMenu)
                # self.connect(button,QtCore.SIGNAL('customContextMenuRequested()'),self.openMenu)
                # button.customContextMenuRequested.connect(self.openMenu)
                # self.connect(button,QtCore.SIGNAL('released()'),self.commandButtonClicked)
                button.customContextMenuRequested.connect(
                    partial(self.commandButtonRightClicked, button.roach, button.command, button))
                # self.connect(button, QtCore.SIGNAL('customContextMenuRequested(const QPoint&)'), self.commandButtonRightClicked)
                roach_i_commandButtons.append(button)
            self.commandButtons.append(roach_i_commandButtons)

        self.contextMenu = QMenu(self)

        # command buttons/labels Layout
        hbox = QHBoxLayout()
        vbox0 = QVBoxLayout(spacing=5)
        vbox0.addWidget(label_roachNum)
        for label in commandLabels:
            vbox0.addWidget(label)
        hbox.addLayout(vbox0)
        for i in range(len(self.roachNums) + 1):
            vbox_i = QVBoxLayout(spacing=5)
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
        quit_action = self.create_action("&Quit", slot=self.close, shortcut="Ctrl+Q", tip="Close the application")
        settings_action = self.create_action("&Settings", shortcut="Ctrl+S", slot=self.settingsWindow.show,
                                             tip="Open Settings Window")
        self.add_actions(self.file_menu, (settings_action, None, quit_action))

        # self.otherCommands_menu = self.menuBar().addMenu("Other &Commands")
        # powerSweep_action = self.create_action("&Power Sweep", shortcut='Ctrl+P',slot=self.on_powerSweep, tip='Run Power Sweep')
        # snapShot_action = self.create_action("Collect Pulse &Template Data", shortcut='Ctrl+T',slot=self.on_snapShot, tip='Do a long snapshot to collect data for pulse templates')
        # self.add_actions(self.otherCommands_menu,(powerSweep_action,snapShot_action))

        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.create_action("&About", shortcut='F1', slot=self.on_about, tip='About the demo')
        self.add_actions(self.help_menu, (about_action,))

    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def create_action(self, text, slot=None, shortcut=None,
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

    # def show_Settings(self):
    #    self.settingsWindow.show()

    def on_about(self):
        msg = ("Click the buttons to execute the commands\n" 
               "Right click for more options\n"
               "\n" 
               "Color Scheme:\n" 
               "  green - \tcommand completed\n" 
               "  darkRed - \tcommand not completed\n" 
               "  cyan - \tcommand in progress\n" 
               "  red - \t\terror\n" 
               "  blue - \t\tcolor for 'all roach' buttons\n" 
               "  gray/grey - \tunused" 
               "\n\n" 
               "Author: Alex Walter\n" 
               "Date: May 15, 2016")
        QMessageBox.about(self, "MKID-ROACH2 software", msg.strip())

    def closeEvent(self, event):
        self.settingsWindow._want_to_close = True
        self.settingsWindow.close()
        time.sleep(.1)
        QtCore.QCoreApplication.instance().quit


# todo intelligent errors on config file errors
# TODO figure out why settings dialog shows waitforv7 as checked even if set false in initgui.cfg
# todo improve programing detection and square greening
# todo add all functionailty of autozdokcal.py into initgui,  make  autozdokcal work with QDR firmware too long adc snaps
# compare with https://mazinlab.atlassian.net/wiki/spaces/READ/pages/edit/36995110?draftId=36995123&draftShareId=7ee6f311-ed8a-4e6a-a4bc-1c76bf0af9c1&


DEFAULT_CFG_FILE = resource_filename('mkidreadout', os.path.join('config','roach.yml'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MKID Init GUI')
    parser.add_argument('roaches', nargs='+', type=int, help='Roach numbers')
    parser.add_argument('-c', '--config', default=DEFAULT_CFG_FILE, dest='config',
                        type=str, help='The config file')
    parser.add_argument('--gencfg', default=False, dest='genconfig', action='store_true',
                        help='generate configs in CWD')
    args = parser.parse_args()

    if args.genconfig:
        copy2(DEFAULT_CFG_FILE, './')
        exit(0)

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
    create_log('Init', logfile='init_{}.log'.format(timestamp),
               console=True, mpsafe=True, propagate=False,
               fmt='%(asctime)s %(name)s %(levelname)s: %(message)s ',
               level=mkidcore.corelog.DEBUG)
    create_log('mkidreadout',
               console=True, mpsafe=True, propagate=False,
               fmt='%(asctime)s %(name)s %(funcName)s: %(levelname)s %(message)s ',
               level=mkidcore.corelog.DEBUG)
    create_log('casperfpga',
               console=True, mpsafe=True, propagate=False,
               fmt='%(asctime)s %(name)s %(funcName)s: %(levelname)s %(message)s ',
               level=mkidcore.corelog.INFO)

    app = QApplication(sys.argv)
    form = InitGui(args.roaches, config=args.config)
    form.show()
    app.exec_()
