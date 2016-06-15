"""
AUTHOR: Alex Walter
DATE: May 15, 2016

The RoachStateMachine class runs the commands on the readout boards. Uses Roach2Controls Class
"""
import sys, time, random
import numpy as np
from PyQt4 import QtCore
from Queue import Queue
from Roach2Controls import Roach2Controls

'''
class RoachThread(QtCore.QThread):
    """
    This class implements a thread. Each readout board is controlled with its own thread.
    The thread checks if a command has been set for its roach, tells the roach to exectue the command, then communicates with a signal when it's done.
    
    INPUTS:
        roachNum - The identifying number of the roach
    
    SIGNALS:
        finishedCommand(int,PyQt_PyObject) - QtCore.SIGNAL that is emitted when a command is done executing. Passes two arguments, the command and any data from the command. (ie. I Q data from sweep)
        commandError(int,PyQt_PyObject) - QtCore.SIGNAL that is emitted when a command errors out. Passes two arguments, the command, and the information about the error. The last argument is actually a tuple but PyQt wraps all pythonic types into the PyQt_PyObject class when passing arguments. int is a c++ type
    """
    finishedCommandSignal = QtCore.SIGNAL("finishedCommand(int,PyQt_PyObject)")
    commandErrorSignal = QtCore.SIGNAL("commandError(int,PyQt_PyObject)")
    #commandSignal = QtCore.SIGNAL("command(int,int)")  # This is only used by the HighTemplar GUI to signal to this thread
    
    def __init__(self,roachNum):
        QtCore.QThread.__init__(self)
        self.roach = RoachStateMachine(roachNum)

    def __del__(self):
        print 'deleting Roach Thread'
        self.wait()     # Waits until event loop is finished before deleting (on exit)

    def run(self):
        while self.roach.hasCommand():
            command=self.roach.popCommand()
            try:
                commandData = self.roach.executeCommand(command)
                self.emit(RoachThread.finishedCommandSignal,command,commandData)
            except:
                exc_info = sys.exc_info()
                self.emit(RoachThread.commandErrorSignal,command,exc_info)
                del exc_info    # if you don't delete this it may prevent garbage collection
        return
'''

class RoachStateMachine(QtCore.QObject):        #Extends QObject for use with QThreads
    """
    This class defines and executes commands on the readout boards
    
    command enums are class variables
        0 - connect
        1 - loadFreq
        2 - etc...
    ie.. RoachStateMachine.connect = 0
         RoachStateMachine.parseCommand(0) = 'Connect'
    
    Commands are encoded as ints and loaded into a queue. The RoachThread pops the command and tells the RoachStateMachine to execute it.
    
    The 'state' attribute contains information about the state of the roach for each command.
    For example. self.state=[2, 2, 1, 1, 0, 0, 0, 0] means that the roach succesfully completed connect, loadFreq commands, and is currently working on DefineLUT and sweep. 
    State enums are class variables
        0 - undefined
        1 - inProgress
        2 - completed
        3 - error
    ie.. RoachStateMachine.completed=2
         RoachStateMachine.parseState(2)='Completed'
    
    
    """
    #FINISHED_SIGNAL = QtCore.SIGNAL("finishedCommand(int,PyQt_PyObject)")
    #ERROR_SIGNAL = QtCore.SIGNAL("commandError(int,PyQt_PyObject)")
    finishedCommand_Signal = QtCore.pyqtSignal(int,object)
    commandError_Signal = QtCore.pyqtSignal(int,tuple)
    finished = QtCore.pyqtSignal()
    
    NUMCOMMANDS = 8
    CONNECT,LOADFREQ,DEFINELUT,SWEEP,ROTATE,CENTER,LOADFIR,LOADTHRESHOLD = range(NUMCOMMANDS)
    NUMSTATES = 4
    UNDEFINED, INPROGRESS, COMPLETED, ERROR = range(NUMSTATES)
    
    @staticmethod
    def parseCommand(command):
        commandsString=['Connect','Load Freqs','Define LUTs','Sweep','Rotate','Center','Load FIRs','Load Thresholds']
        return commandsString[command]
    @staticmethod
    def parseState(state):
        statesString=['Undefined', 'In Progress', 'Completed', 'Error']
        return statesString[state]
    
    def __init__(self, roachNumber):
        super(RoachStateMachine, self).__init__()
        #self.threadClose = 0                # thread continues to listen for commands until this is set to 1
        self.state=[RoachStateMachine.UNDEFINED]*RoachStateMachine.NUMCOMMANDS    # This holds the state for each command type
        #self.nextState=RoachState.connect
        #self.status = 1						#1 if okay to execute next command
        #self.error = 0                      #error code: 0=okay, 1=error in connection, 2=error in loadFreq
        self.num=int(roachNumber)
        self.commandQueue=Queue()
        #self.commandQueue.put(RoachState.connect)
        
    def closeThread(self,setClosed=0):
        self.threadClose = setClosed
        return self.threadClose
    
    def addCommands(self,command):
        """
        This function adds the specified command and any other neccessary commands to the command queue.
        It also sets the state for each command correctly
        
        This is where the state machine logic comes into play. We check what state the roach is currently in, and decide what to do next. 
        ie. If we're currently only connected, but want to sweep, we need to first load freqs, define LUTs, then sweep. 
        
        The only tricky part is that we want to load FIRs only once or when explicitly asked
        
        INPUTS:
            command - the command we want to ultimately execute
            
        OUPUTS:
            self.state - the state for each command so the GUI can change the colors. (ie. the command will be in progress)
        """
        self.state = self.getNextState(command)
        for com in range(len(self.state)):
            if self.state[com] == RoachStateMachine.INPROGRESS:
                self.pushCommand(com)
        
        return self.state
    
    def getNextState(self,command,_n=0):
        """
        Given the current state and a command, determine the next state. This function is recursive
        
        Inputs:
            command - the command we want to execute. ie. RoachStateMachine.loadThreshold
            _n - Internal parameter for recursion. Determines the level of recursion. External calls should always use default value
        
        Outputs:
            nextState - list of states for the roach. See self.state attribute
        """
        nextState = np.copy(self.state)
        
        if _n==0: # n==0 means this is the top level command
            # Any previously unfinished commands should be made undefined
            args_unfinished = np.where(nextState==RoachStateMachine.INPROGRESS)
            nextState[args_unfinished] = RoachStateMachine.UNDEFINED
            # Always redo the command if explicitly asked
            nextState[command] = RoachStateMachine.INPROGRESS
            #usually higher commands become undefined (except loadFIR)
            args_above = np.where((np.arange(RoachStateMachine.NUMCOMMANDS)>command) & (np.arange(RoachStateMachine.NUMCOMMANDS) != RoachStateMachine.LOADFIR))
            if command==RoachStateMachine.CONNECT:
                args_above = []     #reconnecting shouldn't change anything
            elif command==RoachStateMachine.SWEEP:
                args_above = []     #resweeping shouldn't change anything
            nextState[args_above] = RoachStateMachine.UNDEFINED
        elif nextState[command] != RoachStateMachine.COMPLETED:
            # a lower command needs to be run only if it's not already completed
            nextState[command] = RoachStateMachine.INPROGRESS
        
        if command == RoachStateMachine.CONNECT:
            #We've reached the bottom of the command list, so return
            return nextState
        
        if _n==0 and command == RoachStateMachine.LOADFIR:  #Special case
            # loading FIRs only requires connect, loadFreqs, and defineLUTs
            nextState[:RoachStateMachine.DEFINELUT+1] = self.getNextState(RoachStateMachine.DEFINELUT,_n+1)[:RoachStateMachine.DEFINELUT+1]
        else:
            # Everything else requires all lower commands be completed
            nextState[:command] = self.getNextState(command-1,_n+1)[:command]
        return nextState
        
        
    
    def hasCommand(self):
        return (not self.commandQueue.empty())
    def popCommand(self):
        if not self.commandQueue.empty():
            return self.commandQueue.get()
        else:
            return None
    def pushCommand(self,command):
        self.commandQueue.put(command)
    def emptyCommandQueue(self):
        while not self.commandQueue.empty():
            self.commandQueue.get()
    
    @QtCore.pyqtSlot()
    def executeCommands(self):
        while self.hasCommand():
            command=self.popCommand()
            try:
                commandData = self.executeCommand(command)
                #self.emit(RoachStateMachine.FINISHED_SIGNAL,command,commandData)
                self.finishedCommand_Signal.emit(command,commandData)
            except:
                exc_info = sys.exc_info()
                #self.emit(RoachStateMachine.ERROR_SIGNAL,command,exc_info)
                self.commandError_Signal.emit(command,exc_info)
                del exc_info    # if you don't delete this it may prevent garbage collection
        self.finished.emit()
    
    def executeCommand(self,command):
        print "Roach ",self.num," Recieved/executing command: ",RoachStateMachine.parseCommand(command)
        self.state[command] = RoachStateMachine.INPROGRESS
        returnData = None
        try:
            time.sleep(random.randint(1,3))
            if random.randint(1,50) == 1:
                raise NotImplementedError('Error msg here!')
            
            if command == RoachStateMachine.SWEEP:
                self.I_data = []
                self.Q_data = []
                for i in range(10):
                    theta = np.arange(10)/10.*2*np.pi
                    I = np.cos(theta) + (np.random.rand(10)-0.5)/10.
                    Q = np.sin(theta) + (np.random.rand(10)-0.5)/10.
                    self.I_data.append(I)
                    self.Q_data.append(Q)
                returnData = {'I':self.I_data,'Q':self.Q_data}
            
            
            self.state[command] = RoachStateMachine.COMPLETED
        except:
            self.emptyCommandQueue()
            self.state[command] = RoachStateMachine.ERROR
            raise
            
        return returnData
            
    def getIQdata(self):
        try:
            I = self.I_data
            Q = self.Q_data
        except:
            I=[]
            Q=[]
        return I,Q

