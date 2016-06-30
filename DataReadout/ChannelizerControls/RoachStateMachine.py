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

class RoachStateMachine(QtCore.QObject):        #Extends QObject for use with QThreads
    """
    This class defines and executes commands on the readout boards using the Roach2Controls object
    
    command enums are class variables
        0 - CONNECT
        1 - LOADFREQ
        2 - etc...
    ie.. RoachStateMachine.CONNECT = 0
         RoachStateMachine.parseCommand(0) = 'Connect'
    
    Commands are encoded as ints and loaded into a queue. The RoachThread pops the command and tells the RoachStateMachine to execute it.
    
    The 'state' attribute contains information about the state of the roach for each command.
    For example. self.state=[2, 2, 1, 1, 0, 0, 0, 0] means that the roach succesfully completed connect, loadFreq commands, and is currently working on DefineLUT and sweep. 
    State enums are class variables
        0 - UNDEFINED
        1 - INPROGRESS
        2 - COMPLETED
        3 - ERROR
    ie.. RoachStateMachine.COMPLETED=2
         RoachStateMachine.parseState(2)='Completed'
    
    TODO:
        uncomment everything in executeCommand()
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
    
    def __init__(self, roachNumber, config):
        """
        INPUTS:
            roachNumber - 
            config - ConfigParser Object holding all the parameters needed
        """
        super(RoachStateMachine, self).__init__()
        self.state=[RoachStateMachine.UNDEFINED]*RoachStateMachine.NUMCOMMANDS    # This holds the state for each command type
        self.num=int(roachNumber)
        self.commandQueue=Queue()
        self.config=config
        
        
        FPGAParamFile = self.config.get('Roach '+str(self.num),'FPGAParamFile')
        ip = self.config.get('Roach '+str(self.num),'ipaddress')
        
        self.roachController = Roach2Controls(ip,FPGAParamFile,True,False)
    
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
        if _n > 100:
            raise ValueError("Too many recursions!")
        
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
        
        
    

    
    @QtCore.pyqtSlot()
    def executeCommands(self):
        """
        Executes sequentially every command in the command Queue
        
        This slot function is called by a seperate thread from HighTemplar
        """
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
        """
        Executes individual commands
        
        INPUTS:
            command
        """
        print "Roach ",self.num," Recieved/executing command: ",RoachStateMachine.parseCommand(command)
        self.state[command] = RoachStateMachine.INPROGRESS
        returnData = None
        try:
            
            if command == RoachStateMachine.CONNECT:
                '''
                This function connects to the roach2 board and executes any initialization scripts
                '''
                #self.roachController.connect()
                ddsShift = self.config.getint('Roach '+str(self.num),'ddssynclag')
                #self.roachController.loadDdsShift(ddsShift)
                
                returnData=True
                
            elif command == RoachStateMachine.LOADFREQ:
                '''
                Calculates the everything we need to load into the LUTs on the Roach and ADC/DAC board
                '''
                loFreq = int(self.config.getfloat('Roach '+str(self.num),'lo_freq'))
                self.roachController.setLOFreq(loFreq)
                
                fn = self.config.get('Roach '+str(self.num),'freqfile')
                freqs, attens = np.loadtxt(fn,unpack=True)
                
                print 'freqs: ',freqs
                print 'attens: ',attens
                
                
                self.roachController.generateResonatorChannels(freqs)
                self.roachController.generateFftChanSelection()
                
                dacAtten = self.config.getfloat('Roach '+str(self.num),'dacatten_start')
                self.roachController.generateDacComb(resAttenList=attens,globalDacAtten=dacAtten)
                self.roachController.generateDdsTones()
                
                returnData=True
            
            elif command == RoachStateMachine.DEFINELUT:
                '''
                Loads values into ROACH2, ADC/DAC, and IF boards
                    DAC atten 1, 2
                    ADC atten
                    lo freq
                    DAC LUT
                    DDS LUT
                '''
                adcAtten = self.config.getfloat('Roach '+str(self.num),'adcatten')
                dacAtten = self.config.getfloat('Roach '+str(self.num),'dacatten_start')
                dacAtten1 = np.floor(dacAtten*2)/4.
                dacAtten2 = np.ceil(dacAtten*2)/4.
                #self.roachController.changeAtten(1,dacAtten1)
                #self.roachController.changeAtten(2,dacAtten2)
                #self.roachController.changeAtten(3,adcAtten)

                #self.roachController.loadLOFreq()
                
                #self.roachController.loadChanSelection()
                #self.roachController.loadDdsLUT()
                #self.roachController.loadDacLUT()
                
                returnData = True
                
                
            
            elif command == RoachStateMachine.SWEEP:
                print self.config.get('Roach '+str(self.num),'ipAddress')
                nfreqs = len(self.roachController.freqList)
                self.I_data = []
                self.Q_data = []
                for i in range(nfreqs):
                    theta = np.arange(10)/10.*2*np.pi
                    I = np.cos(theta) + (np.random.rand(10)-0.5)/10.
                    Q = np.sin(theta) + (np.random.rand(10)-0.5)/10.
                    self.I_data.append(I)
                    self.Q_data.append(Q)
                returnData = {'I':self.I_data,'Q':self.Q_data}
        
            else:
                time.sleep(random.randint(1,3))
                if random.randint(1,50) == 1:
                    raise NotImplementedError('Error msg here!')
            
            self.state[command] = RoachStateMachine.COMPLETED
        except:
            self.emptyCommandQueue()
            self.state[command] = RoachStateMachine.ERROR
            raise
            
        return returnData
        
            

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

