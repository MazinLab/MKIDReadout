"""
AUTHOR: Matt

The InitStateMachine class runs the commands on the readout boards. 
"""
import os, sys, time, random
import numpy as np
from PyQt4 import QtCore
from Queue import Queue
from Roach2Controls import Roach2Controls
from autoZdokCal import loadDelayCal, findCal
#from autoZdokCal_V2 import loadDelayCal, findCal
from myQdr import Qdr as myQdr

class InitStateMachine(QtCore.QObject):        #Extends QObject for use with QThreads
    """
    This class defines and executes commands on the readout boards using the Roach2Controls object.
    All the important stuff happens in the executeCommand() function
    
    command enums are class variables
        0 - CONNECT
        1 - LOADFREQ
        2 - etc...
    ie.. InitStateMachine.CONNECT = 0
         InitStateMachine.parseCommand(0) = 'Connect'
    
    Commands are encoded as ints and loaded into a queue. The RoachThread pops the command and tells the InitStateMachine to execute it.
    
    The 'state' attribute contains information about the state of the roach for each command.
    For example. self.state=[2, 2, 1, 1, 0, 0, 0, 0] means that the roach succesfully completed connect, loadFreq commands, and is currently working on DefineRoachLUT and DefineDacLUT. 
    State enums are class variables
        0 - UNDEFINED
        1 - INPROGRESS
        2 - COMPLETED
        3 - ERROR
    ie.. InitStateMachine.COMPLETED=2
         InitStateMachine.parseState(2)='Completed'
    
    TODO:
        uncomment everything in executeCommand()
    """
    #FINISHED_SIGNAL = QtCore.SIGNAL("finishedCommand(int,PyQt_PyObject)")
    #ERROR_SIGNAL = QtCore.SIGNAL("commandError(int,PyQt_PyObject)")
    finishedCommand_Signal = QtCore.pyqtSignal(int,object)
    commandError_Signal = QtCore.pyqtSignal(int,tuple)
    finished = QtCore.pyqtSignal()
    reset = QtCore.pyqtSignal(object)
    
    #NUMCOMMANDS = 8
    #CONNECT,LOADFREQ,DEFINELUT,SWEEP,ROTATE,CENTER,LOADFIR,LOADTHRESHOLD = range(NUMCOMMANDS)
    NUMCOMMANDS = 5
    CONNECT,PROGRAM_V6,INIT_V7,CAL_ZDOK,CAL_QDR = range(NUMCOMMANDS)
    NUMSTATES = 4
    UNDEFINED, INPROGRESS, COMPLETED, ERROR = range(NUMSTATES)
    
    @staticmethod
    def parseCommand(command):
        if command < 0: return 'Reset'
        commandsString=['Connect','Program V6','Initialize V7','Calibrate Z-DOK','Calibrate QDR']
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
        super(InitStateMachine, self).__init__()
        self.state=[InitStateMachine.UNDEFINED]*InitStateMachine.NUMCOMMANDS    # This holds the state for each command type
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
            if self.state[com] == InitStateMachine.INPROGRESS:
                self.pushCommand(com)
        
        return self.state
    
    def getNextState(self,command,_n=0):
        """
        Given the current state and a command, determine the next state. This function is recursive. Don't change default value of _n
        
        NOTE: if command < 0 then it resets everything (with _n=0)
              if command >= InitStateMachine.NUMCOMMANDS then it ensures every command is completed but doesn't redo any if they're already completed
        
        Inputs:
            command - the command we want to execute. ie. InitStateMachine.loadThreshold
            _n - Internal parameter for recursion. Determines the level of recursion. External calls should always use default value
        
        Outputs:
            nextState - list of states for the roach. See self.state attribute
        """
        if _n > 100:
            raise ValueError("Too many recursions!")
        
        nextState = np.copy(self.state)
        
        if _n==0: # n==0 means this is the top level command
            if command<InitStateMachine.CONNECT:
                return [InitStateMachine.UNDEFINED]*InitStateMachine.NUMCOMMANDS
                
            # Any previously unfinished commands should be made undefined
            args_unfinished = np.where(nextState==InitStateMachine.INPROGRESS)
            nextState[args_unfinished] = InitStateMachine.UNDEFINED
            if command>=InitStateMachine.NUMCOMMANDS:
                #Make sure everything's completed but don't explicitly run anything if they already are
                command = InitStateMachine.NUMCOMMANDS-1
                if nextState[command] != InitStateMachine.COMPLETED: nextState[command] = InitStateMachine.INPROGRESS
            else:
                #redo the command if explicitly asked
                nextState[command] = InitStateMachine.INPROGRESS
            
            #reprogramming the roach2 causes CAL_ZDOK and CAL_QDR to be undefined
            if command==InitStateMachine.PROGRAM_V6:
                nextState[[InitStateMachine.CAL_ZDOK, InitStateMachine.CAL_QDR]]=InitStateMachine.UNDEFINED
        elif nextState[command] != InitStateMachine.COMPLETED:
            # a lower command needs to be run only if it's not already completed
            nextState[command] = InitStateMachine.INPROGRESS
        
        if command <= InitStateMachine.CONNECT:
            #We've reached the bottom of the command list, so return
            return nextState
        
        nextState[:command] = self.getNextState(command-1,_n+1)[:command]
        return nextState
    
    @QtCore.pyqtSlot()
    @QtCore.pyqtSlot(int)
    def resetStateTo(self,command=-1):
        """
        Reset the roach to the state given by command. ie. Pretend we just clicked command and so set higher commands as undefined. But leave the command in the current state
        
        command<0 resets everything
        
        INPUTS:
            command - command to reset state to
        """
        print "Resetting r"+str(self.num)+' to '+InitStateMachine.parseCommand(command)
        self.state = np.asarray(self.getNextState(command))
        self.state[np.where(self.state==InitStateMachine.INPROGRESS)]=InitStateMachine.UNDEFINED
        self.state[command]=InitStateMachine.UNDEFINED
        self.reset.emit(self.state)
        self.finished.emit()
    
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
                #self.emit(InitStateMachine.FINISHED_SIGNAL,command,commandData)
                self.finishedCommand_Signal.emit(command,commandData)
            except:
                exc_info = sys.exc_info()
                #self.emit(InitStateMachine.ERROR_SIGNAL,command,exc_info)
                self.commandError_Signal.emit(command,exc_info)
                del exc_info    # if you don't delete this it may prevent garbage collection
        self.finished.emit()
    
    def connect(self):
        '''
        This function connects to the roach2 board and executes any initialization scripts
        '''
        ipaddress = self.config.get('Roach '+str(self.num),'ipaddress')
        self.roachController.ip = ipaddress
        self.roachController.connect()
        
        return True
    
    def programV6(self):
        fpgPath = self.config.get('Roach '+str(self.num),'fpgPath')
        self.roachController.fpga.upload_to_ram_and_program(fpgPath)
        print 'Fpga Clock Rate:',self.roachController.fpga.estimate_fpga_clock()
        return True
        
    def initV7(self):
        waitForV7Ready=self.config.getboolean('Roach '+str(self.num),'waitForV7Ready')
        self.roachController.initializeV7UART(waitForV7Ready=waitForV7Ready)
        print 'initialized uart'
        self.roachController.initV7MB()
        print 'initialized mb'
        #self.config.set('Roach '+str(self.num),'waitForV7Ready',False)
        self.roachController.setLOFreq(2.e9)
        self.roachController.loadLOFreq()
        print 'Set LO to 2 GHz'
        return True

    def calZdok(self):
        self.roachController.sendUARTCommand(0x4)
        print 'switched on ADC ZDOK Cal ramp'
        time.sleep(.1)

        nBitsRemovedInFFT = self.config.getint('Roach '+str(self.num),'nBitsRemovedInFFT')
        # if(nBitsRemovedInFFT == 0):
        #     self.roachController.setAdcScale(0.9375) #Max ADC scale value
        # else:
        #     self.roachController.setAdcScale(1./(2**nBitsRemovedInFFT))

        self.roachController.fpga.write_int('run',1)
        busDelays = [14,18,14,13]
        busStarts = [0,14,28,42]
        busBitLength = 12
        for iBus in xrange(len(busDelays)):
            delayLut = zip(np.arange(busStarts[iBus],busStarts[iBus]+busBitLength), 
                busDelays[iBus] * np.ones(busBitLength))
            loadDelayCal(self.roachController.fpga,delayLut)

        # calDict = findCal(self.roachController.fpga,nBitsRemovedInFFT)
        calDict = findCal(self.roachController.fpga)
        print calDict
        
        self.roachController.sendUARTCommand(0x5)
        print 'switched off ADC ZDOK Cal ramp'
        
        if not calDict['solutionFound']:
            raise ValueError
            
        return True
            
    def calQDR(self):
        bQdrFlip = True
        calVerbosity = 0
        bFailHard = False
        #self.roachController.fpga.get_system_information()
        results = {}
        for iQdr,qdr in enumerate(self.roachController.fpga.qdrs):
            mqdr = myQdr.from_qdr(qdr)
            print qdr.name
            results[qdr.name] = mqdr.qdr_cal2(fail_hard=bFailHard,verbosity=calVerbosity)

        print 'Qdr cal results:',results
        for qdrName in results.keys():
            if not results[qdrName]:
                raise ValueError
        return True
    
    def executeCommand(self,command):
        """
        Executes individual commands
        
        INPUTS:
            command
        """
        print "Roach ",self.num," Recieved/executing command: ",InitStateMachine.parseCommand(command)
        self.state[command] = InitStateMachine.INPROGRESS
        returnData = None
        time.sleep(random.randint(1,3))
        try:
            
            if command == InitStateMachine.CONNECT:
                returnData = self.connect()
            elif command == InitStateMachine.PROGRAM_V6:
                returnData = self.programV6()
            elif command == InitStateMachine.INIT_V7:
                returnData = self.initV7()
            elif command == InitStateMachine.CAL_ZDOK:
                returnData = self.calZdok()
            elif command == InitStateMachine.CAL_QDR:
                returnData = self.calQDR()
            else:
                raise NotImplementedError('No command: '+str(command))
            self.state[command] = InitStateMachine.COMPLETED
        except:
            self.emptyCommandQueue()
            self.state[command] = InitStateMachine.ERROR
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

