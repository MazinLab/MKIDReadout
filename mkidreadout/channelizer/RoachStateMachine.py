"""
AUTHOR: Alex Walter
DATE: May 15, 2016

The RoachStateMachine class runs the commands on the readout boards. Uses Roach2Controls Class
"""
import os, sys, time, random
import traceback
import numpy as np
from PyQt4 import QtCore
from Queue import Queue
from mkidreadout.channelizer.Roach2Controls import Roach2Controls
from mkidreadout.utils import iqsweep
from mkidcore.corelog import getLogger
from pkg_resources import resource_filename
import mkidreadout.configuration.powersweep.sweepdata as sweepdata

class RoachStateMachine(QtCore.QObject):  # Extends QObject for use with QThreads
    """
    This class defines and executes commands on the readout boards using the Roach2Controls object.
    All the important stuff happens in the executeCommand() function
    
    command enums are class variablesi
        0 - CONNECT
        1 - LOADFREQ
        2 - etc...
    ie.. RoachStateMachine.CONNECT = 0
         RoachStateMachine.parseCommand(0) = 'Connect'
    
    Commands are encoded as ints and loaded into a queue. The RoachThread pops the command and tells the RoachStateMachine to execute it.
    
    The 'state' attribute contains information about the state of the roach for each command.
    For example. self.state=[2, 2, 1, 1, 0, 0, 0, 0] means that the roach succesfully completed connect, loadFreq commands, and is currently working on DefineRoachLUT and DefineDacLUT. 
    State enums are class variables
        0 - UNDEFINED
        1 - INPROGRESS
        2 - COMPLETED
        3 - ERROR
    ie.. RoachStateMachine.COMPLETED=2
         RoachStateMachine.parseState(2)='Completed'
    
    SIGNALS:
        finishedCommand_Signal - Emitted whenever a command is finished. 
                                 The first argument is the command, the second is any data from the command
        commandError_Signal - Emitted when a command errors out
                              The first argument is the command, the second is a tuple containing the sys.exc_info()
        finished - Emitted when the queue is empty
                   also emited after other slots are called and complete succesfully
        reset - Emitted when the roach succesfully resets to some state
                argument is the state
        snapPhase - Emitted after snapping phase data
                    arg1 = channel, arg2 = phaseSnapDict
                    phaseSnapDict is a dictionary with keywords 'phase', 'trig'
        timestreamPhase - Emitted after collecting a a long phase snapshot over ethernet
                          arg1 = channel, arg2 = phase data
        ddsShift - Emitted when we load a DDS Sync lag into the ROACH2
                   arg1 = dds lag
    """
    finishedCommand_Signal = QtCore.pyqtSignal(int, object)
    commandError_Signal = QtCore.pyqtSignal(int, tuple)
    finished = QtCore.pyqtSignal()
    reset = QtCore.pyqtSignal(object)

    snapPhase = QtCore.pyqtSignal(int, object)
    timestreamPhase = QtCore.pyqtSignal(int, object)
    ddsShift = QtCore.pyqtSignal(int)

    NUMCOMMANDS = 9
    CONNECT, LOADFREQ, DEFINEROACHLUT, DEFINEDACLUT, SWEEP, ROTATE, TRANSLATE, LOADFIR, LOADTHRESHOLD = range(
        NUMCOMMANDS)
    NUMSTATES = 4
    UNDEFINED, INPROGRESS, COMPLETED, ERROR = range(NUMSTATES)

    @staticmethod
    def parseCommand(command):
        # commandsString=['Connect','Load Freqs','Define LUTs','Sweep','Rotate','Center','Load FIRs','Load Thresholds']
        # commandsString=['Connect','Read Freqs','Define Roach LUTs','Define DAC LUTs','Sweep','Fit Loops','Load FIRs','Load Thresholds']
        commandsString = ['Connect', 'Read Freqs', 'Define Roach LUTs', 'Define DAC LUTs', 'Sweep', 'Rotate Loops',
                          'Load Centers', 'Load FIRs', 'Load Thresholds']
        if command < 0: return 'Reset'
        return commandsString[command]

    @staticmethod
    def parseState(state):
        statesString = ['Undefined', 'In Progress', 'Completed', 'Error']
        return statesString[state]

    def __init__(self, roachNumber, config):
        """
        INPUTS:
            roachNumber - 
            config - ConfigParser Object holding all the parameters needed
        """
        super(RoachStateMachine, self).__init__()
        self.state = [RoachStateMachine.UNDEFINED] * RoachStateMachine.NUMCOMMANDS  # This holds the state for each command type
        self.num = int(roachNumber)
        self.commandQueue = Queue()
        self.config = config

        FPGAParamFile = self.config.roaches.get('r{}.fpgaparamfile'.format(self.num))
        fl = self.config.roaches.get('r{}.feedline'.format(self.num))
        range = self.config.roaches.get('r{}.range'.format(self.num))
        ip = self.config.roaches.get('r{}.ip'.format(self.num))

        self.roachController = Roach2Controls(ip, FPGAParamFile, feedline=fl, num=self.num,
                                              range=range, verbose=True, debug=False)

    def addCommands(self, command):
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

    def getNextState(self, command, _n=0):
        """
        Given the current state and a command, determine the next state. This function is recursive. Don't change default value of _n
        
        NOTE: if command < 0 then it resets everything (with _n=0)
              if command >= RoachStateMachine.NUMCOMMANDS then it ensures every command is completed but doesn't redo any if they're already completed
        
        Inputs:
            command - the command we want to execute. ie. RoachStateMachine.loadThreshold
            _n - Internal parameter for recursion. Determines the level of recursion. External calls should always use default value
        
        Outputs:
            nextState - list of states for the roach. See self.state attribute
        """
        if _n > 100:
            raise ValueError("Too many recursions!")

        nextState = np.copy(self.state)

        if _n == 0:  # n==0 means this is the top level command
            if command < RoachStateMachine.CONNECT:
                return [RoachStateMachine.UNDEFINED] * RoachStateMachine.NUMCOMMANDS

            # Any previously unfinished commands should be made undefined
            args_unfinished = np.where(nextState == RoachStateMachine.INPROGRESS)
            nextState[args_unfinished] = RoachStateMachine.UNDEFINED
            if command >= RoachStateMachine.NUMCOMMANDS:
                # Make sure everything's completed but don't explicitly run anything if they already are
                command = RoachStateMachine.NUMCOMMANDS - 1
                if nextState[command] != RoachStateMachine.COMPLETED: nextState[command] = RoachStateMachine.INPROGRESS
            else:
                # redo the command if explicitly asked
                nextState[command] = RoachStateMachine.INPROGRESS
            # usually higher commands become undefined (except loadFIR)
            args_above = np.where((np.arange(RoachStateMachine.NUMCOMMANDS) > command) & (
                        np.arange(RoachStateMachine.NUMCOMMANDS) != RoachStateMachine.LOADFIR))
            if command == RoachStateMachine.CONNECT:
                args_above = []  # reconnecting shouldn't change anything
            elif command == RoachStateMachine.SWEEP:
                args_above = []  # resweeping shouldn't change anything
            nextState[args_above] = RoachStateMachine.UNDEFINED
        elif nextState[command] != RoachStateMachine.COMPLETED:
            # a lower command needs to be run only if it's not already completed
            nextState[command] = RoachStateMachine.INPROGRESS

        if command <= RoachStateMachine.CONNECT:
            # We've reached the bottom of the command list, so return
            return nextState

        if _n == 0 and command == RoachStateMachine.LOADFIR:  # Special case
            # loading FIRs only requires connect and loadFreqs
            nextState[:RoachStateMachine.LOADFREQ + 1] = self.getNextState(RoachStateMachine.LOADFREQ, _n + 1)[
                                                         :RoachStateMachine.LOADFREQ + 1]
        else:
            # Everything else requires all lower commands be completed
            nextState[:command] = self.getNextState(command - 1, _n + 1)[:command]
        return nextState

    @QtCore.pyqtSlot()
    @QtCore.pyqtSlot(int)
    def resetStateTo(self, command=-1):
        """
        Reset the roach to the state given by command. ie. Pretend we just clicked command and so set higher commands as undefined. But leave the command in the current state
        
        command<0 resets everything
        
        INPUTS:
            command - command to reset state to
        """
        getLogger(__name__).info("Resetting r{} to {}".format(self.num, RoachStateMachine.parseCommand(command)))
        self.state = self.getNextState(command)
        self.state[np.where(self.state == RoachStateMachine.INPROGRESS)] = RoachStateMachine.UNDEFINED
        self.state[command] = RoachStateMachine.UNDEFINED
        self.reset.emit(self.state)
        self.finished.emit()

    @QtCore.pyqtSlot()
    def executeCommands(self):
        """
        Executes sequentially every command in the command Queue
        
        This slot function is called by a seperate thread from HighTemplar
        """
        while self.hasCommand():
            command = self.popCommand()
            try:
                commandData = self.executeCommand(command)
                self.finishedCommand_Signal.emit(command, commandData)
            except:
                exc_info = sys.exc_info()
                self.commandError_Signal.emit(command, exc_info)
                del exc_info  # if you don't delete this it may prevent garbage collection
        self.finished.emit()

    def connect(self):
        """
        This function connects to the roach2 board and executes any initialization scripts
        """
        ipaddress = self.config.roaches.get('r{}.ip'.format(self.num))
        self.roachController.ip = ipaddress
        self.roachController.connect()
        # self.roachController.initV7MB()
        self.loadDdsShift()
        self.roachController.loadFullDelayCal()

        return True

    def loadFreq(self):
        """
        Loads the resonator freq files (and attenuations, resIDs)
        divides the resonators into streams
        """
        try:
            getLogger(__name__).info('old Freq: {}'.format(self.roachController.freqList))
        except:
            pass
        fn = self.roachController.tagfile(self.config.roaches.get('r{}.freqfileroot'.format(self.num)),
                                          dir=self.config.paths.data)
        fn2 = '{0}_new.{1}'.format(*fn.rpartition('.')[::2])
        if os.path.isfile(fn2):
            fn = fn2

        getLogger(__name__).info('Loading freqs from ' + fn)

        sd = sweepdata.SweepMetadata(fn)

        resIDs, freqs, attens = sd.templar_data(self.config.roaches.get('r{}.lo_freq'.format(self.num)))
        #TODO Neelay, alex what about phaseOffsList and iqRatioList in the metadatafile
        phaseOffsList = np.zeros_like(attens)
        iqRatioList = np.ones_like(attens)

        for i in range(len(freqs)):
            getLogger(__name__).info("{} {} {} {} {} {}".format(i, resIDs[i], freqs[i], attens[i], phaseOffsList[i],
                                                                iqRatioList[i]))

        self.roachController.generateResonatorChannels(freqs)
        self.roachController.attenList = attens
        self.roachController.resIDs = resIDs
        self.roachController.phaseOffsList = phaseOffsList
        self.roachController.iqRatioList = iqRatioList
        getLogger(__name__).info('new Freq: {}'.format(self.roachController.freqList))

        return True

    def defineRoachLUTs(self):
        """
        Defines LO Freq but doesn't load it yet
        Defines and loads channel selection blocks
        Defines and loads DDS LUTs

        writing the QDR takes a long time! :-(
        """
        loFreq = self.config.roaches.get('r{}.lo_freq'.format(self.num))
        self.roachController.setLOFreq(loFreq)
        self.roachController.generateFftChanSelection()
        self.roachController.generateDdsTones()
        self.roachController.loadChanSelection()
        self.roachController.loadDdsLUT()
        return True

    def defineDacLUTs(self):
        """
        Defines and loads DAC comb
        Loads LO Freq
        Loads DAC attens 1, 2
        Loads ADC attens 1, 2
        """

        adcAtten = self.config.roaches.get('r{}.adcatten'.format(self.num))
        dacAtten = self.config.roaches.get('r{}.dacatten_start'.format(self.num))
        dacAtten1 = np.floor(dacAtten * 2) / 4.
        dacAtten2 = np.ceil(dacAtten * 2) / 4.
        adcAtten1 = np.floor(adcAtten * 2) / 4.
        adcAtten2 = np.ceil(adcAtten * 2) / 4.

        self.roachController.generateDacComb(globalDacAtten=dacAtten)

        getLogger(__name__).info("Initializing ADC/DAC board communication")
        self.roachController.initializeV7UART()
        getLogger(__name__).info("Setting DAC Atten")
        self.roachController.changeAtten(1, dacAtten1)
        self.roachController.changeAtten(2, dacAtten2)
        # self.roachController.changeAtten(3,adcAtten1)
        # self.roachController.changeAtten(4,adcAtten2)
        getLogger(__name__).info("Setting LO Freq")
        self.roachController.loadLOFreq()
        getLogger(__name__).info("Loading DAC LUT")
        self.roachController.loadDacLUT()
        getLogger(__name__).info("Auto Setting ADC Atten")
        newADCAtten = self.roachController.getOptimalADCAtten(adcAtten)

        # return True
        return newADCAtten

    def sweep(self):
        """
        Run power sweep
        If multiple dac attenuation values are specified then we loop over them and save a power sweep file
        See SDR repository for power sweep info. It's uncommented so your guess is as good as mine.

        sets the following attributes:
            self.I_data - [nFreqs, nLOsteps] array of I points
            self.Q_data -
            self.centers - [nFreqs, 2] array of I,Q loop centers

        NOTE: each I,Q point is actually an average of 1024 points done in firmware

        OUTPUTS:
            dictionary with keywords
            I - [nFreqs, nLOsteps] list of I points for each resonator
            Q -
            freqOffsets - list of LO offsets in Hz
            centers - [nFreqs, 2] list of I,Q centers
            IonRes - [nFreqs, 20] list of I points on Resonance (20 is just some arbitrary number)
            QonRes -
        """
        LO_freq = self.roachController.LOFreq
        LO_span = self.config.roaches.get('r{}.sweeplospan'.format(self.num))
        LO_start = LO_freq - LO_span / 2.
        LO_end = LO_freq + LO_span / 2.
        LO_step = self.config.roaches.get('r{}.sweeplostep'.format(self.num))
        LO_offsets = np.arange(LO_start, LO_end, LO_step) - LO_freq
        start_DACAtten = self.config.roaches.get('r{}.dacatten_start'.format(self.num))
        stop_DACAtten  = self.config.roaches.get('r{}.dacatten_stop'.format(self.num))
        start_ADCAtten = self.config.roaches.get('r{}.adcatten'.format(self.num))
        newADCAtten = start_ADCAtten

        powerSweepFile = self.roachController.tagfile(self.config.roaches.get('r{}.powersweeproot'.format(
            self.num)),
                                                      dir=self.config.paths.data,
                                                      epilog=time.strftime("%Y%m%d-%H%M%S", time.localtime()))
        for dacAtten in np.arange(start_DACAtten, stop_DACAtten + 1):
            if stop_DACAtten > start_DACAtten:
                dacAtten1 = np.floor(dacAtten * 2) / 4.
                dacAtten2 = np.ceil(dacAtten * 2) / 4.
                self.roachController.changeAtten(1, dacAtten1)
                self.roachController.changeAtten(2, dacAtten2)
                getLogger(__name__).info('Changed DAC atten: {}'.format(dacAtten))
                # keep total power on the ADC the same
                newADCAtten = self.roachController.getOptimalADCAtten(newADCAtten)
                getLogger(__name__).info('Changed ADC atten: {}'.format(newADCAtten))

            iqData = self.roachController.performIQSweep(LO_start / 1.e6, LO_end / 1.e6, LO_step / 1.e6)
            self.I_data = iqData['I']
            self.Q_data = iqData['Q']
            self.freqOffsets = iqData['freqOffsets']
            if stop_DACAtten > start_DACAtten:

                # Save the power sweep
                nSteps = len(self.freqOffsets)
                for n in range(len(self.roachController.freqList)):
                    w = iqsweep.IQsweep()
                    w.f0 = self.roachController.freqList[n]
                    w.span = LO_span / 1e6
                    w.fsteps = nSteps
                    w.atten1 = self.roachController.attenList[n] - start_DACAtten + dacAtten
                    w.atten2 = 0
                    w.scale = 1.
                    w.PreadoutdB = -w.atten1 - 20 * np.log10(w.scale)
                    w.Tstart = 0.100
                    w.Tend = 0.100
                    w.I0 = 0.0
                    w.Q0 = 0.0
                    w.resnum = n
                    w.resID = self.roachController.resIDs[n]
                    w.freq = w.f0 + self.freqOffsets
                    w.I = self.I_data[n]
                    w.Q = self.Q_data[n]
                    w.Isd = np.zeros(nSteps)
                    w.Qsd = np.zeros(nSteps)
                    w.time = time.time()
                    w.savenoise = 0
                    w.Save(powerSweepFile, 'r0', 'a')  # always r0

        # Get freq list, center, IQonResonance
        # Only for last sweep if power sweeping
        self.fitLoopCenters()  # uses self.I_data, self.Q_data and instantiates self.centers
        nPoints = 20  # arbitrary number. 20 seems fine. Could add this to config file in future
        # time.sleep(.1)  # I'm not sure if we need this but might need to wait for LO to stabilize after sweep
        iqOnRes = self.roachController.takeAvgIQData(nPoints)

        if stop_DACAtten > start_DACAtten:  # reset the dac/adc atten to the start value again if we did a power sweep
            dacAtten1 = np.floor(start_DACAtten * 2) / 4.
            dacAtten2 = np.ceil(start_DACAtten * 2) / 4.
            self.roachController.changeAtten(1, dacAtten1)
            self.roachController.changeAtten(2, dacAtten2)
            getLogger(__name__).info('Returned DAC atten: {}'.format(start_DACAtten))

            adcAtten1 = np.floor(start_ADCAtten * 2) / 4.
            adcAtten2 = np.ceil(start_ADCAtten * 2) / 4.
            self.roachController.changeAtten(3, adcAtten1)
            self.roachController.changeAtten(4, adcAtten2)
            getLogger(__name__).info('Returned ADCatten: {}'.format(start_ADCAtten))

        fList = np.copy(self.roachController.dacQuantizedFreqList)
        fList[fList > (self.roachController.params['dacSampleRate'] / 2.)] -= self.roachController.params['dacSampleRate']
        fList += LO_freq

        return {'I': np.copy(self.I_data), 'Q': np.copy(self.Q_data), 'freqOffsets': np.copy(self.freqOffsets),
                # 'freqList':np.copy(self.roachController.freqList),
                'freqList': fList,
                'centers': np.copy(self.centers), 'IonRes': np.copy(iqOnRes['I']), 'QonRes': np.copy(iqOnRes['Q'])}

        # return {'I':self.I_data,'Q':self.Q_data}

        '''
        nfreqs = len(self.roachController.freqList)
        self.I_data = []
        self.Q_data = []
        nSteps = int(LO_span/LO_step)
        for i in range(nfreqs):
            theta = np.linspace(0.,1,nSteps)*2*np.pi
            I = np.cos(theta) + (np.random.rand(nSteps)-0.5)/10.
            Q = np.sin(theta) + (np.random.rand(nSteps)-0.5)/10.
            self.I_data.append(I)
            self.Q_data.append(Q)
        return {'I':self.I_data,'Q':self.Q_data}
        '''

    def rotateLoops(self):
        """
        Rotate loops so that the on resonance phase=0.
        It uses the loop center data from the last sweep performed.

        NOTE: We rotate around I,Q = 0. Not around the center of the loop
              When we translate after, we need to resweep

        Find rotation phase
            - Get average I and Q at resonant frequency
        Rewrite the DDS LUT with new phases

        OUTPUTS:
            dictionary with keywords:
            IonRes - The average I value on resonance for each resonator
            QonRes - The average Q value on resonance for each resonator
            rotation - The rotation angle for each resonator before phasing the DDS LUT
        """
        nIQPoints = 100  # arbitrary number. 100 seems fine. Could add this to config file in future
        averageIQ = self.roachController.takeAvgIQData(nIQPoints)
        avg_I = np.average(averageIQ['I'], 1) - self.centers[:, 0]
        avg_Q = np.average(averageIQ['Q'], 1) - self.centers[:, 1]
        rotation_phases = np.arctan2(avg_Q, avg_I)
        # rotation_phases = np.ones(rotation_phases.shape)*90*np.pi/180.

        phaseList = np.copy(self.roachController.ddsPhaseList)
        # channels, streams = self.roachController.freqChannelToStreamChannel()
        channels, streams = self.roachController.getStreamChannelFromFreqChannel()
        for i in range(len(channels)):
            getLogger(__name__).info("ch{} str{}".format(channels[i], streams[i]))
            getLogger(__name__).info("phaseList shape: {}".format(phaseList.shape))
            phaseList[channels[i], streams[i]] = phaseList[channels[i], streams[i]] + rotation_phases[i]

        # for i in range(len(self.roachController.freqList)):
        #    arg = np.where(self.roachController.freqChannels == self.roachController.freqList[i])
        #    #phaseList[arg]+=rotation_phases[i]
        #    phaseList[arg]+=-1*np.pi/2.

        self.roachController.generateDdsTones(phaseList=phaseList)
        self.roachController.loadDdsLUT()

        return {'IonRes': np.copy(averageIQ['I']), 'QonRes': np.copy(averageIQ['Q']),
                'rotation': np.copy(rotation_phases)}

    def translateLoops(self):
        """
        This function loads the IQ loop center into firmware

        First needs to sweep
        then find centers (happens in self.sweep())
        then load them

        NOTE: technically we don't have to resweep everytime. We just need to sweep after rotating

        OUTPUTS:
            sweepDict - see self.sweep()
        """
        sweepDict = self.sweep()
        # self.centers = sweepDict['centers'] # Actually, it already sets these in sweep()
        self.roachController.loadIQcenters(self.centers)
        return sweepDict

    '''
    def fitLoops(self):
        """
        Find the center
            - If centerbool is false then use the old center if it exists
        Upload center to ROACH2
        
        Find rotation phase
            - Get average I and Q at resonant frequency
        Rewrite the DDS LUT with new phases
        
        Sets self.centers
        """
        recenter = self.config.getboolean('Roach '+str(self.num),'centerbool')
        if not hasattr(self, 'centers'): recenter = True
        if recenter: 
            self.fitLoopCenters()
            self.roachController.loadIQcenters(self.centers)
        
        nIQPoints = 100
        averageIQ = self.roachController.takeAvgIQData(nIQPoints)
        avg_I = np.average(averageIQ['I'],1) - self.centers[:,0]
        avg_Q = np.average(averageIQ['Q'],1) - self.centers[:,1]
        rotation_phases = np.arctan2(avg_Q,avg_I)
        
        phaseList = np.copy(self.roachController.ddsPhaseList)
        for i in range(len(self.roachController.freqList)):
            arg = np.where(self.roachController.freqChannels == self.roachController.freqList[i])
            phaseList[arg]-=rotation_phases[i]
        
        self.roachController.generateDdsTones(phaseList=phaseList)
        self.roachController.loadDdsLUT()
        
        return {'centers':self.centers, 'iqOnRes':np.transpose([avg_I,avg_Q])}
        
        return True
    '''

    def fitLoopCenters(self):
        """
        Finds the (I,Q) center of the loops
        sets self.centers - [nFreqs, 2]
        """
        I_centers = (np.percentile(self.I_data, 95, axis=1) + np.percentile(self.I_data, 5, axis=1)) / 2.
        Q_centers = (np.percentile(self.Q_data, 95, axis=1) + np.percentile(self.Q_data, 5, axis=1)) / 2.

        self.centers = np.transpose([I_centers.flatten(), Q_centers.flatten()])

    def loadFIRs(self):
        """
        Loads FIR coefficients from file into firmware
        """
        firname = self.config.roaches.get('r{}.fircoefffile'.format(self.num))
        file = resource_filename('mkidreadout', os.path.join('resources', 'firfilters', firname))
        self.roachController.loadFIRCoeffs(file)
        return True

    def loadThreshold(self):
        """
        Grab phase snapshot a bunch of times and take std for each channel
        set threshold for each channel

        OUTPUTS:
            thresh - list of thresholds in radians
        """
        self.roachController.setMaxCountRate()  # default is 2500

        nfreqs = len(self.roachController.freqList)
        threshSig = self.config.roaches.get('r{}.numsigs_thresh'.format(self.num))
        nSnap = self.config.roaches.get('r{}.numsnaps_thresh'.format(self.num))
        thresh = []
        for i in range(nfreqs):
            data = []
            for k in range(nSnap):
                # sys.stdout.write("\rCollecting Phase on Ch: "+str(i)+" Snap "+str(k+1)+'/'+str(nSnap))
                # sys.stdout.flush()
                data.append(self.getPhaseFromSnap(i))
            thresh.append(-1 * np.std(data) * threshSig)
        # self.roachController.loadThresholds(thresh)
        # sys.stdout.write("\n")
        for i in range(nfreqs):
            # ch, stream = np.where(self.roachController.freqChannels == self.roachController.freqList[i])
            # ch = ch+stream*self.roachController.params['nChannelsPerStream']
            # self.roachController.setThresh(thresh[i],ch)
            self.roachController.setThreshByFreqChannel(thresh[i], i)

        # self.roachController.thresholds=thresh
        return thresh

    def executeCommand(self, command):
        """
        Executes individual commands
        
        INPUTS:
            command
        """
        getLogger(__name__).info("Roach {} Received/executing command: {}".format(self.num,
                                                                                 RoachStateMachine.parseCommand(
                                                                                     command)))
        self.state[command] = RoachStateMachine.INPROGRESS
        returnData = None
        time.sleep(random.randint(1, 3))  #TODO WTFH?
        try:

            if command == RoachStateMachine.CONNECT:
                returnData = self.connect()
            elif command == RoachStateMachine.LOADFREQ:
                returnData = self.loadFreq()
            elif command == RoachStateMachine.DEFINEROACHLUT:
                returnData = self.defineRoachLUTs()
            elif command == RoachStateMachine.DEFINEDACLUT:
                returnData = self.defineDacLUTs()
            elif command == RoachStateMachine.SWEEP:
                returnData = self.sweep()
            # elif command == RoachStateMachine.FIT:
            #    returnData = self.fitLoops()
            elif command == RoachStateMachine.ROTATE:
                returnData = self.rotateLoops()
            elif command == RoachStateMachine.TRANSLATE:
                returnData = self.translateLoops()
            elif command == RoachStateMachine.LOADFIR:
                returnData = self.loadFIRs()
            elif command == RoachStateMachine.LOADTHRESHOLD:
                returnData = self.loadThreshold()
            else:
                raise ValueError('No command: ' + str(command))
            self.state[command] = RoachStateMachine.COMPLETED
        except:
            self.emptyCommandQueue()
            self.state[command] = RoachStateMachine.ERROR
            raise

        return returnData

    @QtCore.pyqtSlot(int)
    def getPhaseFromSnap(self, channel):
        """
        This function grabs the phase timestream from the snapblock
        
        INPUTS:
            channel - the i'th frequency in the frequency list
        OUTPUTS:
            phaseSnap - list of phase in radians
        """

        getLogger(__name__).info("r{}: ch{} Getting phase snap".format(self.num, channel))
        # try:
        #    ch, stream = np.where(self.roachController.freqChannels == self.roachController.freqList[channel])
        # except AttributeError:
        #    print "Need to load freqs first!"
        #    self.finished.emit()
        #    return
        # ch = ch+stream*self.roachController.params['nChannelsPerStream']
        # phaseSnapDict = self.roachController.takePhaseSnapshot(channel=ch)

        try:
            phaseSnapDict = self.roachController.takePhaseSnapshotOfFreqChannel(channel)
        except:
            getLogger(__name__).error(exc_info=True)
            self.finished.emit()
            return

        # data=np.random.uniform(-.1,.1,200)+channel
        self.snapPhase.emit(channel, phaseSnapDict)
        self.finished.emit()
        return phaseSnapDict['phase']

    @QtCore.pyqtSlot(int, float)
    def getPhaseStream(self, channel, duration=2):
        """
        This function continuously observes the phase stream for a given amount of time
        
        INPUTS:
            channel - the i'th frequency in the frequency list
            time - [seconds] the amount of time to observe for
        OutPUTS:
            data - list of phase in radians
        """
        resID = self.roachController.resIDs[channel]
        getLogger(__name__).info("r{}:ch{} Collecting phase timestream".format(self.num, channel))
        # try:
        #    ch, stream = np.where(self.roachController.freqChannels == self.roachController.freqList[channel])
        # except AttributeError:
        #    print "Need to load freqs first!"
        #    self.finished.emit()
        #    return
        hostip = self.config.packetmaster.ip
        port = self.config.roaches.get('r{}.phaseport'.format(self.num))
        # ch = ch+stream*self.roachController.params['nChannelsPerStream']
        # data=self.roachController.takePhaseStreamData(selChanIndex=ch, duration=duration, hostIP=hostip)
        try:
            data = self.roachController.takePhaseStreamDataOfFreqChannel(freqChan=channel, duration=duration,
                                                                         hostIP=hostip, fabric_port=port)

            el = 'resID{:.0f}_{}'.format(resID, time.strftime("%Y%m%d-%H%M%S", time.localtime()))
            longSnapFN = self.roachController.tagfile(self.config.roaches.get('r{}.longsnaproot'.format(self.num)),
                                                      dir=self.config.paths.data, epilog=el)
            np.savez(longSnapFN, data)
        except IOError:
            path = longSnapFN.rsplit('/', 1)
            if len(path) <= 1:
                raise
            getLogger(__name__).info('Making directory: ' + path[0])
            os.mkdir(path[0])
            np.savez(longSnapFN, data)
        except:
            traceback.print_exc()
            self.finished.emit()
            return
        # time.sleep(timelen)
        # data=np.random.uniform(-.1,.1,timelen*10.**6)
        self.timestreamPhase.emit(channel, data)
        self.finished.emit()
        return data

    @QtCore.pyqtSlot()
    @QtCore.pyqtSlot(int)
    def loadDdsShift(self, ddsShift=-1):
        """
        This function loads the dds sync lag into the firmware
        
        INPUTS:
            ddsShit - Number of clock cycles for the shift
                      if -1, then automatically find out the number needed
        """
        if ddsShift is None or ddsShift < 0:
            ddsShift = self.roachController.checkDdsShift()
        newDdsShift = self.roachController.loadDdsShift(ddsShift)
        self.ddsShift.emit(newDdsShift)
        self.finished.emit()

    @QtCore.pyqtSlot(float)
    def loadADCAtten(self, adcAtten):
        """
        loads a new adc attenuation

        INPUTS:
            adcAtten - dB
        """
        adcAtten1 = np.floor(adcAtten * 2) / 4.
        adcAtten2 = np.ceil(adcAtten * 2) / 4.
        self.roachController.changeAtten(3, adcAtten1)
        self.roachController.changeAtten(4, adcAtten2)
        getLogger(__name__).info('r{}  Changed ADC Atten to {}+{}'.format(self.num, adcAtten1, adcAtten2))
        # self.roachController.changeAtten(3, adcAtten)
        # print 'r'+str(self.num)+'Changed ADCAtten to '+str(adcAtten)
        self.finished.emit()

    @QtCore.pyqtSlot(float)
    def loadDACAtten(self, dacAtten):
        """
        loads a new dac attenuation.
        dacAtten is the total DAC attenuation.
        It needs to be divided into the two attenuators

        INPUTS:
            dacAtten - dB
        """
        dacAtten1 = np.floor(dacAtten * 2) / 4.
        dacAtten2 = np.ceil(dacAtten * 2) / 4.
        self.roachController.changeAtten(1, dacAtten1)
        self.roachController.changeAtten(2, dacAtten2)
        getLogger(__name__).info('r{}  Changed DAC Atten to {}+{}'.format(self.num, dacAtten1, dacAtten2))
        self.finished.emit()

    @QtCore.pyqtSlot(object)
    def initializeToState(self, state):
        """
        This function is convenient for debugging.
        If we restart templar, we don't need to reload everything into the boards again
        This function can be used to force templar to initialize everything without actually communicating to the boards

        INPUTS:
            state - Array of length numCommands with each value the state of that command
                    We will force templar to think it is in this state
        """
        for com in range(len(state)):
            if state[com] == RoachStateMachine.COMPLETED and self.state[com] != RoachStateMachine.COMPLETED:
                if com == RoachStateMachine.CONNECT:
                    returnData = self.connect()
                    self.finishedCommand_Signal.emit(com, returnData)
                if com == RoachStateMachine.LOADFREQ:
                    returnData = self.loadFreq()
                    self.finishedCommand_Signal.emit(com, returnData)
                if com == RoachStateMachine.DEFINEROACHLUT:
                    loFreq = self.config.roaches.get('r{}.lo_freq'.format(self.num))
                    self.roachController.setLOFreq(loFreq)
                    self.roachController.generateFftChanSelection()
                    self.roachController.generateDdsTones()
                    self.finishedCommand_Signal.emit(com, True)
                if com == RoachStateMachine.DEFINEDACLUT:
                    dacAtten = self.config.roaches.get('r{}.dacatten_start'.format(self.num))
                    self.roachController.generateDacComb(globalDacAtten=dacAtten)
                    self.finishedCommand_Signal.emit(com, True)
                if com == RoachStateMachine.SWEEP:
                    self.I_data = np.ones((len(self.roachController.freqList), 2))
                    self.Q_data = np.copy(self.I_data)
                    self.centers = np.ones((len(self.roachController.freqList), 2))
                    freqOffsets = np.asarray([-1, 1])
                    iOnRes = np.copy(self.I_data)
                    qOnRes = np.copy(self.I_data)
                    returnData = {'I': self.I_data, 'Q': self.Q_data, 'freqOffsets': freqOffsets,
                                  'centers': self.centers, 'IonRes': iOnRes, 'QonRes': qOnRes}
                    self.finishedCommand_Signal.emit(com, returnData)
                # if com ==RoachStateMachine.FIT:
                #    self.centers=np.zeros((len(self.roachController.freqList),2))
                #    returnData={'centers':self.centers, 'iqOnRes':self.centers}
                #    self.finishedCommand_Signal.emit(com,returnData)
                if com == RoachStateMachine.ROTATE:
                    iOnRes = np.ones(len(self.roachController.freqList))
                    returnData = {'IonRes': iOnRes, 'QonRes': np.copy(iOnRes), 'rotation': np.copy(iOnRes)}
                    self.finishedCommand_Signal.emit(com, returnData)
                if com == RoachStateMachine.TRANSLATE:
                    self.I_data = np.ones((len(self.roachController.freqList), 2))
                    self.Q_data = np.copy(self.I_data)
                    self.centers = np.ones((len(self.roachController.freqList), 2))
                    returnData = {'I': self.I_data, 'Q': self.Q_data, 'freqOffsets': np.asarray([-1, 1]),
                                  'centers': self.centers, 'IonRes': np.copy(self.I_data),
                                  'QonRes': np.copy(self.I_data)}
                    self.finishedCommand_Signal.emit(com, returnData)
                if com == RoachStateMachine.LOADFIR:
                    self.finishedCommand_Signal.emit(com, True)
                if com == RoachStateMachine.LOADTHRESHOLD:
                    thresh = np.asarray([-.1] * len(self.roachController.freqList))
                    returnData = thresh
                    self.finishedCommand_Signal.emit(com, returnData)

            if state[com] == RoachStateMachine.INPROGRESS:
                self.pushCommand(com)
            self.state[com] = state[com]

        self.reset.emit(self.state)
        self.finished.emit()

    def hasCommand(self):
        return (not self.commandQueue.empty())

    def popCommand(self):
        if not self.commandQueue.empty():
            return self.commandQueue.get()
        else:
            return None

    def pushCommand(self, command):
        self.commandQueue.put(command)

    def emptyCommandQueue(self):
        while not self.commandQueue.empty():
            self.commandQueue.get()
