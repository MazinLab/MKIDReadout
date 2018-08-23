"""
Author: Alex Walter
Date: March 26, 2018

This code runs a widesweep with the digital readout.
It takes a list of board numbers, a templarconfig.cfg file, and a file prefix for the output files

Usage:
From command line
$ python digitalWS.py 220 221 -c /home/data/MEC/20180530/templarconf.cfg -o /home/data/MEC/20180530/HypatiaFL7b

From python
>>> digWS = DigitalWideSweep([220,221], 'hightemplar.cfg', '/home/data/MEC/20180330/example')
>>> digWS.startWS(roachNums=[220,221], startFreqs=[3.5E9, 5.5E9], endFreqs=[5.5E9,7.5E9],lo_step=None, DACatten=6., ADCatten=30.,makeNewFreqs=True, resAtten=65)


Note: Power in dBm is approximately -(resAtten + 12). (so a res atten of 65 will have a power of -77 dBm)
"""


import traceback, sys, warnings
from functools import partial
import ConfigParser
from PyQt4 import QtCore
import numpy as np
import matplotlib.pyplot as plt

from mkidreadout.channelizer.RoachStateMachine import RoachStateMachine






class DigitalWideSweep(QtCore.QObject):

    def __init__(self, roachNums=None, defaultValues=None, outputPath=None,debug=False):
        '''
        Initialize DigitalWideSweep object
        
        INPUTS:
            roachNums - list of roach numbers. ie. [220,221,230,234]
            defaultValues - path to config file Same as templarConfig. See documentation on ConfigParser
            outputPath - the path to put output files plus any prefix on files
                         eg. '/home/data/MEC/20180330/test'
                         By default the frequency file (if created) will be saved in the current directory
                         By default the widesweep file will be saved in the same directory as the freq file
                         If there is no '/' in outputPath then it's assumed to be just a file prefix
                         If outputPath ends in '/' then it's assumed to be just a file path
            debug - If True, initialize the roach to sweep state. 
        '''
        super(QtCore.QObject, self).__init__()
        #if roachNums is None or len(roachNums) ==0:
        #    roachNums = range(10)
        self.debug=debug
        self.roachNums = np.unique(roachNums)       # sorts and removes duplicates
        self.numRoaches = len(self.roachNums)       # (int) number of roaches connected
        self.config = ConfigParser.ConfigParser()

        #todo change to from channelizer.hightemplar import defaultconfig or somesuch
        if defaultValues is None:
            defaultValues = '../../channelizer/hightemplar.cfg'
        self.config.read(defaultValues)

        #TODO make sure that the config file is sensible, e.g. error out if dacatten setting would result in
        # ValueError: Not enough dynamic range in DAC! Try decreasing the global DAC Attenuator by 36 dB


        self.outPath=''                             # Output directory for freq or widesweep files generated
        self.outPrefix=''                           # Prefix to put on files (like the name of the device or FL)
        if len(outputPath.rsplit('/',1))==2:
            self.outPath = outputPath.rsplit('/',1)[0]
            self.outPrefix=outputPath.rsplit('/',1)[1]
        else:
            self.outPrefix=outputPath.rsplit('/',1)[0]
        if len(self.outPrefix)>0: self.outPrefix+='_'

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
        
        #parameters for making random freq list
        self.toneBandwidth = 500.0E3   # Hz                                        # frequencies will not be closer together than this to avoid bandwidth overlap
        self.minNominalSpacing = 800.0E3    #Hz                                         # Will fit 1024 frequencies into bandwidth specified or fewer
        self.numoverlapPoints = 10                                              # neighboring tones will be swept such that at least 10 points overlap in frequency
        self.execQApp=0     #if 0 then stop the QApp. Otherwise, keep it running.
        self.app = QtCore.QCoreApplication(sys.argv)    # The signals emmited by RoachStateMachine objects require a QEventLoop to be running from a QApplication.
    
    def quitQApp(self):
        """
        Stop the QApplication to end the QEventLoop
        """
        self.execQApp -=1
        if self.execQApp<=0:
            self.app.quit()
            #print "Quit QApp"


    def catchRoachError(self,roachNum, command, exc_info=None):
        """
        This function is executed when the commandError signal is sent from a RoachThread
        
        INPUTS:
            roachNum - (int) the roach number
            command - (int) the command that finished
            exc_info - from the exception
            
        """
        traceback.print_exception(*exc_info)
        roachArg = np.where(np.asarray(self.roachNums) == roachNum)[0][0]
        print 'Roach ',roachNum,' errored out: ',RoachStateMachine.parseCommand(command)
        self.quitQApp()
    
    def catchRoachSignal(self,roachNum,command,commandData):
        """
        This function is executed when the finishedCommand signal is sent from a RoachThread
        
        INPUTS:
            roachNum - (int) the roach number
            command - (int) the command that finished
            commandData - Data from the command. For Example, after sweep it returns a dictionary of I and Q values
        """
        print "Finished r"+str(roachNum)+' '+RoachStateMachine.parseCommand(command)

        #if command == RoachStateMachine.DEFINEDACLUT: self.quitQApp()

        if command == RoachStateMachine.SWEEP:
            self.data = commandData
            self.writeWSdata(roachNum, commandData, 'raw_')
            try:
                calData = self.calibrateWSdata(roachNum, commandData)
                self.writeWSdata(roachNum, calData)
            except IndexError:
                pass
            self.quitQApp()


    def calibrateWSdata(self, roachNum, data):
        overlap_tolerance=1. # say that any tone less than 1Hz away is the same frequency

        I=data['I']
        Q=data['Q']
        A = (I**2. + Q**2.)**0.5
        #toneCorrections=[]
        corr=1.
        for ch in range(len(A)-1):
            tone2_min = data['freqList'][ch+1] + data['freqOffsets'][0]
            tones1 = data['freqOffsets'] + data['freqList'][ch]
            overlap_arg=np.where(np.isclose(tones1, [tone2_min]*len(data['freqOffsets']), atol = overlap_tolerance))[0][0]
            
            tone_corrs = 1.0*A[ch][overlap_arg:] / A[ch+1][:-1*overlap_arg]
            #toneCorrections.append(np.median(tone_corrs))
            corr=np.median(tone_corrs)
            data['I'][ch+1] = data['I'][ch+1]*corr
            data['Q'][ch+1] = data['Q'][ch+1]*corr
            A[ch+1]*=corr

        return data


    def writeWSdata(self, roachNum, data, filePrefix=None):
        roachArg = np.where(np.asarray(self.roachNums) == roachNum)[0][0]
        #freqFN = self.roaches[roachArg].config.get('Roach '+str(roachNum),'freqfile')
        #path=freqFN.rsplit('/',1)[0]
        widesweepFN = self.outPath+'/'+self.outPrefix+'digWS_r'+str(roachNum)+'.txt'
        if filePrefix is not None:
            widesweepFN = self.outPath+'/'+filePrefix+self.outPrefix+'digWS_r'+str(roachNum)+'.txt'
        

        I=data['I'].flatten()
        Q=data['Q'].flatten()
        freqs = np.tile(data['freqOffsets'], len(data['freqList'])) + np.repeat(data['freqList'], len(data['freqOffsets']))
        args = np.argsort(freqs)
        args=np.sort(args)  # actually, don't sort them for now
        
        outData = np.asarray([ freqs[args]/1.E9, I[args], Q[args]]).T
        #print outData.shape

        header='Widesweep with Digital Readout:\n'+\
               'Roach#: '+str(roachNum)+'\n'+\
               'freqlist FN: '+self.roaches[roachArg].config.get('Roach '+str(roachNum),'freqfile')+'\n'+\
               'LO [GHz]: '+str(self.roaches[roachArg].config.getfloat('Roach '+str(roachNum),'lo_freq')/1.E9)+'\n'+\
               'LO span [MHz]: '+str(self.roaches[roachArg].config.getfloat('Roach '+str(roachNum),'sweeplospan')/1.E6)+'\n'+\
               'LO step [kHz]: '+str(self.roaches[roachArg].config.getfloat('Roach '+str(roachNum),'sweeplostep')/1.E3)+'\n'+\
               'Res atten: '+str(self.roaches[roachArg].roachController.attenList[0])+'\n'+\
               'DAC atten: '+str(self.roaches[roachArg].config.getfloat('Roach '+str(roachNum),'dacatten_start'))+'\n'+\
               'ADC atten: '+str(self.roaches[roachArg].config.getfloat('Roach '+str(roachNum),'adcatten'))
        np.savetxt(widesweepFN, outData, fmt="%.9f %.9f %.9f",header=header)
        

    def startWS(self, roachNums=None, startFreqs=None, endFreqs=None, lo_step=None, DACatten=None, ADCatten=None,
                makeNewFreqs=True, **kwargs):
        """
        This function starts a widesweep on the roaches specified
        
        INPUTS:
            roachNums - The roaches to run a power sweep on. If none, then run all of them
            startFreqs - The starting sweep frequency for each roach. If None then default to 1GHz below LO
            endFreqs - The ending sweep frequency for each roach. If None then default to 1GHz above LO
            lo_step - LO step size. If not given, then uses value in the templar config file
            DACatten - 
            ADCatten - 
            makeNewFreqs - If true, then generate a new random freq file
            **kwargs - additional keywords for makeRandomFreqList() and atten from saveFreqList()
        """
        if not roachNums:
            roachNums=self.roachNums
        try: resAtten=kwargs.pop('resAtten')
        except KeyError: resAtten=None

        threadsToStart=[]
        for i, roach_i in enumerate(roachNums):
            roachArg = np.where(np.asarray(self.roachNums) == roach_i)[0][0]
            DAC_freqResolution = self.roaches[roachArg].roachController.params['dacSampleRate']/(self.roaches[roachArg].roachController.params['nDacSamplesPerCycle']*self.roaches[roachArg].roachController.params['nLutRowsToUse'])
            if self.roachThreads[roachArg].isRunning():
                print 'Roach '+str(roach_i)+' is busy'
            else: 
                freqFN = self.roaches[roachArg].config.get('Roach '+str(roach_i),'freqfile')
                LO = self.roaches[roachArg].config.getfloat('Roach '+str(roach_i),'lo_freq')
                maxBandwidth = self.roaches[0].roachController.params['dacSampleRate']
                try: startFreq = startFreqs[i]
                except: startFreq=LO - maxBandwidth/2.
                try: endFreq = endFreqs[i]
                except: endFreq=LO + maxBandwidth/2.
                #todo should the same list be used for each board if making?
                if makeNewFreqs:
                    freqs, LO, span = self.makeRandomFreqList(startFreq, endFreq,**kwargs)
                    #path=freqFN.rsplit('/',1)[0]
                    freqFN=self.outPath+'/'+self.outPrefix+'freq_WS_r'+str(roach_i)+'.txt'
                    self.saveFreqList(freqs, freqFN, resAtten=resAtten)
                    
                    self.roaches[roachArg].config.set('Roach '+str(roach_i),'freqfile',freqFN)
                    self.roaches[roachArg].config.set('Roach '+str(roach_i),'lo_freq',str(LO))
                else:
                    freqFN2=freqFN.rsplit('.',1)[0]+'_NEW.'+ freqFN.rsplit('.',1)[1]
                    try: freqData = np.loadtxt(freqFN2)
                    except IOError: freqData=np.loadtxt(freqFN)
                    if len(self.outPath)==0: self.outPath=freqFN.rsplit('/',1)[0]
                    resIDs = np.atleast_1d(freqData[:,0])
                    freqs = np.atleast_1d(freqData[:,1])
                    self.saveFreqList(freqs, freqFN2, resAtten=resAtten, resIDs=resIDs)
                    span = np.amax([freqs[0] - startFreq, endFreq - freqs[-1], np.amax(np.diff(freqs))])
                    span += 1.0*self.numoverlapPoints*DAC_freqResolution  # force at least 10 overlap points between tones

                
                #span=0.2E6
                lo_step = DAC_freqResolution
                self.roaches[roachArg].config.set('Roach '+str(roach_i),'sweeplostep',str(lo_step))
                self.roaches[roachArg].config.set('Roach '+str(roach_i),'sweeplospan',str(span))
                #if lo_step: self.roaches[roachArg].config.set('Roach '+str(roach_i),'sweeplostep',str(lo_step))
                if DACatten:
                    self.roaches[roachArg].config.set('Roach '+str(roach_i),'dacatten_start',str(DACatten))
                    self.roaches[roachArg].config.set('Roach '+str(roach_i),'dacatten_stop',str(DACatten))
                if ADCatten: self.roaches[roachArg].config.set('Roach '+str(roach_i),'adcatten',str(ADCatten))
                

                if self.debug:  #Initialize to sweep state to save
                    state = [RoachStateMachine.UNDEFINED]*RoachStateMachine.NUMCOMMANDS
                    state[:RoachStateMachine.SWEEP] = [RoachStateMachine.COMPLETED]*RoachStateMachine.SWEEP
                    self.roaches[roachArg].initializeToState(state)
                #self.roaches[roachArg].addCommands(RoachStateMachine.DEFINEDACLUT)
                self.roaches[roachArg].addCommands(RoachStateMachine.SWEEP)        # add command to roach queue
                threadsToStart.append(self.roachThreads[roachArg])

        for t in threadsToStart:
            t.start()
            self.execQApp+=1
        if self.execQApp>0:
            #print "Starting QApp"
            self.app.exec_()    #Start the QApplication so we can see signals on the QEventLoop
    
    def saveFreqList(self,freqs, outfilename='test.txt', resAtten=None, resIDs=None):
        if resAtten is None: resAtten=50
        attens=np.asarray([np.rint(resAtten*4.)/4.]*len(freqs))
        if resIDs is None: resIDs=np.asarray(range(len(freqs)))
        data = np.asarray([resIDs, freqs, attens]).T
        np.savetxt(outfilename, data, fmt="%4i %10.1f %4i")
    
    def makeRandomFreqSideband(self, startFreq, endFreq, nChannels, toneBandwidth, freqResolution):
        if nChannels <1:
            return np.asarray([],dtype=np.int)
        #if nChannels==1:
        #    return np.asarray([startFreq])
        avgSpacing = (endFreq - startFreq)/nChannels
        freqs = np.linspace(startFreq, endFreq, nChannels, False)+avgSpacing/2.
        freqs+= np.random.rand(len(freqs)) * avgSpacing - avgSpacing/2.
        
        #Correct doubles
        for arg in range(len(freqs)-1):
            if (freqs[arg+1] - freqs[arg]) <toneBandwidth:
                if arg==0: f_low =startFreq - toneBandwidth + freqResolution
                else: f_low = freqs[arg -1]
                if arg>=(len(freqs)-2): f_high = endFreq + toneBandwidth - freqResolution
                else: f_high = freqs[arg+2]
                
                f_spacing = (f_high - f_low) / 3.0
                if f_spacing>=toneBandwidth:       #push the two tones between f_high and f_low
                    freqs[arg]=f_low+f_spacing
                    freqs[arg+1]=f_low+2.0*f_spacing
                elif (f_high - f_low) /2.0 >= toneBandwidth:   #push one tone halfway between f_high and f_low
                    freqs[arg]=(f_high - f_low)/2.0
                    freqs[arg+1] = freqs[arg]
                else:                                       #remove both tones
                    freqs[arg] = f_low
                    freqs[arg+1]=f_low
        freqs=np.unique(freqs)
        
        return freqs
    
    def makeRandomFreqList(self, startFreq, endFreq, toneBandwidth=512.0E3, minNominalFreqSpacing=800.0E3):
        """
        This function makes a list of random frequencies in the frequency range specified
        
        INPUTS:
            startFreq - bandwidth to sweep
            endFreq - cannot sweep more than 2 GHz range
            toneBandwidth - minimum distance between freqs to avoid bandwidth overlap
            nominalFreqSpacing - nominal avg distance between freqs. May be forced to go larger to cover freq range
                                 Needs to be at least toneBandwidth+100.*freqResolution
        """
        #todo create a frequencylist object
        #todo move into frequencylist object

        assert endFreq >= startFreq, "Must use a positive Freq range"
        maxBandwidth = self.roaches[0].roachController.params['dacSampleRate']
        freqResolution = self.roaches[0].roachController.params['dacSampleRate']/(self.roaches[0].roachController.params['nDacSamplesPerCycle']*self.roaches[0].roachController.params['nLutRowsToUse'])
        #print 'freq res:', freqResolution
        #assert (endFreq - startFreq) <= maxBandwidth, "Must use range less than 2 GHz"
        if endFreq - startFreq >maxBandwidth:
            warnings.warn("Must do more than 1 sweep for full frequency range")
            endFreq = startFreq+maxBandwidth
        minNominalFreqSpacing = max(toneBandwidth+100.*freqResolution, minNominalFreqSpacing)
        
        #Determine number of tones needed
        nChannels=self.roaches[0].roachController.params['nChannels']
        avgSpacing = (endFreq - startFreq - toneBandwidth)/nChannels
        if avgSpacing < (minNominalFreqSpacing):
            avgSpacing = minNominalFreqSpacing
            nChannels = int ((endFreq - startFreq-toneBandwidth)/avgSpacing)
        if nChannels<=1:
            nChannels=1
            avgSpacing=0.
            
        LO = (startFreq + endFreq)/2.0 + float(nChannels==1)*(2.0*minNominalFreqSpacing+toneBandwidth/2.)
        maxBandwidth = min( maxBandwidth, max((endFreq - startFreq)+2.0*toneBandwidth, 3.0*toneBandwidth))

        #Make freqList
        freqs_low=self.makeRandomFreqSideband(startFreq+freqResolution, LO-toneBandwidth/2.-freqResolution, np.ceil(nChannels/2.), toneBandwidth, freqResolution)
        freqs_high=self.makeRandomFreqSideband(LO+toneBandwidth/2.+freqResolution, endFreq-freqResolution, np.floor(nChannels/2.), toneBandwidth, freqResolution)
        freqList = np.append(freqs_low, freqs_high)
        
        #freqList = np.rint(freqList).astype(np.int)
        maxSpan = np.amax([(endFreq - startFreq) - (freqList[-1] - freqList[0]), np.amax(np.diff(freqList))])
        maxSpan +=1.0*self.numoverlapPoints*freqResolution    #force at least 10 points overlap in sweep
        offset=freqList[0]-startFreq - maxSpan/2.
        freqList = freqList - offset
        LO = LO - offset
        return freqList, LO, maxSpan


if __name__ == "__main__":
    
    args = sys.argv[1:]
    defaultValues=None
    filePrefix=None
    if '-c' in args:
        indx = args.index('-c')
        defaultValues=args[indx+1]
        try: args = args[:indx]+args[indx+2:]
        except IndexError:args = args[:indx]
    if '-o' in args:
        indx = args.index('-o')
        filePrefix=args[indx+1]
        try: args = args[:indx]+args[indx+2:]
        except IndexError:args = args[:indx]
    roachNums = np.asarray(args, dtype=np.int)
    print defaultValues,roachNums, filePrefix
    
    debug=False
    #TODO make this into argparse args
    #todo error out if it cant find config file
    startFreqs=np.asarray([3.5E9]*len(roachNums))
    startFreqs[1::2]+=2.E9
    stopFreqs=np.asarray([5.5E9]*len(roachNums))
    stopFreqs[1::2]+=2.E9

    #TODO make program merge the


    #startFreqs = [5.5E9]
    #stopFreqs=[7.5E9]
    digWS = DigitalWideSweep(roachNums, defaultValues,filePrefix,debug=debug)
    digWS.startWS(roachNums=None, startFreqs=startFreqs, endFreqs=stopFreqs,DACatten=None, ADCatten=None,resAtten=65,makeNewFreqs=not debug)



#TODO fix console logging!!!!
