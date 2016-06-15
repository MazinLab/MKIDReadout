"""
Author:    Alex Walter
Date:      April 25, 2016
Firmware:  pgbe0_2016_Feb_19_2018.fpg

This class is for setting and reading LUTs, registers, and other memory components in the ROACH2 Virtex 6 FPGA using casperfpga tools.
It's also the IO for the ADC/DAC board's Virtex 7 FPGA through the ROACH2

NOTE: All freqencies are considered positive. A negative frequency can be asserted by the aliased signal of large positive frequency (by adding sample rate). This makes things easier for coding since I can check valid frequencies > 0 and also for calculating which fftBin a frequency resides in (see generateFftChanSelection()). 


Example usage:
    # Collect MKID info
    nFreqs=1024
    loFreq = 5.e9
    spacing = 2.e6
    freqList = np.arange(loFreq-nFreqs/2.*spacing,loFreq+nFreqs/2.*spacing,spacing)
    freqList+=np.random.uniform(-spacing,spacing,nFreqs)
    freqList = np.sort(freqList)
    attenList = np.random.randint(23,33,nFreqs)
    
    # Talk to Roach
    roach_0 = FpgaControls(ip, params, True, True)
    roach_0.setLOFreq(loFreq)
    roach_0.generateResonatorChannels(freqList)
    roach_0.generateFftChanSelection()
    roach_0.generateDacComb(freqList=None, resAttenList=attenList, globalDacAtten=17)
    roach_0.generateDdsTones()
    
    roach_0.loadChanSelection()
    roach_0.loadDacLUT()




List of Functions:
    __init__ -                      Connects to Roach2 FPGA, sets the delay between the dds lut and the end of the fft block
    generateResonatorChannels -     Figures out which stream:channel to assign to each resonator frequency
    generateFftChanSelection -      Assigns fftBins to each steam:channel
    loadSingleChanSelection -       Loads a channel for each stream into the channel selector LUT
    loadChanSelection -             Loops through loadSingleChanSelection()
    setLOFreq -                     Defines LO frequency as an attribute, self.LOFreq
    generateTones -                 Returns a list of I,Q time series for each frequency provided
    generateDacComb -               Returns a single I,Q time series representing the DAC freq comb
    loadDacLut -                    Loads the freq comb from generateDacComb() into the LUT
    generateDdsTones -              Defines interweaved tones for dds
    

    
List of useful class attributes:
    ip -                            ip address of roach2
    params -                        Dictionary of parameters
    freqList -                      List of resonator frequencies
    freqChannels -                  2D array of frequencies. Each column is the a stream and each row is a channel. 
                                    If uneven number of frequencies this array is padded with -1's
    fftBinIndChannels -             2D array of fftBin indices corresponding to the frequencies/streams/channels in freqChannels. freq=-1 maps to fftBin=0.
    dacPhaseList -                  List of the most recent relative phases used for generating DAC frequency comb
    dacScaleFactor -                Scale factor for frequency comb to scale the sum of tones onto the DAC's dynamic range. 
                                    Careful, this is 1/scaleFactor we defined for ARCONS templar
    dacQuantizedFreqList -          List of frequencies used to define DAC frequency comb. Quantized to DAC digital limits
    dacFreqComb -                   Complex time series signal used for DAC frequency comb. 
    LOFreq -                        LO frequency of IF board
    ddsQuantizedFreqList -          2D array of frequencies shaped like freqChannels. Quantized to dds digital limits
    ddsPhaseList -                  2D array of frequencies shaped like freqChannels. Used to rotate loops.
    


TODO:
    uncomment self.fpgs, DDS Shift in __init__
    uncomment register writes in loadSingleChanSelection()
    uncomment register writes in loadDacLut()
    add code for setting LO freq in loadLOFreq()
    write code collecting data from ADC
"""

import sys,os,time,struct
import warnings, inspect
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import casperfpga
from readDict import readDict       #Part of the ARCONS-pipeline/util

class Roach2Controls:

    def __init__(self, ip, paramFile, verbose=False, debug=False):
        '''
        Input:
            ip - ip address string of ROACH2
            paramFile - param object or directory string to dictionary containing important info
            verbose - show print statements
            debug - Save some things to disk for debugging
        '''
        self.verbose=verbose
        self.debug=debug
        
        self.ip = ip
        try:
            self.params = readDict()             
            self.params.readFromFile(paramFile)
        except TypeError:
            self.params = paramFile
        
        if debug and not os.path.exists(self.params['debugDir']):
            os.makedirs(self.params['debugDir'])
        '''
        self.fpga = casperfpga.katcp_fpga.KatcpFpga(ip,timeout=50.)
        time.sleep(1)
        if not self.fpga.is_running():
            print 'Firmware is not running. Start firmware, calibrate, and load wave into qdr first!'
            #sys.exit(0)
    
        self.fpga.get_system_information()

        #set the delay between the dds lut and the end of the fft block (firmware dependent)
        self.fpga.write_int(self.params['ddsShift_reg'],self.params['ddsShift'])
        '''
        
        #Some more parameters
        self.freqPadValue = -1      # pad frequency lists so that we have a multiple of number of streams
        self.fftBinPadValue = 0     # pad fftBin selection with fftBin 0
        self.ddsFreqPadValue = -1
        
        
    def generateDdsTones(self, freqChannels=None, fftBinIndChannels=None, phaseList=None):
        """
        Create and interweave dds frequencies
        
        Call setLOFreq(), generateResonatorChannels(), generateFftChanSelection() first.
        
        INPUT:
            freqChannels - Each column contains the resonantor frequencies in a single stream. The row index is the channel number. It's padded with -1's. 
                           Made by generateResonatorChannels(). If None, use self.freqChannels
            fftBinIndChannels - Same shape as freqChannels but contains the fft bin index. Made by generateFftChanSelection(). If None, use self.fftBinIndChannels
            phaseList - Same shape as freqChannels. Contains phase offsets (0 to 2Pi) for dds sampling. If None, set all to zero
        
        OUTPUT:
            dictionary with following keywords
            'iStreamList' - 2D array. Each row is an interweaved list of i values for a single stream. 
            'qStreamList' - q values
            'quantizedFreqList' - 2d array of dds frequencies. (same shape as freqChannels) Padded with self.ddsFreqPadValue
            'phaseList' - 2d array of phases for each frequency (same shape as freqChannels) Padded with 0's
        """
        #Interpret Inputs
        if freqChannels is None:
            freqChannels = self.freqChannels
        if len(np.ravel(freqChannels))>self.params['nChannels']:
            raise ValueError("Too many freqs provided. Can only accommodate "+str(self.params['nChannels'])+" resonators")
        self.freqChannels = freqChannels
        if fftBinIndChannels is None:
            fftBinIndChannels = self.fftBinIndChannels
        if len(np.ravel(fftBinIndChannels))>self.params['nChannels']:
            raise ValueError("Too many freqs provided. Can only accommodate "+str(self.params['nChannels'])+" resonators")
        self.fftBinIndChannels = fftBinIndChannels
        if phaseList is None:
            phaseList = np.zeros(np.asarray(freqChannels).shape)
    
        # quantize resonator tones to dds resolution
        if not hasattr(self,'LOFreq'):
            raise ValueError("Need to set LO freq by calling setLOFreq()")
        dacFreqList = freqChannels-self.LOFreq
        dacFreqList[np.where(dacFreqList<0.)] += self.params['dacSampleRate']  #For +/- freq
        dacFreqResolution = self.params['dacSampleRate']/(self.params['nDacSamplesPerCycle']*self.params['nLutRowsToUse'])
        dacQuantizedFreqList = np.round(dacFreqList/dacFreqResolution)*dacFreqResolution
        fftBinSpacing = self.params['dacSampleRate']/self.params['nFftBins']
        fftBinCenterFreqList = fftBinIndChannels*fftBinSpacing
        ddsFreqList = dacQuantizedFreqList - fftBinCenterFreqList
        ddsSampleRate = self.params['nDdsSamplesPerCycle'] * self.params['fpgaClockRate'] / self.params['nCyclesToLoopToSameChannel']
        ddsFreqList[np.where(ddsFreqList<0)]+=ddsSampleRate     # large positive frequencies are aliased back to negative freqs
        nDdsSamples = self.params['nDdsSamplesPerCycle']*self.params['nQdrRows']/self.params['nCyclesToLoopToSameChannel']
        ddsFreqResolution = 1.*ddsSampleRate/nDdsSamples
        ddsQuantizedFreqList = np.round(ddsFreqList/ddsFreqResolution)*ddsFreqResolution
        print ddsQuantizedFreqList
        ddsQuantizedFreqList[np.where(freqChannels<0)] = self.ddsFreqPadValue     # Pad excess frequencies with 0
        print ddsQuantizedFreqList
        self.ddsQuantizedFreqList = ddsQuantizedFreqList    # Some of these are negative frequencies
        
        # For each Stream, generate tones and interweave time streams for the dds time multiplexed multiplier
        nStreams = int(self.params['nChannels']/self.params['nChannelsPerStream'])        #number of processing streams. For Gen 2 readout this should be 4
        iStreamList = []
        qStreamList = []
        for i in range(nStreams):
            # generate individual tone time streams
            toneParams={
                'freqList': ddsQuantizedFreqList[:,i][np.where(dacQuantizedFreqList[:,i]>0)],
                'nSamples': nDdsSamples,
                'sampleRate': ddsSampleRate,
                'amplitudeList': None,  #defaults to 1
                'phaseList': phaseList[:,i][np.where(dacQuantizedFreqList[:,i]>0)]}
            toneDict = self.generateTones(**toneParams)
            
            #scale amplitude to number of bits in memory and round
            nBitsPerSampleComponent = self.params['nBitsPerDdsSamplePair']/2
            maxValue = int(np.round(2**(nBitsPerSampleComponent - 1)-1))       # 1 bit for sign
            iValList = np.array(np.round(toneDict['I']*maxValue),dtype=np.int)
            qValList = np.array(np.round(toneDict['Q']*maxValue),dtype=np.int)
            
            #interweave the values such that we have two samples from freq 0 (row 0), two samples from freq 1, ... to freq 256. Then have the next two samples from freq 1 ...
            freqPad = np.zeros((self.params['nChannelsPerStream'] - len(toneDict['quantizedFreqList']),nDdsSamples))
            iValList = np.append(iValList,freqPad,0)    #First pad with missing resonators
            qValList = np.append(qValList,freqPad,0)
            iValList = np.reshape(iValList,(self.params['nChannelsPerStream'],-1,self.params['nDdsSamplesPerCycle']))
            qValList = np.reshape(qValList,(self.params['nChannelsPerStream'],-1,self.params['nDdsSamplesPerCycle']))
            iValList = np.swapaxes(iValList,0,1)
            qValList = np.swapaxes(qValList,0,1)
            iValues = iValList.flatten('C')
            qValues = qValList.flatten('C')
            
            # put into list
            iStreamList.append(iValues)
            qStreamList.append(qValues)
            phaseList[:len(toneDict['phaseList']),i] = toneDict['phaseList']    # We need this if we let self.generateTones() choose random phases
        
        self.ddsPhaseList = phaseList
        
        return {'iStreamList':iStreamList, 'qStreamList':qStreamList, 'quantizedFreqList':ddsQuantizedFreqList, 'phaseList':phaseList}
    
    
    
    def writeBram(self, memName, valuesToWrite, start=0, nRows=2**10):
        """
        format values and write them to bram
        
        
        """
        nBytesPerSample = 8
        formatChar = 'Q'
        memValues = np.array(valuesToWrite,dtype=np.uint64) #cast signed values
        nValues = len(valuesToWrite)
        toWriteStr = struct.pack('>{}{}'.format(nValues,formatChar),*memValues)
        self.fpga.blindwrite(memName,toWriteStr,start)
        
    def writeQdr(self, memName, valuesToWrite, start=0, bQdrFlip=True, nQdrRows=2**20):
        """
        format and write 64 bit values to qdr
        
        INPUTS:
        """
        nBytesPerSample = 8
        formatChar = 'Q'
        memValues = np.array(valuesToWrite,dtype=np.uint64) #cast signed values
        nValues = len(valuesToWrite)
        if bQdrFlip: #For some reason, on Roach2 with the current qdr calibration, the 64 bit word seen in firmware
            #has the first and second 32 bit chunks swapped compared to the 64 bit word sent by katcp, so to accommodate
            #we swap those chunks here, so they will be in the right order in firmware
            mask32 = int('1'*32,2)
            memValues = (memValues >> 32)+((memValues & mask32) << 32)
            #Unfortunately, with the current qdr calibration, the addresses in katcp and firmware are shifted (rolled) relative to each other
            #so to compensate we roll the values to write here
            memValues = np.roll(memValues,-1)
        toWriteStr = struct.pack('>{}{}'.format(nValues,formatChar),*memValues)
        self.fpga.blindwrite(memName,toWriteStr,start)
    
    def formatWaveForMem(self, iVals, qVals, nBitsPerSamplePair=24, nSamplesPerCycle=8, nMems=3, nBitsPerMemRow=64, earlierSampleIsMsb=True):
        """
        put together IQ values from tones to be loaded to a firmware memory LUT
        
        INPUTS:
            iVals - time series of I values
            qVals - 
            
        """
        nBitsPerSampleComponent = nBitsPerSamplePair / 2
        #I vals and Q vals are 12 bits, combine them into 24 bit vals
        iqVals = (iVals << nBitsPerSampleComponent) + qVals
        iqRows = np.reshape(iqVals,(-1,nSamplesPerCycle))
        #we need to set dtype to object to use python's native long type
        colBitShifts = nBitsPerSamplePair*(np.arange(nSamplesPerCycle,dtype=object))
        if earlierSampleIsMsb:
            #reverse order so earlier (more left) columns are shifted to more significant bits
            colBitShifts = colBitShifts[::-1]
        
        iqRowVals = np.sum(iqRows<<colBitShifts,axis=1) #shift each col by specified amount, and sum each row
        #Now we have 2**20 row values, each is 192 bits and contain 8 IQ pairs 
        #next we divide these 192 bit rows into three 64-bit qdr rows

        #Mem0 has the most significant bits
        memRowBitmask = int('1'*nBitsPerMemRow,2)
        memMaskShifts = nBitsPerMemRow*np.arange(nMems,dtype=object)[::-1]
        #now do bitwise_and each value with the mask, and shift back down
        memRowVals = (iqRowVals[:,np.newaxis] >> memMaskShifts) & memRowBitmask

        #now each column contains the 64-bit qdr values to be sent to a particular qdr
        return memRowVals
    
    def loadDacLUT(self, combDict=None):
        """
        Load frequency comb in DAC look up tables
        
        Call generateDacComb() first
        
        INPUTS:
            combDict - return value from generateDacComb(). If None, it trys to gather information from attributes
        """
        if combDict is None:
            try:
                combDict = {'I':np.real(self.dacFreqComb).astype(np.int), 'Q':np.imag(self.dacFreqComb).astype(np.int), 'quantizedFreqList':self.dacQuantizedFreqList}
            except AttributeError:
                print "Run generateDacComb() first!"
                raise

        #Format comb for onboard memory
        dacMemDict={
            'iVals':combDict['I'],
            'qVals':combDict['Q'],
            'nBitsPerSamplePair':self.params['nBitsPerSamplePair'],
            'nSamplesPerCycle':self.params['nDacSamplesPerCycle'],
            'nMems':len(self.params['dacMemNames_reg']),
            'nBitsPerMemRow':self.params['nBytesPerMemSample']*8,
            'earlierSampleIsMsb':True}
        memVals = self.formatWaveForMem(**dacMemDict)
        
        if self.debug:
            np.savetxt(self.params['debugDir']+'dacFreqs.txt', combDict['quantizedFreqList']/10**6., fmt='%3.11f', header="Array of DAC frequencies [MHz]")
        
        #Write data to LUTs
        '''
        self.fpga.write_int(self.params['start_reg'],0) #do not read from qdr while writing
        time.sleep(.1)
        self.fpga.write_int(self.params['nDacLutRows_reg'],self.params['nLutRowsToUse'])
        memType = self.params['memType']
        memNames = self.params['dacMemNames_reg']
        for iMem in xrange(len(memNames)):
            if memType == 'qdr':
                print 'writeQDR: ',memNames[iMem]
                #self.writeQdr(memName=memNames[iMem],valuesToWrite=memVals[:,iMem],bQdrFlip=True)
            elif memType == 'bram':
                print 'writeBram: ',memNames[iMem]
                #self.writeBram(memName=memNames[iMem],valuesToWrite=memVals[:,iMem])
        time.sleep(.5)
        self.fpga.write_int(self.params['start_reg'],1)
        time.sleep(.5)
        '''
    
    def setLOFreq(self,LOFreq):
        self.LOFreq = LOFreq
    
    def loadLOFreq(self,LOFreq=None):
        if LOFreq is None:
            LOFreq = self.LOFreq
        
        # load into IF board
        pass
    
    def generateDacComb(self, freqList=None, resAttenList=None, globalDacAtten = 0, phaseList=None, dacScaleFactor=None):
        """
        Creates DAC frequency comb by adding many complex frequencies together with specified relative amplitudes and phases.
        
        Note: If dacScaleFactor=self.dacScaleFactor this should keep the power going through MKIDs the same regardless of changes to globalDacAtten.
        ie. if the global attenuation is decreased by 1 dB (stronger), the amplitude of the signals in the DAC LUT is decreased by 1 dB (weaker)
        
        If dacScaleFactor is None then it scales the largest value to the largest allowed DAC value. There's a fudge factor of .25 dB hardcoded in so we aren't exactly at the limit
        
        INPUTS:
            freqList - list of all resonator frequencies. If None, use self.freqList
            resAttenList - list of attenuation values (dB) for each resonator. If None, use 0's
            globalDacAtten - (int) global attenuation for entire DAC
            dacPhaseList - list of phases for each complex signal. If None, generates random phases. Old phaseList is under self.dacPhaseList
            dacScaleFactor - scale factor to fit signal amplitudes into DAC dynamic range. Use self.dacScaleFactor or None to automatically create one
            
        OUTPUTS:
            dictionary with keywords
            I - I(t) values for frequency comb [signed 32-bit integers]
            Q - Q(t)
            quantizedFreqList - list of frequencies after digitial quantiziation
        """
        # Interpret Inputs
        if freqList is None:
            freqList=self.freqList
        if len(freqList)>self.params['nChannels']:
            warnings.warn("Too many freqs provided. Can only accommodate "+str(self.params['nChannels'])+" resonators")
            freqList = freqList[:self.params['nChannels']]
        self.freqList = np.ravel(freqList)
        if resAttenList is None:
            warnings.warn("Individual resonator attenuations assumed to be 0")
            resAttenList=np.zeros(len(freqList))
        if len(resAttenList)>self.params['nChannels']:
            warnings.warn("Too many attenuations provided. Can only accommodate "+str(self.params['nChannels'])+" resonators")
            resAttenList = resAttenList[:self.params['nChannels']]
        resAttenList = np.ravel(resAttenList)
        if len(freqList) != len(resAttenList):
            raise ValueError("Need exactly one attenuation value for each resonant frequency!")
        if (phaseList is not None) and len(freqList) != len(phaseList):
            raise ValueError("Need exactly one phase value for each resonant frequency!")
        
        # Calculate relative amplitudes. 0 point is tied to globalDacAtten instead of the min attenuation
        #amplitudeList = 10**(-(globalDacAtten + resAttenList)/20.)
        amplitudeList = 10**((globalDacAtten-resAttenList)/20.)
        
        # Calculate nSamples and sampleRate
        nSamples = self.params['nDacSamplesPerCycle']*self.params['nLutRowsToUse']
        sampleRate = self.params['dacSampleRate']
        
        # Calculate resonator frequencies for DAC
        if not hasattr(self,'LOFreq'):
            raise ValueError("Need to set LO freq by calling setLOFreq()")
        dacFreqList = self.freqList-self.LOFreq
        dacFreqList[np.where(dacFreqList<0.)] += self.params['dacSampleRate']  #For +/- freq
        
        # Generate and add up individual tone time series. Then scale to DAC dynamic range
        if self.verbose:
            print 'Generating DAC comb...'
        toneDict = self.generateTones(dacFreqList, nSamples, sampleRate, amplitudeList, phaseList)
        self.dacQuantizedFreqList = toneDict['quantizedFreqList']
        self.dacPhaseList = toneDict['phaseList']
        iValues = np.sum(toneDict['I'],axis=0)
        qValues = np.sum(toneDict['Q'],axis=0)
        nBitsPerSampleComponent = self.params['nBitsPerSamplePair']/2
        maxValue = int(np.round(self.params['dynamicRange']*2**(nBitsPerSampleComponent - 1)-1))       # 1 bit for sign
        if dacScaleFactor is None:
            highestVal = np.max((np.abs(iValues).max(),np.abs(qValues).max()))
            scaleFudgeFactor = 10**(-0.25/20.)    # don't scale exactly to maxValue, leave .25 dB of wiggle room. 
            self.dacScaleFactor = maxValue/highestVal*scaleFudgeFactor
        else:
            self.dacScaleFactor=dacScaleFactor
        iValues = np.array(np.round(self.dacScaleFactor * iValues),dtype=np.int)
        qValues = np.array(np.round(self.dacScaleFactor * qValues),dtype=np.int)

        highestVal = np.max((np.abs(iValues).max(),np.abs(qValues).max()))
        if 1.0*maxValue/highestVal > 10**((1)/20.):
            # all amplitudes in DAC less than 1 dB below max allowed by dynamic range
            warnings.warn("DAC Dynamic range not fully utilized. Increase global attenuation by: "+str(np.floor(20.*np.log10(1.0*maxValue/highestVal)))+' dB')
        elif maxValue < highestVal:
            raise ValueError("Not enough dynamic range in DAC! Try dacScaleFactor=None")
        expectedHighestVal_sig = scipy.special.erfinv((len(iValues)-0.1)/len(iValues))*np.sqrt(2.)   # 10% of the time there should be a point this many sigmas higher than average
        if highestVal > expectedHighestVal_sig*np.max((np.std(iValues),np.std(qValues))):
            warnings.warn("The freq comb's relative phases may have added up sub-optimally. You should calculate new random phases")
        
        self.dacFreqComb = iValues + 1j*qValues

        if self.verbose:
            print '\tUsing '+str(1.0*highestVal/maxValue*100)+' percent of DAC dynamic range'
            print '\thighest: '+str(highestVal)+' out of '+str(maxValue)
            print '\tsigma_I: '+str(np.std(iValues))+' sigma_Q: '+str(np.std(qValues))
            print '\tLargest val_I: '+str(1.0*np.abs(iValues).max()/np.std(iValues))+' sigma. Largest val_Q: '+str(1.0*np.abs(qValues).max()/np.std(qValues))+' sigma.'
            print '\tExpected val: '+str(expectedHighestVal_sig)+' sigmas'
            print '\tdacScaleFactor: '+str(self.dacScaleFactor)
            print '...Done!'
        
        if self.debug:
            plt.figure()
            plt.plot(iValues)
            plt.plot(qValues)
            std_i = np.std(iValues)
            std_q = np.std(qValues)
            plt.axhline(y=std_i,color='k')
            plt.axhline(y=2*std_i,color='k')
            plt.axhline(y=3*std_i,color='k')
            plt.axhline(y=expectedHighestVal_sig*std_i,color='r')
            plt.axhline(y=expectedHighestVal_sig*std_q,color='r')
            
            plt.figure()
            plt.hist(iValues,1000)
            plt.hist(qValues,1000)
            x_gauss = np.arange(-maxValue,maxValue,maxValue/2000.)
            i_gauss = len(iValues)/(std_i*np.sqrt(2.*np.pi))*np.exp(-x_gauss**2/(2.*std_i**2.))
            q_gauss = len(qValues)/(std_q*np.sqrt(2.*np.pi))*np.exp(-x_gauss**2/(2.*std_q**2.))
            plt.plot(x_gauss,i_gauss)
            plt.plot(x_gauss,q_gauss)
            plt.axvline(x=std_i,color='k')
            plt.axvline(x=2*std_i,color='k')
            plt.axvline(x=3*std_i,color='k')
            plt.axvline(x=expectedHighestVal_sig*std_i,color='r')
            plt.axvline(x=expectedHighestVal_sig*std_q,color='r')
            
            plt.figure()
            sig = np.fft.fft(self.dacFreqComb)
            sig_freq = np.fft.fftfreq(len(self.dacFreqComb),1./self.params['dacSampleRate'])
            plt.plot(sig_freq, np.real(sig),'b')
            plt.plot(sig_freq, np.imag(sig),'g')
            for f in self.dacQuantizedFreqList:
                x_f=f
                if f > self.params['dacSampleRate']/2.:
                    x_f=f-self.params['dacSampleRate']
                plt.axvline(x=x_f, ymin=np.amin(np.real(sig)), ymax = np.amax(np.real(sig)), color='r')
            #plt.show()
            
        return {'I':iValues,'Q':qValues,'quantizedFreqList':self.dacQuantizedFreqList}
        
    
    def generateTones(self, freqList, nSamples, sampleRate, amplitudeList, phaseList):
        """
        Generate a list of complex signals with amplitudes and phases specified and frequencies quantized
        
        INPUTS:
            freqList - list of resonator frequencies
            nSamples - Number of time samples
            sampleRate - Used to quantize the frequencies
            amplitudeList - list of amplitudes. If None, use 1.
            phaseList - list of phases. If None, use random phase
        
        OUTPUTS:
            dictionary with keywords
            I - each element is a list of I(t) values for specific freq
            Q - Q(t)
            quantizedFreqList - list of frequencies after digitial quantiziation
            phaseList - list of phases for each frequency
        """
        if amplitudeList is None:
            amplitudeList = np.asarray([1.]*len(freqList))
        if phaseList is None:
            phaseList = np.random.uniform(0,2.*np.pi,len(freqList))
        if len(freqList) != len(amplitudeList) or len(freqList) != len(phaseList):
            raise ValueError("Need exactly one phase and amplitude value for each resonant frequency!")
        
        # Quantize the frequencies to their closest digital value
        freqResolution = sampleRate/nSamples
        quantizedFreqList = np.round(freqList/freqResolution)*freqResolution
        
        # generate each signal
        iValList = []
        qValList = []
        dt = 1. / sampleRate
        t = dt*np.arange(nSamples)
        for i in range(len(quantizedFreqList)):
            phi = 2.*np.pi*quantizedFreqList[i]*t
            expValues = amplitudeList[i]*np.exp(1.j*(phi+phaseList[i]))
            iValList.append(np.real(expValues))
            qValList.append(np.imag(expValues))
        
        '''
        if self.debug:
            plt.figure()
            for i in range(len(quantizedFreqList)):
                plt.plot(iValList[i])
                plt.plot(qValList[i])
            #plt.show()
        '''
        return {'I':np.asarray(iValList),'Q':np.asarray(qValList),'quantizedFreqList':quantizedFreqList,'phaseList':phaseList}
        
        
    def generateResonatorChannels(self, freqList,order='F'):
        """
        Algorithm for deciding which resonator frequencies are assigned to which stream and channel number.
        This is used to define the dds LUTs and calculate the fftBin index for each freq to set the appropriate chan_sel block
        
        Try to evenly distribute the given frequencies into each stream
        
        INPUTS:
            freqList - list of resonator frequencies (Assumed sequential but doesn't really matter)
            order - 'F' places sequential frequencies into a single stream
                    'C' places sequential frequencies into the same channel number
        OUTPUTS:
            self.freqChannels - Each column contains the resonantor frequencies in a single stream. The row index is the channel number. It's padded with -1's. 
        """
        #Interpret inputs...
        if order not in ['F','C','A']:  #if invalid, grab default value
            args,__,__,defaults = inspect.getargspec(roach.hashPixelStreamChannel)
            order = defaults[args.index('order')-len(args)]
            if self.verbose: print "Invalid 'order' parameter for generateResonatorChannels(). Changed to default: "+str(order)
        if len(freqList)>self.params['nChannels']:
            warnings.warn("Too many freqs provided. Can only accommodate "+str(self.params['nChannels'])+" resonators")
            freqList = freqList[:self.params['nChannels']]
        self.freqList = np.ravel(freqList)
        self.freqChannels = self.freqList
        
        #Pad with freq = -1 so that freqChannels's length is a multiple of nStreams
        nStreams = int(self.params['nChannels']/self.params['nChannelsPerStream'])        #number of processing streams. For Gen 2 readout this should be 4
        padNum = (nStreams - (len(self.freqChannels) % nStreams))%nStreams  # number of empty elements to pad
        padValue = self.freqPadValue   #pad with freq=-1
        if order == 'F':
            for i in range(padNum):
                ind = len(self.freqChannels)-i*np.ceil(len(self.freqChannels)*1.0/nStreams)
                self.freqChannels=np.insert(self.freqChannels,int(ind),padValue)
        else:
            self.freqChannels = np.append(self.freqChannels, [padValue]*(padNum))
        
        #Split up to assign channel numbers
        self.freqChannels = np.reshape(self.freqChannels,(-1,nStreams),order)
        return self.freqChannels
        
        
        
    def generateFftChanSelection(self,freqChannels=None):
        '''
        This calculates the fftBin index for each resonant frequency and arranges them by stream and channel.
        Used by channel selector block
        Call setLOFreq() and generateResonatorChannels() first.
        
        INPUTS (optional):
            freqChannels - 2D array of frequencies where each column is the a stream and each row is a channel. If freqChannels isn't given then try to grab it from attribute. 
        
        OUTPUTS:
            self.fftBinIndChannels - Array with each column containing the fftbin index of a single stream. The row index is the channel number
            
        '''
        if freqChannels is None:
            try:
                freqChannels = self.freqChannels
            except AttributeError:
                print "Run generateResonatorChannels() first!"
                raise
        freqChannels = np.asarray(freqChannels)
        
        #The frequencies seen by the fft block are actually from the DAC, up/down converted by the IF board, and then digitized by the ADC
        dacFreqChannels = (freqChannels-self.LOFreq)
        dacFreqChannels[np.where(dacFreqChannels<0)]+=self.params['dacSampleRate']
        freqResolution = self.params['dacSampleRate']/(self.params['nDacSamplesPerCycle']*self.params['nLutRowsToUse'])
        dacQuantizedFreqChannels = np.round(dacFreqChannels/freqResolution)*freqResolution
        
        #calculate fftbin index for each freq
        binSpacing = self.params['dacSampleRate']/self.params['nFftBins']
        genBinIndex = dacQuantizedFreqChannels/binSpacing
        self.fftBinIndChannels = np.round(genBinIndex)
        self.fftBinIndChannels[np.where(freqChannels<0)]=self.fftBinPadValue      # empty channels have freq=-1. Assign this to fftBin=0
        
        self.fftBinIndChannels = self.fftBinIndChannels.astype(np.int)
        return self.fftBinIndChannels

        
    def loadChanSelection(self,fftBinIndChannels=None):
        """
        Loads fftBin indices to all channels (in each stream), to configure chan_sel block in firmware on FPGA
        Call generateFftChanSelection() first

        
        INPUTS (optional):
            fftBinIndChannels - Array with each column containing the fftbin index of a single stream. The row is the channel number
        """
        if fftBinIndChannels is None:
            try:
                fftBinIndChannels = self.fftBinIndChannels
            except AttributeError:
                print "Run generateFftChanSelection() first!"
                raise
        
        if self.verbose: print 'Configuring chan_sel block...\n\tCh: Stream'+str(range(len(fftBinIndChannels[0])))
        for row in range(len(fftBinIndChannels)):
            if row > self.params['nChannelsPerStream']:
                warnings.warn("Too many freqs provided. Can only accommodate "+str(self.params['nChannels'])+" resonators")
                break
            self.loadSingleChanSelection(selBinNums=fftBinIndChannels[row],chanNum=row)
        if self.verbose: print '...Done!'
        if self.debug:
            np.savetxt(self.params['debugDir']+'freqChannels.txt', self.freqChannels/10**9.,fmt='%2.25f',header="2D Array of MKID frequencies [GHz]. \nEach column represents a stream and each row is a channel")
            np.savetxt(self.params['debugDir']+'fftBinIndChannels.txt', self.fftBinIndChannels,fmt='%8i',header="2D Array of fftBin Indices. \nEach column represents a stream and each row is a channel")
        
    def loadSingleChanSelection(self,selBinNums,chanNum=0):
        """
        Assigns bin numbers to a single channel (in each stream), to configure chan_sel block
        Used by loadChanSelection()

        INPUTS:
            selBinNums: array of bin numbers (for each stream) to be assigned to chanNum (4 element int array for Gen 2 firmware)
            chanNum: the channel number to be assigned
        """
        nStreams = int(self.params['nChannels']/self.params['nChannelsPerStream'])        #number of processing streams. For Gen 2 readout this should be 4
        if selBinNums is None or len(selBinNums) != nStreams:
            raise TypeError,'selBinNums must have number of elements matching number of streams in firmware'
        '''
        self.fpga.write_int(self.params['chanSelLoad_reg'],0) #set to zero so nothing loads while we set other registers.

        #assign the bin number to be loaded to each stream
        for i in range(nStreams):
            self.fpga.write_int(self.params['chanSel_regs'][i],selBinNums[i])
        time.sleep(.1)
        
        #in the register chan_sel_load, the lsb initiates the loading of the above bin numbers into memory
        #the 8 bits above the lsb indicate which channel is being loaded (for all streams)
        loadVal = (chanNum << 1) + 1
        self.fpga.write_int(self.params['chanSelLoad_reg'],loadVal)
        time.sleep(.1) #give it a chance to load

        self.fpga.write_int(self.params['chanSelLoad_reg'],0) #stop loading
        '''
        if self.verbose: print '\t'+str(chanNum)+': '+str(selBinNums)

if __name__=='__main__':
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip='10.0.0.112'
    if len(sys.argv) > 2:
        params = sys.argv[2]
    else:
        params=os.getenv('HOME', default="/home/abwalter")+'/MkidDigitalReadout/DataReadout/ChannelizerControls/DarknessFpga.param'
    print ip
    print params

    #warnings.filterwarnings('error')
    #freqList = [7.32421875e9, 8.e9, 9.e9, 10.e9,11.e9,12.e9,13.e9,14.e9,15e9,16e9,17.e9,18.e9,19.e9,20.e9,21.e9,22.e9,23.e9]
    nFreqs=17
    loFreq = 5.e9
    spacing = 2.e6
    freqList = np.arange(loFreq-nFreqs/2.*spacing,loFreq+nFreqs/2.*spacing,spacing)
    freqList+=np.random.uniform(-spacing,spacing,nFreqs)
    freqList = np.sort(freqList)
    attenList = np.random.randint(23,33,nFreqs)
    
    #attenList = attenList[np.where(freqList > loFreq)]
    #freqList = freqList[np.where(freqList > loFreq)]
    
    roach_0 = FpgaControls(ip, params, True, True)
    roach_0.setLOFreq(loFreq)
    roach_0.generateResonatorChannels(freqList)
    roach_0.generateFftChanSelection()
    roach_0.loadChanSelection()
    
    
    
    #roach_0.generateDacComb(freqList, attenList, 17)
    #print roach_0.phaseList
    #print 10**(-0.25/20.)
    #roach_0.generateDacComb(freqList, attenList, 17, phaseList = roach_0.phaseList, dacScaleFactor=roach_0.dacScaleFactor*10**(-3./20.))
    #roach_0.generateDacComb(freqList, attenList, 20, phaseList = roach_0.phaseList, dacScaleFactor=roach_0.dacScaleFactor)
    #roach_0.loadDacLUT()
    
    roach_0.generateDdsTones()
    if roach_0.debug: plt.show()
    

