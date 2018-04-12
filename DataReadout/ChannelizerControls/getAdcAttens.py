import os, sys
import numpy as np
import matplotlib.pyplot as plt
from Roach2Controls import Roach2Controls

#TODO: error check threshold on max to 3.5 RMS at a low RMS value

def snapZdok(fpga,nRolls=0):
    snapshotNames = fpga.snapshots.names()

    #fpga.write_int('trig_qdr',0)#initialize trigger
    fpga.write_int('adc_in_trig',0)
    for name in snapshotNames:
        fpga.snapshots[name].arm(man_valid=False,man_trig=False)

    time.sleep(.1)
    #fpga.write_int('trig_qdr',1)#trigger snapshots
    fpga.write_int('adc_in_trig',1)
    time.sleep(.1) #wait for other trigger conditions to be met, and fill buffers
    #fpga.write_int('trig_qdr',0)#release trigger
    fpga.write_int('adc_in_trig',0)
    
    adcData0 = fpga.snapshots['adc_in_snp_cal0_ss'].read(timeout=5,arm=False)['data']
    adcData1 = fpga.snapshots['adc_in_snp_cal1_ss'].read(timeout=5,arm=False)['data']
    adcData2 = fpga.snapshots['adc_in_snp_cal2_ss'].read(timeout=5,arm=False)['data']
    adcData3 = fpga.snapshots['adc_in_snp_cal3_ss'].read(timeout=5,arm=False)['data']
    bus0 = np.array([adcData0['data_i0'],adcData0['data_i1'],adcData1['data_i2'],adcData1['data_i3']]).flatten('F')
    bus1 = np.array([adcData2['data_i4'],adcData2['data_i5'],adcData3['data_i6'],adcData3['data_i7']]).flatten('F')
    bus2 = np.array([adcData0['data_q0'],adcData0['data_q1'],adcData1['data_q2'],adcData1['data_q3']]).flatten('F')
    bus3 = np.array([adcData2['data_q4'],adcData2['data_q5'],adcData3['data_q6'],adcData3['data_q7']]).flatten('F')

    adcData = dict()
    adcData.update(adcData0)
    adcData.update(adcData1)
    adcData.update(adcData2)
    adcData.update(adcData3)
    iDataKeys = ['data_i0','data_i1','data_i2','data_i3','data_i4','data_i5','data_i6','data_i7']
    iDataKeys = np.roll(iDataKeys,nRolls)
    #collate
    iValList = np.array([adcData[key] for key in iDataKeys])
    iVals = iValList.flatten('F')
    qDataKeys = ['data_q0','data_q1','data_q2','data_q3','data_q4','data_q5','data_q6','data_q7']
    qDataKeys = np.roll(qDataKeys,nRolls)
    #collate
    qValList = np.array([adcData[key] for key in qDataKeys])
    qVals = qValList.flatten('F')

    return {'bus0':bus0,'bus1':bus1,'bus2':bus2,'bus3':bus3,'adcData':adcData,'iVals':iVals,'qVals':qVals}


def loadDelayCal(fpga,delayLut):
    nLoadDlyRegBits = 6
    notLoadVal = int('1'*nLoadDlyRegBits,2) #when load_dly is this val, no bit delays are loaded
    fpga.write_int('adc_in_load_dly',notLoadVal)
    for iRow,(bit,delay) in enumerate(delayLut):
        fpga.write_int('adc_in_dly_val',delay)
        fpga.write_int('adc_in_load_dly',bit)
        time.sleep(.01)
        fpga.write_int('adc_in_load_dly',notLoadVal)
        
def streamSpectrum(iVals,qVals):
    sampleRate = 2.e9 # 2GHz
    MHz = 1.e6
    adcFullScale = 2.**11

    signal = iVals+1.j*qVals
    signal = signal / adcFullScale

    nSamples = len(signal)
    spectrum = np.fft.fft(signal)
    spectrum = 1.*spectrum / nSamples

    freqsMHz = np.fft.fftfreq(nSamples)*sampleRate/MHz

    freqsMHz = np.fft.fftshift(freqsMHz)
    spectrum = np.fft.fftshift(spectrum)

    spectrumDb = 20*np.log10(np.abs(spectrum))

    peakFreq = freqsMHz[np.argmax(spectrumDb)]
    peakFreqPower = spectrumDb[np.argmax(spectrumDb)]
    times = np.arange(nSamples)/sampleRate * MHz
    #print 'peak at',peakFreq,'MHz',peakFreqPower,'dB'
    return {'spectrumDb':spectrumDb,'freqsMHz':freqsMHz,'spectrum':spectrum,'peakFreq':peakFreq,'times':times,'signal':signal,'nSamples':nSamples}

def checkErrorsAndSetAtten(roach, startAtten=40, iqBalRange=[0.7, 1.3], rmsRange=[0.2,0.3], verbose=False):
    adcFullScale = 2.**11
    curAtten=startAtten
    rmsTarget = np.mean(rmsRange)

    roach.loadFullDelayCal()
    
    while True:
        atten3 = np.floor(curAtten*2)/4.
        atten4 = np.ceil(curAtten*2)/4.

        if verbose:
            print 'atten3', atten3
            print 'atten4', atten4
            
        roach.changeAtten(3, atten3)
        roach.changeAtten(4, atten4)
        snapDict = roach.snapZdok(nRolls=0)
        
        iVals = snapDict['iVals']/adcFullScale
        qVals = snapDict['qVals']/adcFullScale
        iRms = np.sqrt(np.mean(iVals**2))
        qRms = np.sqrt(np.mean(qVals**2))
        
        if verbose:
            print 'iRms', iRms
            print 'qRms', qRms
        iqRatio = iRms/qRms

        if iqRatio<iqBalRange[0] or iqRatio>iqBalRange[1]:
            raise Exception('IQ balance out of range!')

        if rmsRange[0]<iRms<rmsRange[1] and rmsRange[0]<qRms<rmsRange[1]:
            break

        else:
            iDBOffs = 20*np.log10(rmsTarget/iRms)
            qDBOffs = 20*np.log10(rmsTarget/qRms)
            dbOffs = (iDBOffs + qDBOffs)/2
            curAtten -= dbOffs
            curAtten = np.round(4*curAtten)/4.

    return curAtten 

def checkSpectrumForSpikes(specDict):
    sortedSpectrum=np.sort(specDict['spectrumDb'])
    spectrumFlag=0
    #checks if there are spikes above the forest. If there are less than 5 tones at least 10dB above the forest are cosidered spikes
    for i in range(-5,-1):
        if (sortedSpectrum[-1]-sortedSpectrum[i])>10:
            spectrumFlag=1
            break
    return spectrumFlag
    

if __name__=='__main__':
    roachList = []
    specDictList = []
    plotSnaps = True
    startAtten = 40

    for arg in sys.argv[1:]:
        ip = '10.0.0.'+arg
        roach = Roach2Controls(ip, 'DarknessFpga_V2.param', True)
        roach.connect()
        roach.initializeV7UART()
        roachList.append(roach)
    
    for roach in roachList:
        atten = checkErrorsAndSetAtten(roach, startAtten)
        print 'Roach', roach.ip[-3:], 'atten =', atten
        
    print 'Checking for spikes in ADC Spectrum...'
    if plotSnaps:
        specFigList = []
        specAxList = []
    for roach in roachList:
        snapDict = roach.snapZdok()
        specDict = streamSpectrum(snapDict['iVals'], snapDict['qVals'])
        specDictList.append(specDict)
        flag = checkSpectrumForSpikes(specDict)
        if flag!=0:
            print 'Spikes in spectrum for Roach', roach.ip
            if plotSnaps:
                fig,ax = plt.subplots(1, 1)
                ax.plot(specDict['freqsMHz'], specDict['spectrumDb'])
                ax.set_xlabel('Frequency (MHz)')
                ax.set_title('Spectrum for Roach ' + roach.ip[-3:])
                specFigList.append(fig)
                specAxList.append(ax)

    print 'Done!'
        

    if plotSnaps:
        figList = []
        axList = []
        for i,specDict in enumerate(specDictList):
            fig,ax = plt.subplots(1, 1)
            ax.plot(specDict['times'], specDict['signal'].real, color='b', label='I')
            ax.plot(specDict['times'], specDict['signal'].imag, color='g', label='Q')
            ax.set_title('Roach ' + roachList[i].ip[-3:] + ' Timestream')
            ax.set_xlabel('Time (us)')
            ax.set_xlim([0,0.5])
            ax.legend()

        plt.show()

     
    
    
    
            

