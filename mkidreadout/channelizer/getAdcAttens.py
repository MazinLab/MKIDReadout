import sys

import matplotlib.pyplot as plt

from mkidreadout.channelizer.Roach2Controls import Roach2Controls
from mkidreadout.channelizer.adcTools import *


#TODO: error check threshold on max to 3.5 RMS at a low RMS value

def checkErrorsAndSetAtten(roach, startAtten=40, iqBalRange=[0.7, 1.3], rmsRange=[0.15,0.19], verbose=False):
    #TODO Merge with roach2controls
    adcFullScale = 2.**11
    curAtten=startAtten
    rmsTarget = np.mean(rmsRange)

    roach.loadFullDelayCal()
    
    while True:
        atten3 = np.floor(curAtten*2)/4.
        atten4 = np.ceil(curAtten*2)/4.

        if verbose or curAtten<0 or curAtten>31.75*2.:
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

    

if __name__=='__main__':
    # TODO Merge with roach2controls main
    roachList = []
    specDictList = []
    plotSnaps = True
    startAtten = 40

    for arg in sys.argv[1:]:
        ip = '10.0.0.'+arg
        roach = Roach2Controls(ip, 'darknessfpga.param', True)
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
        if flag==True:
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

     
    
    
    
            

