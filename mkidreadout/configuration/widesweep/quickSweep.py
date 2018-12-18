import numpy as np
import matplotlib.pyplot as plt

from mkidreadout.channelizer.RoachStateMachine import RoachStateMachine
from mkidreadout.configuration.widesweep.digitalWS import DigitalWideSweep
import time, sys



def quickSweep(defaultValues,roachNums, filePrefix, debug):

    startFreqs=np.asarray([3.5E9]*len(roachNums))
    startFreqs[1::2]+=2.E9
    stopFreqs=np.asarray([5.5E9]*len(roachNums))
    stopFreqs[1::2]+=2.E9

    offset = np.arange(0,len(startFreqs))*1.e6
    #startFreqs+=offset
    #stopFreqs+=offset


    #startFreqs = [5.5E9]
    #stopFreqs=[7.5E9]
    digWS = DigitalWideSweep(roachNums, defaultValues,filePrefix,debug=False)

    #initilize roaches to be in sweep state (don't redefine LUTs)
    for r in digWS.roaches:
        r.addCommands(RoachStateMachine.LOADFREQ)
        digWS.execQApp+=2   # finished is emmitted in autoddssync and when done all commands
        r.finished.connect(digWS.quitQApp)                                       # When all the commands are finished stop the QApplication
    for t in digWS.roachThreads:
        t.start()
    digWS.app.exec_()
    for i, r in enumerate(digWS.roaches):
        r.finished.disconnect(digWS.quitQApp)    
        r.state[:4]=[RoachStateMachine.COMPLETED]*4
        loFreq = int(r.config.getfloat('Roach '+str(r.num),'lo_freq'))+offset[i]
        r.roachController.setLOFreq(loFreq)
        r.roachController.generateFftChanSelection()
        
        dacFreqList = r.roachController.freqList-loFreq
        dacFreqList[np.where(dacFreqList<0.)] += r.roachController.params['dacSampleRate']
        dacFreqResolution = r.roachController.params['dacSampleRate']/(r.roachController.params['nDacSamplesPerCycle']*r.roachController.params['nLutRowsToUse'])
        dacQuantizedFreqList = np.round(dacFreqList/dacFreqResolution)*dacFreqResolution
        r.roachController.dacQuantizedFreqList=dacQuantizedFreqList

    

    widesweepFN = digWS.outPath+'/'+'raw_'+digWS.outPrefix+'digWS_r'
    roachNums = np.copy(digWS.roachNums)

    digWS.startWS(roachNums=None, startFreqs=startFreqs, endFreqs=stopFreqs,DACatten=None, ADCatten=None,resAtten=65,makeNewFreqs=False)

    del digWS.app   #important in order to use matplotlib later!
    del digWS

    return widesweepFN, roachNums


def plotWS(widesweepFN, roachNums):
    
    colors=['blue', 'darkblue', 'red', 'darkred', 'lime', 'green', 'violet', 'purple', 'cyan', 'teal', 'yellow', 'olive','silver', 'dimgrey']
    
    plt.figure()
    for i, rNum in enumerate(roachNums):
        fn= widesweepFN+str(rNum)+'.txt'
        
        data=np.loadtxt(fn)
        s21 = np.log10(data[:,1]**2. + data[:,2]**2.)
        freqs = data[:,0]

        plt.plot(freqs, s21, ls='-',c=colors[i], label=str(rNum))

    plt.xlabel('Freq [GHz]')
    plt.ylabel('Power')
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    
    #plt.figure()
    #plt.plot([0],[1])
    #print plt.get_current_fig_manager()
    #plt.close()

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

    widesweepFN, roachNums=quickSweep(defaultValues,roachNums, filePrefix, debug)


    plotWS(widesweepFN, roachNums)


    








