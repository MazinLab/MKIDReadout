import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, sys, struct, time
from mkidcore.corelog import getLogger, INFO, DEBUG
import mkidreadout.config

from mkidreadout.channelizer.Roach2Controls import Roach2Controls
from mkidreadout.configuration.sweepdata import SweepMetadata
from mkidreadout.channelizer.binTools import reinterpretBin
from mkidreadout.channelizer.adcTools import streamSpectrum

# TODO: merge DAC LUT loading with Roach2Controls
def loadDACLUT(roach, metadata, lofreq):
    resIDs, freqs, attens, phases, iqRatios = metadata.templar_data(lofreq)
    roach.generateResonatorChannels(freqs)
    roach.setAttenList(attens)
    roach.setLOFreq(lofreq)
    roach.resIDs = resIDs
    roach.iqPhaseOffsList = phases
    roach.iqRatioList = iqRatios

    roach.generateFftChanSelection()
    roach.loadLOFreq()
    newDacAtten = roach.generateDacComb()['dacAtten']
    getLogger(__name__).info("Setting DAC atten to " + str(newDacAtten))
    roach.changeAtten(1,np.floor(newDacAtten*2)/4.)
    roach.changeAtten(2,np.ceil(newDacAtten*2)/4.)
    roach.loadDacLUT()
    newADCAtten = roach.getOptimalADCAtten(15)
    getLogger(__name__).info("Auto Setting ADC Atten to " + str(newADCAtten))

    
def takeQdrSnap(fpga):
    assert 'trig_qdr' in fpga.listdev(), 'Must load qdrloop firmware!'
    fpga.write_int('trig_qdr',0)#release trigger
    time.sleep(.1)  # wait for other trigger conditions to be met, and fill buffers
    fpga.write_int('trig_qdr',1)#trigger snapshots
    time.sleep(.1)  # wait for other trigger conditions to be met, and fill buffers
    fpga.write_int('trig_qdr',0)#release trigger

    nQdrSamples = 2**20
    nBytesPerQdrSample = 8
    nBitsPerQdrSample = nBytesPerQdrSample*8
    qdr0Vals = np.array(readMemory(fpga,'qdr0_memory',nQdrSamples,nBytesPerQdrSample,bQdrFlip=True),dtype=np.uint64)
    qdr1Vals = np.array(readMemory(fpga,'qdr1_memory',nQdrSamples,nBytesPerQdrSample,bQdrFlip=True),dtype=np.uint64)
    qdr2Vals = np.array(readMemory(fpga,'qdr2_memory',nQdrSamples,nBytesPerQdrSample,bQdrFlip=True),dtype=np.uint64)
    #qdr0 has the most significant 64 bits, followed by qdr1, and last qdr2
    #smush them all toghether into 192 bit words
    wholeQdrWords = (qdr0Vals<<(nBitsPerQdrSample*2))+(qdr1Vals<<(nBitsPerQdrSample))+qdr2Vals
    print 'qdrdtype', wholeQdrWords.dtype

    nBitsPerSample = 12 #I or Q sample
    nSamplesPerWholeWord = 16 # 8 IQ pairs per clock cycle
    bitmask = int('1'*nBitsPerSample,2)
    # bitshifts = nBitsPerSample*np.arange(nSamplesPerWholeWord)[::-1]
    # bitshifts = np.array(bitshifts, dtype=np.uint64)
    qdr0Bitshifts = np.arange(nBitsPerQdrSample%nBitsPerSample, nBitsPerQdrSample, nBitsPerSample)[::-1]
    qdr1Bitshifts = qdr0Bitshifts + nBitsPerQdrSample%nBitsPerSample
    qdr2Bitshifts = qdr1Bitshifts + nBitsPerQdrSample%nBitsPerSample
    qdr0Bitshifts = np.append(qdr0Bitshifts, 0)
    qdr1Bitshifts = np.append(qdr1Bitshifts, 0)
    qdr2Bitshifts = np.append(qdr2Bitshifts, 0)
    
    qdr0Bitshifts = np.array(qdr0Bitshifts, dtype=np.uint64)
    qdr1Bitshifts = np.array(qdr1Bitshifts, dtype=np.uint64)
    qdr2Bitshifts = np.array(qdr2Bitshifts, dtype=np.uint64)
    
    # print 'qdr0Bitshifts', qdr0Bitshifts
    # print 'qdr1Bitshifts', qdr1Bitshifts
    # print 'qdr2Bitshifts', qdr2Bitshifts

    qdr0Samples = ((qdr0Vals[:,np.newaxis]) >> qdr0Bitshifts)&bitmask
    qdr1Samples = ((qdr1Vals[:,np.newaxis]) >> qdr1Bitshifts)&bitmask
    qdr2Samples = ((qdr2Vals[:,np.newaxis]) >> qdr2Bitshifts)&bitmask
    
    qdr0Samples[:,-1] = (((qdr0Samples[:,-1]&(2**(nBitsPerQdrSample%nBitsPerSample)-1)))<<8) + (qdr1Samples[:,0]&(2**(nBitsPerSample-nBitsPerQdrSample%nBitsPerSample)-1))
    qdr1Samples = np.delete(qdr1Samples, 0, axis=1)
    qdr1Samples[:,-1] = (((qdr1Samples[:,-1]&(2**(2*nBitsPerQdrSample%nBitsPerSample)-1)))<<4) + (qdr2Samples[:,0]&(2**(nBitsPerSample-2*nBitsPerQdrSample%nBitsPerSample)-1))
    qdr2Samples = np.delete(qdr2Samples, 0, axis=1)
    samples = np.concatenate((qdr0Samples, qdr1Samples, qdr2Samples), axis=1)
    samples = samples.flatten(order='C')
    samples = reinterpretBin(samples,nBits=nBitsPerSample,binaryPoint=0)
    iVals = samples[::2]
    qVals = samples[1::2]

    # samples = (wholeQdrWords[:,np.newaxis]) >> bitshifts
    # samples = samples & bitmask
    # samples = reinterpretBin(samples,nBits=nBitsPerSample,binaryPoint=0)
    # samples = samples.flatten(order='C')
    # iVals = samples[::2]
    # qVals = samples[1::2]


    return {'iVals':iVals,'qVals':qVals}

def readMemory(fpga,memName,nSamples,nBytesPerSample=4,bQdrFlip=False):
    """read a byte string from a bram or qdr, and parse it into an array"""
    if nBytesPerSample == 4:
        formatChar = 'L'
    elif nBytesPerSample == 8:
        formatChar = 'Q'
    else:
        raise TypeError('nBytesPerSample must be 4 or 8')

    memStr = fpga.read(memName,nSamples*nBytesPerSample)
    memValues = np.array(list(struct.unpack('>{}{}'.format(nSamples,formatChar),memStr)),dtype=np.uint64)
    if bQdrFlip:
        memValues = np.right_shift(memValues,32)+np.left_shift(np.bitwise_and(memValues, int('1'*32,2)),32)
        #Unfortunately, with the current qdr calibration, the addresses in katcp and firmware are shifted (rolled) relative to each other
        #so to compensate we roll the values to write here
        #this will work if you are reading the same length vector that you wrote (and rolled) in katcp
        memValues = np.roll(memValues,1)
    return list(memValues)

def measureSidebands(ifFreqList, fftFreqs, spectrumDB, dacSampleRate=2.e9, nDacSamples=262144):
    freqQuantization = dacSampleRate/nDacSamples
    ifFreqList = np.round(ifFreqList/freqQuantization)*freqQuantization
    getLogger(__name__).info('DAC freq quantization {}'.format(freqQuantization))
    print 'DAC freq quantization {}'.format(freqQuantization)
    assert np.all(np.abs(ifFreqList) < 1.e9)
    freqLocs = np.zeros(len(ifFreqList), dtype=np.int)
    sbLocs = np.zeros(len(ifFreqList), dtype=np.int)
    for i, freq in enumerate(ifFreqList):
        freqLocs[i] = np.argmin(np.abs(freq - fftFreqs))
        sbLocs[i] = np.argmin(np.abs(freq + fftFreqs))

    return spectrumDB[freqLocs] - spectrumDB[sbLocs], freqLocs, sbLocs

def plotSBSuppression(fftFreqs, spectrumDB, freqLocs, sbLocs):
    fig = plt.figure(figsize=[18,10])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(fftFreqs[freqLocs], spectrumDB[freqLocs], '.', label='tones')
    ax1.plot(fftFreqs[freqLocs], spectrumDB[sbLocs], '.', label='sidebands')
    ax1.vlines(fftFreqs[freqLocs], spectrumDB[sbLocs], spectrumDB[freqLocs] , alpha=0.3, color='#d62728')
    ax1.legend(prop={'size':15})
    ax1.set_ylabel('dBFS', size=15)
    ax1.set_ylim((-115, -30))
    ax1.set_title('Sideband Suppression', size=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    ax2.plot(fftFreqs[freqLocs], spectrumDB[freqLocs] - spectrumDB[sbLocs], label='total suppression')
    ax2.set_ylabel('dBFS', size=15)
    ax2.set_xlabel('IF Band Frequency', size=15)
    ax2.set_ylim((-10,60))
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.legend(prop={'size':15})

    plt.show()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for taking QDR longsnaps')
    parser.add_argument('roach', type=int, help='Roach number')
    parser.add_argument('-c', '--config', default=None, dest='config', type=str, 
                    help='The config file - will setup using list, lo, etc from this file.')
    parser.add_argument('-o', '--output', default=None, help='output npz file')
    parser.add_argument('-l', '--load-lut', action='store_true', 
                    help='loads a new DAC LUT (from config file) before taking snapshot')
    parser.add_argument('-b', '--sidebands', action='store_true',
                    help='measures, plots, and saves tone sideband power from provided freq list')
    args = parser.parse_args()

    getLogger(__name__, setup=True)
    getLogger('mkidreadout').setLevel(DEBUG)
    getLogger('casperfpga').setLevel(INFO)

    if args.load_lut and args.config is None:
        raise Exception('Must specify config file to load frequency list')
    if args.sidebands and args.config is None:
        raise Exception('Must specify config file to measure sidebands')

    if args.config is not None:
        config = mkidreadout.config.load(args.config)
        roach = Roach2Controls(config.roaches.get('r{}.ip'.format(args.roach)), 
                    feedline=config.roaches.get('r{}.feedline'.format(args.roach)),
                    range=config.roaches.get('r{}.range'.format(args.roach)))
    else:
        roach = Roach2Controls('10.0.0.' + str(args.roach))

    roach.connect()
    assert 'trig_qdr' in roach.fpga.listdev(), 'QDR firmware not loaded!'

    if args.load_lut:
        getLogger(__name__).info("Loading new DAC LUT from config file")
        fn = str(config.roaches.get('r{}.freqfileroot'.format(args.roach)))
        fn = os.path.join(config.paths.data, fn.format(roach=args.roach, 
                    feedline=config.roaches.get('r{}.feedline'.format(args.roach)), 
                    range=config.roaches.get('r{}.range'.format(args.roach))))
        metadata = SweepMetadata(file=fn)
        loadDACLUT(roach, metadata, config.roaches.get('r{}.lo_freq'.format(args.roach)))


    roach.loadFullDelayCal()
    snapDict = takeQdrSnap(roach.fpga)
    nBins = 2**21 #out of 2**23
    specDict = streamSpectrum(iVals=snapDict['iVals'],qVals=snapDict['qVals'],nBins=nBins)
    snapDict.update(specDict)

    if args.sidebands:
        fn = str(config.roaches.get('r{}.freqfileroot'.format(args.roach)))
        fn = os.path.join(config.paths.data, fn.format(roach=args.roach, 
                    feedline=config.roaches.get('r{}.feedline'.format(args.roach)), 
                    range=config.roaches.get('r{}.range'.format(args.roach))))
        metadata = SweepMetadata(file=fn)
        loFreq = config.roaches.get('r{}.lo_freq'.format(args.roach))
        _, freqs, _, _, _ = metadata.templar_data(loFreq)        
        ifFreqs = freqs - loFreq
        nSamples = roach.params['nDacSamplesPerCycle'] * roach.params['nLutRowsToUse']
        sbPowers, freqLocs, sbLocs = measureSidebands(ifFreqs, 1.e6*snapDict['freqsMHz'], 
                          snapDict['spectrumDb'], roach.params['dacSampleRate'], nSamples)
        snapDict.update({'sidebandPowers':sbPowers, 'freqLocs':freqLocs, 'sbLocs':sbLocs})
        plt.plot(ifFreqs, sbPowers)
        plt.show()

    timeLabel = time.strftime("%Y%m%d-%H%M%S",time.localtime())
    if args.output is None:
        if args.config is not None:
            args.output = os.path.join(config.paths.data, 'adcDataLong_{}.npz'.format(timeLabel))
            if not os.path.exists(config.paths.data):
                os.mkdir(config.paths.data)
        else:
            args.output = 'adcDataLong_{}.npz'.format(timeLabel)
    else:
        args.output = args.output.split('.')[0] + '_{}.npz'.format(timeLabel)

    getLogger(__name__).info('Saving data in {}'.format(args.output))
    np.savez(args.output, **snapDict) 


    

