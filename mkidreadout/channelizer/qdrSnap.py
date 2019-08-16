import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, sys
from mkidcore.corelog import getLogger, INFO, DEBUG
import mkidreadout.config

from mkidreadout.channelizer.Roach2Controls import Roach2Controls
from mkidreadout.configuration.sweepdata import SweepMetadata

def loadDACLUT(roach, metadata, lofreq):
    resIDs, freqs, attens, phases, iqRatios = metadata.templar_data(lofreq)
    roach.generateResonatorChannels(freqs)
    roach.setAttenList(attens)
    roach.setLOFreq(lofreq)
    roach.resIDs = resIDs
    roach.phaseOffsList = phases
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

def measureSidebands(ifFreqList, fftFreqs, spectrumDB):
    assert np.all(np.abs(ifFreqList) < 1.e9)
    freqLocs = np.zeros(len(ifFreqList), dtype=np.int)
    sbLocs = np.zeros(len(ifFreqList), dtype=np.int)
    for i, freq in enumerate(ifFreqList):
        freqLocs[i] = np.argmin(np.abs(freq - fftFreqs))
        sbLocs[i] = np.argmin(np.abs(freq + fftFreqs))

    return spectrumDB[freqLocs] - spectrumDB[sbLocs]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for taking QDR longsnaps')
    parser.add_argument('roach', type=int, help='Roach number')
    parser.add_argument('-c', '--config', default=None, dest='config', type=str, 
                    help='The config file - will setup using list, lo, etc from this file.')
    parser.add_argument('-o', '--output', default=None)
    args = parser.parse_args()

    getLogger(__name__, setup=True)
    getLogger(__name__).setLevel(INFO)

    if args.config is not None:
        getLogger(__name__).info("Loading new DAC LUT from config file")
        config = mkidreadout.config.load(args.config)
        fn = str(config.roaches.get('r{}.freqfileroot'.format(args.roach)))
        fn = os.path.join(config.paths.data, fn.format(roach=args.roach, 
                    feedline=config.roaches.get('r{}.feedline'.format(args.roach)), 
                    range=config.roaches.get('r{}.range'.format(args.roach))))
        metadata = SweepMetadata(file=fn)
        roach = Roach2Controls(config.roaches.get('r{}.ip'.format(args.roach)), 
                    feedline=config.roaches.get('r{}.feedline'.format(args.roach)),
                    range=config.roaches.get('r{}.range'.format(args.roach)))
        roach.connect()

        loadDACLUT(roach, metadata, config.roaches.get('r{}.lo_freq'.format(args.roach)))

