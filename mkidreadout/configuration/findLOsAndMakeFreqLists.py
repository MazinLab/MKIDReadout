'''
Author: Neelay Fruitwala

Script for finding optimal LOs from frequency lists. Assumes wide/power sweep using
digital readout; ensures LO is in narrow range around LO used for sweep.

'''
import os
import sys
import argparse
import re

import numpy as np

import mkidcore.config
import mkidreadout.configuration.sweepdata as sd


def findLOs(freqsA, sweepLOA, freqsB=None, sweepLOB=None, loRange=10.e6, nIters=10000, colParamWeight=1, resBW=200.e3, ifHole=3.e6):
    '''
    Finds the optimal LO frequencies for a feedline, given a list of resonator frequencies.
    Does Monte Carlo optimization to minimize the number of out of band tones and sideband 
    collisions.
    
    Parameters
    ----------
        freqsA - list of LF board resonator frequencies, in Hz
        freqsB - list of HF board resonator frequencies, in Hz
        sweepLOA - WPS LO center for LF board
        sweepLOB - WPS LO center for HF board
        loRange - size of LO search band, in Hz; power output should be approx uniform across this 
            band so solution from digital WPS is still valid
        nIters - number of optimization interations
        colParamWeight - relative weighting between number of collisions and number of omitted
            tones in cost function. 1 usually gives good performance, set to 0 if you don't want
            to optimize for sideband collisions. 
        resBW - bandwidth of resonator channels. Tones are considered collisions if their difference
            is less than this value.
        ifHole - tones within this distance from LO are not counted
    Returns
    -------
        lo1, lo2 - low and high frequency LOs (in Hz)
    '''
    if sweepLOB is None:
        sweepLOB = sweepLOA + 2.e9 + 2*loRange
    if freqsB is None:
        freqsB = np.array([sweepLOB - 50.e6, sweepLOB + 60.e6])

    lfRange = np.array([sweepLOA - loRange/2., sweepLOA + loRange/2.])
    hfRange = np.array([sweepLOB - loRange/2., sweepLOB + loRange/2.])
    
    nCollisionsOpt = len(freqsA) + len(freqsB) #number of sideband collisions
    nFreqsOmittedOpt = len(freqsA) + len(freqsB) #number of frequencies outside LO band
    costOpt = nCollisionsOpt + colParamWeight*nCollisionsOpt
    for i in range(nIters):
        loA = np.random.rand(1)[0]*loRange + lfRange[0]
        hflolb = max(hfRange[0], loA + 2.e9) #lower bound of hf sampling range; want LOs to be 1 GHz apart
        loB = np.random.rand(1)[0]*(hfRange[1]-hflolb) + hflolb

        #find nFreqsOmitted
        freqsIFA = freqsA - loA
        freqsIFB = freqsB - loB
        isInBandA = (np.abs(freqsIFA) < 1.e9) & (np.abs(freqsIFA) > ifHole)
        isInBandB = (np.abs(freqsIFB) < 1.e9) & (np.abs(freqsIFB) > ifHole)
        nFreqsOmitted = np.sum(~isInBandA) + np.sum(~isInBandB)

        #find nCollisions
        freqsIFA = freqsIFA[isInBandA]
        freqsIFB = freqsIFB[isInBandB]
        freqsIFASB = np.sort(np.abs(freqsIFA))
        freqsIFBSB = np.sort(np.abs(freqsIFB))
        nLFColl = np.sum(np.diff(freqsIFASB)<resBW)
        nHFColl = np.sum(np.diff(freqsIFBSB)<resBW)
        nCollisions = nLFColl + nHFColl

        #pdb.set_trace()

        cost = nFreqsOmitted + colParamWeight*nCollisions
        if cost<costOpt:
            costOpt = cost
            nCollisionsOpt = nCollisions
            nFreqsOmittedOpt = nFreqsOmitted
            loAOpt = loA
            loBOpt = loB
            print 'nCollOpt', nCollisionsOpt
            print 'nFreqsOmittedOpt', nFreqsOmittedOpt
            print 'los', loA, loB

    print 'Optimal nCollisions', nCollisionsOpt
    print 'Optimal nFreqsOmitted', nFreqsOmittedOpt
    print 'LOA', loAOpt
    print 'LOB', loBOpt

    return loAOpt, loBOpt

def getRoachNumberFromFile(fn):
    return int(re.findall('\d{3}', fn)[0])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_data', nargs='+', help='Sweep npz files for a feedline (LF then HF)')
    parser.add_argument('-m', '--metadata', nargs='+', help='Sweep metadata files')
    parser.add_argument('-c', '--config', help='roach.yml file to modify. WILL BE OVERWRITTEN.')
    args = parser.parse_args()

    if len(args.sweep_data) != len(args.metadata):
        raise Exception('Must specify metadata/sweep file for every board')

    if len(args.sweep_data) > 2:
        raise Exception('Sorry, only one FL at a time for now :(')

    sweepA = sd.FreqSweep(args.sweep_data[0])
    mdA = sd.SweepMetadata(file=args.metadata[0])
    freqsA = mdA.freq[(mdA.flag & sd.ISGOOD) == sd.ISGOOD]

    if len(args.sweep_data) > 1:
        sweepB = sd.FreqSweep(args.sweep_data[1])
        mdB = sd.SweepMetadata(file=args.metadata[1])
        freqsB = mdB.freq[(mdB.flag & sd.ISGOOD) == sd.ISGOOD]
        loA, loB = findLOs(freqsA, sweepA.lo, freqsB, sweepB.lo)
        useB = True

    else:
        loA, _ = findLOs(freqsA, sweepA.lo)
        useB = False


    if args.config is not None:
        cfg = mkidcore.config.load(args.config)
        rnumA = getRoachNumberFromFile(os.path.basename(args.sweep_data[0]))
        cfg.register('r{}.lo_freq'.format(rnumA), float(loA), update=True)
        if useB:
            rnumB = getRoachNumberFromFile(os.path.basename(args.sweep_data[1]))
            cfg.register('r{}.lo_freq'.format(rnumB), float(loB), update=True)

        cfg.save(args.config)
        
