"""
Use to refind resonator locations after some global shift, like
warmup to room temp. Powers are assumed to be correct. 

INPUTS:
    - FreqSweep npz file containing hightemplar sweeps of all resonators at 
      their correct powers. TODO: decide whether to make this its own class
    - Metadata file to modify

Algorithm:
    1. For each resonator in MD file, find the IQV peak within some window
    2. Fit (f_iqvpeak - f_mdfile) vs f_mdfile to polynomial
    3. Use polynomial fit to correct frequencies in MD file
"""
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import mkidreadout.configuration.sweepdata as sd

def findIQVPeaks(freqs, sweep, winSize=None):
    #loSpan = sweep.freqs[0, -1] - sweep.freqs[0, 0]
    if winSize is None:
        winSize = sweep.freqs[0, 0] - sweep.freqs[0, -1]
    if sweep.natten > 1:
        raise Exception('Multiple attens not supported')

    newFreqs = []
    iqVels = np.sqrt(np.diff(sweep.i, axis=2)**2 + np.diff(sweep.q, axis=2)**2)
    iqVels = np.squeeze(iqVels) #now has shape (nTone, nLOStep)
    for f in freqs:
        toneInd = np.argmin(np.abs(f - sweep.freqs[:, sweep.nlostep/2]))
        if not sweep.freqs[toneInd, 0] <= f <= sweep.freqs[toneInd, -1]:
            raise Exception('No Sweep found for f = {}'.format(f))
        newFreqInd = np.argmax(iqVels[toneInd, :])
        newFreqs.append(sweep.freqs[toneInd, newFreqInd])

    return np.array(newFreqs)


def fitDeltaF(oldFreqs, iqvPeaks, order=2):
    deltaF = iqvPeaks - oldFreqs
    fitParams = np.polyfit(oldFreqs, deltaF)
    fittedFreqs = np.zeros(len(oldFreqs))
    for i, p in enumerate(range(order+1)[::-1]):
        fittedFreqs += fitParams[i]*oldFreqs**p

    return fittedFreqs, fitParams

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep', help='Sweep npz file. Should be high templar version w/ only one atten')
    parser.add_argument('metadata', help='Metadata file to modify. Won\'t be overwritten')
    parser.add_argument('-o', '--metadata-out', help='Output metadata file', default=None)
    args = parser.parse_args()

    sweep = sd.FreqSweep(args.sweep)
    metadata = sd.SweepMetadata(file=args.metadata)

    if args.metadata_out is None:
        outFile = args.metadata.split('.')[0] + '_corrected.txt'
    else:
        outFile = args.metadata_out

    goodMask = metadata.flag & sd.ISGOOD
    freqsToFix = metadata.freq[goodMask]

    iqvPeaks = findIQVPeaks(freqsToFix, sweep)
    fittedFreqs, fitParams = fitDeltaF(freqsToFix, iqvPeaks)

    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    ax0.plot(freqsToFix, iqvPeaks, '.')
    ax0.plot(freqsToFix, fittedFreqs, '-')
    ax1.plot(freqsToFix, iqvPeaks - fittedFreqs, '.')
    plt.show()
    
