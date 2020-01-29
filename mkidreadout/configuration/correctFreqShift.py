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
import warnings
import mkidreadout.configuration.sweepdata as sd

def findIQVPeaks(freqs, sweep, winSize=None, scaleWin=False, convWin=11):
    #loSpan = sweep.freqs[0, -1] - sweep.freqs[0, 0]
    if winSize is None:
        winSize = sweep.freqs[0, -1] - sweep.freqs[0, 0]
    if sweep.natten > 1:
        raise Exception('Multiple attens not supported')

    newFreqs = []
    iqVels = np.sqrt(np.diff(sweep.i, axis=2)**2 + np.diff(sweep.q, axis=2)**2)
    iqVels = np.squeeze(iqVels) #now has shape (nTone, nLOStep)
    for f in freqs:
        toneInd = np.argmin(np.abs(f - sweep.freqs[:, sweep.nlostep/2]))
        if not sweep.freqs[toneInd, 0] <= f <= sweep.freqs[toneInd, -1]:
            warnings.warn('No Sweep found for f = {}'.format(f))
            newFreqs.append(f)
            continue

        if scaleWin:
            curWinSize = winSize*f/3.5e9 #provided winsize is LF bound 
        else:
            curWinSize = winSize

        freqInd = np.argmin(np.abs(f - sweep.freqs[toneInd, :]))
        nWinPoints = int(curWinSize/sweep.freqStep)
        startInd = max(freqInd - nWinPoints/2, 0)
        endInd = min(freqInd + nWinPoints/2, sweep.nlostep-1)
        windowedIQV = iqVels[toneInd, startInd:endInd]

        if convWin > 0:
            newFreqInd = np.argmax(np.convolve(windowedIQV, np.ones(convWin), mode='same')) + startInd
        else:
            newFreqInd = np.argmax(windowedIQV) + startInd
        newFreqs.append(sweep.freqs[toneInd, newFreqInd])

    return np.array(newFreqs)


def fitDeltaF(oldFreqs, iqvPeaks, order=2):
    deltaF = iqvPeaks - oldFreqs
    fitParams = np.polyfit(oldFreqs, deltaF, 2)
    fittedFreqs = np.zeros(len(oldFreqs))
    for i, p in enumerate(range(order+1)[::-1]):
        fittedFreqs += fitParams[i]*oldFreqs**p
    fittedFreqs += oldFreqs

    return fittedFreqs, fitParams

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep', help='Sweep npz file. Should be high templar version w/ only one atten')
    parser.add_argument('metadata', help='Metadata file to modify. Won\'t be overwritten')
    parser.add_argument('-o', '--metadata-out', help='Output metadata file', default=None)
    parser.add_argument('--snap', action='store_true', help='Final snap in small: something like (50*3.5e9/f0) kHz')
    args = parser.parse_args()

    sweep = sd.FreqSweep(args.sweep)
    metadata = sd.SweepMetadata(file=args.metadata)

    if args.metadata_out is None:
        outFile = args.metadata.split('.')[0] + '_corrected.txt'
        if args.snap:
            outFile = outFile.split('.')[0] + '_snap.txt'
    else:
        outFile = args.metadata_out

    goodMask = (metadata.flag & sd.ISGOOD).astype(bool)
    freqsToFix = metadata.freq[goodMask]

    iqvPeaks = findIQVPeaks(freqsToFix, sweep)
    medCorFreqs = freqsToFix + np.median(iqvPeaks - freqsToFix) #shift by median deviation
    iqvPeaks = findIQVPeaks(medCorFreqs, sweep, winSize=1.e6) #find IQV peaks around median-centered data to remove bias
    fittedFreqs, fitParams = fitDeltaF(freqsToFix, iqvPeaks) 

    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    ax0.plot(freqsToFix, iqvPeaks - freqsToFix, '.')
    ax0.plot(freqsToFix, fittedFreqs - freqsToFix, '-')
    ax1.plot(freqsToFix, iqvPeaks - fittedFreqs, '.')
    ax0.set_title('Initial (noisy) fit')
    plt.show()

    refinedIQVPeaks = findIQVPeaks(fittedFreqs, sweep, winSize=600.e3)
    refinedFreqs, fitParams = fitDeltaF(freqsToFix, refinedIQVPeaks)

    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    ax0.plot(freqsToFix, refinedIQVPeaks - freqsToFix, '.')
    ax0.plot(freqsToFix, refinedFreqs - freqsToFix, '-')
    ax1.plot(freqsToFix, refinedIQVPeaks - refinedFreqs, '.')
    ax0.set_title('Fit after 1 Refinements')
    plt.show()

    #refinedIQVPeaks = findIQVPeaks(refinedFreqs, sweep, winSize=250.e3, convWin=0)
    #refinedFreqs, fitParams = fitDeltaF(freqsToFix, refinedIQVPeaks)

    #fig = plt.figure()
    #ax0 = fig.add_subplot(211)
    #ax1 = fig.add_subplot(212)

    #ax0.plot(freqsToFix, refinedIQVPeaks - freqsToFix, '.')
    #ax0.plot(freqsToFix, refinedFreqs - freqsToFix, '-')
    #ax1.plot(freqsToFix, refinedIQVPeaks - refinedFreqs, '.')
    #ax0.set_title('Fit after 2 Refinements')
    #plt.show()

    if args.snap:
        finalSnapFreqs = findIQVPeaks(refinedFreqs, sweep, winSize=50.e3, scaleWin=True, convWin=3)
        metadata.freq[goodMask] = finalSnapFreqs
        plt.plot(freqsToFix, finalSnapFreqs - refinedFreqs)
        plt.show()
    else:
        metadata.freq[goodMask] = refinedFreqs

    metadata.save(outFile)
    
