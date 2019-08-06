import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from mkidreadout.configuration.sweepdata import SweepMetadata, ISGOOD
from wpsnn import COLLISION_FREQ_RANGE
from checkWPSPerformance import matchResonators, matchAttens

def retrieveManResList(metadata):
    goodMask = ~np.isnan(metadata.atten)
    collMask = np.append(np.diff(metadata.freq) < COLLISION_FREQ_RANGE, True)
    goodMask = goodMask & (~collMask) #humans are unreliable!
    goodMask = goodMask & (metadata.atten != np.nanmax(metadata.atten))
    goodMask = goodMask & (metadata.atten != -1)

    resIDs = metadata.resIDs
    freqs = metadata.freq
    attens = metadata.atten

    return resIDs, freqs, attens, goodMask

def retrieveMLResList(metadata, threshold=0):
    goodMask = ((metadata.flag & ISGOOD) == ISGOOD)
    goodMask = goodMask & (metadata.ml_isgood_score >= args.threshold)

    resIDs = metadata.resIDs
    freqs = metadata.mlfreq
    attens = metadata.mlatten

    return resIDs, freqs, attens, goodMask

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compare resonator classification between multiple methods')
    parser.add_argument('manMDFiles', nargs='+', help='manual md files')
    parser.add_argument('-ml', '--ml-metadata', default=None, help='ML metadata file')
    parser.add_argument('-df', '--max-df', type=float, default=250.e3)
    parser.add_argument('-t', '--threshold', type=float, default=0)
    parser.add_argument('-c', '--cut', type=float, default=1.0)
    parser.add_argument('-l', '--lower', default=2.5, type=float)
    parser.add_argument('-u', '--upper', default=2.5, type=float)
    parser.add_argument('-p', '--plotConfusion', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-m', '--match-res', action='store_true', help='If true, resIDs are assumed to not correspond')
    args = parser.parse_args()

    if int(args.ml_metadata is not None) + len(args.manMDFiles) > 2:
        raise Exception('Can only have 2 files total (combined man and ml)')

    aFileName = args.manMDFiles[0]
    mdA = SweepMetadata(file=aFileName)
    resIDA, freqA, attenA, goodMaskA = retrieveManResList(mdA) #first file, assumed to be manual

    if len(args.manMDFiles) > 1:
        bFileName = args.manMDFiles[1]
        mdB = SweepMetadata(file=bFileName) 
        resIDB, freqB, attenB, goodMaskB = retrieveManResList(mdB) #compare two manual files
    elif args.ml_metadata is not None:
        bFileName = args.manMDFiles[1]
        mdB = SweepMetadata(file=bFileName) 
        resIDB, freqB, attenB, goodMaskB = retrieveMLResList(mdB, args.threshold) #use ML inference from provided ML file
    else: 
        aFileName = 'manual'
        bFileName = 'ml'
        resIDB = mdA.resIDs[goodMaskA]
        freqB = mdA.mlfreq[goodMaskA]
        attenB = mdA.mlatten[goodMaskA]

    if args.match_res:
        atob = matchResonators(resIDA, resIDB, freqA, freqB, args.max_df)
        bNotInA = np.empty((0, 2))

        for i, resID in enumerate(mlResIDs):
            if not np.any(i == manToML[:, 0]):
                bNotInA = np.vstack((bNotInA, np.array([resID, mlScores[i]])))

        print np.sum(~np.isnan(atob[:,0])), 'resonators matched between files'
        print np.sum(np.isnan(atob[:,0])), 'resonators in', aFileName, 'not found in bFileName'
        print len(bNotInA), 'resonators in', aFileName, 'not found in bFileName'

        if args.verbose:
            print 'A not in B ResIDs', resIDA[np.isnan(atob[:, 0])]
            print 'B not in A ResIDs', bNotInA

        attenAMatched, attenBMatched = matchAttens(attenA, attenB, atob)

    else:
        matchedMask = goodMaskA & goodMaskB
        attenAMatched = attenA
        attenBMatched = attenB

    attenDiff = attenBMatched - attenAMatched

    plt.hist(attenDiff, bins=10, range=(-5,5))
    plt.show()




 
