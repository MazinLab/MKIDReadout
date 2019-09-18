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
    goodMask = goodMask & ((metadata.flag & ISGOOD) == ISGOOD)

    resIDs = metadata.resIDs
    freqs = metadata.freq
    attens = metadata.atten

    return resIDs, freqs, attens, goodMask

def retrieveMLResList(metadata, threshold=0):
    goodMask = ((metadata.flag & ISGOOD) == ISGOOD)
    goodMask = goodMask & (metadata.ml_isgood_score >= args.threshold)
    goodMask = goodMask & (metadata.mlatten != np.nanmax(metadata.atten))

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
    parser.add_argument('-o', '--output-flag', default=None, 
                    help='Outputs the average of the two provided files, named using provided flag')
    args = parser.parse_args()

    if int(args.ml_metadata is not None) + len(args.manMDFiles) > 2:
        raise Exception('Can only have 2 files total (combined man and ml)')

    mdA = SweepMetadata(file=args.manMDFiles[0])
    aFileName = os.path.basename(args.manMDFiles[0]).split('.')[0] + '_manual'
    resIDA, freqA, attenA, goodMaskA = retrieveManResList(mdA) #first file, assumed to be manual

    if len(args.manMDFiles) > 1:
        mdB = SweepMetadata(file=args.manMDFiles[1]) 
        bFileName = os.path.basename(args.manMDFiles[1]).split('.')[0] + '_manual'
        resIDB, freqB, attenB, goodMaskB = retrieveManResList(mdB) #compare two manual files
    elif args.ml_metadata is not None:
        bFileName = os.path.basename(args.ml_metadata).split('.')[0] + '_ml'
        mdB = SweepMetadata(file=args.ml_metadata)
        resIDB, freqB, attenB, goodMaskB = retrieveMLResList(mdB, args.threshold) #use ML inference from provided ML file
    else: 
        aFileName = 'manual'
        bFileName = 'ml'
        resIDB, freqB, attenB, goodMaskB = retrieveMLResList(mdA, args.threshold) #use ML inference from provided ML file

    print np.sum(goodMaskA), 'resonators in', aFileName
    print np.sum(goodMaskB), 'resonators in', bFileName

    if args.match_res:
        resIDA = resIDA[goodMaskA]
        freqA = freqA[goodMaskA]
        attenA = attenA[goodMaskA]
        resIDB = resIDB[goodMaskB]
        freqB = freqB[goodMaskB]
        attenB = attenB[goodMaskB]

        sortedIndA = np.argsort(freqA)
        resIDA = resIDA[sortedIndA]
        freqA = freqA[sortedIndA]
        attenA = attenA[sortedIndA]

        sortedIndB = np.argsort(freqB)
        resIDB = resIDB[sortedIndB]
        freqB = freqB[sortedIndB]
        attenB = attenB[sortedIndB]
    

        atob = matchResonators(resIDA, resIDB, freqA, freqB, args.max_df)
        bNotInA = np.empty((0, 2))

        for i, resID in enumerate(resIDB):
            if not np.any(i == atob[:, 0]):
                bNotInA = np.vstack((bNotInA, np.array([resID, 0])))

        print np.sum(~np.isnan(atob[:,0])), 'resonators matched between files'
        print np.sum(np.isnan(atob[:,0])), 'resonators in', aFileName, 'not found in', bFileName
        print len(bNotInA), 'resonators in', bFileName, 'not found in', aFileName

        if args.verbose:
            print 'A not in B ResIDs', resIDA[np.isnan(atob[:, 0])]
            print 'B not in A ResIDs', bNotInA

        attenAMatched, attenBMatched = matchAttens(attenA, attenB, atob)
        freqAMatched, freqBMatched = matchAttens(freqA, freqB, atob)

    else:
        matchedMask = goodMaskA & goodMaskB
        aNotInBMask = goodMaskA & (~matchedMask)
        bNotInAMask = goodMaskB & (~matchedMask)
        attenAMatched = attenA[matchedMask]
        attenBMatched = attenB[matchedMask]
        freqAMatched = freqA[matchedMask]
        freqBMatched = freqB[matchedMask]
        print np.sum(matchedMask), 'resonators matched between files'
        print np.sum(aNotInBMask), 'resonators in', aFileName, 'not found in', bFileName
        print np.sum(bNotInAMask), 'resonators in', bFileName, 'not found in', aFileName

        if args.verbose:
            print 'A not in B ResIDs', resIDA[aNotInBMask]
            print 'B not in A ResIDs', resIDB[bNotInAMask]

    attenDiff = attenBMatched - attenAMatched
    freqDiff = freqBMatched - freqAMatched

    plt.hist(attenDiff, bins=10, range=(-10,10))
    plt.show()

    plt.hist(freqDiff, bins=20, range=(-100.e3, 100.e3))
    plt.show()




 
