import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from mkidreadout.configuration.sweepdata import SweepMetadata, ISGOOD
#from wpsnn import COLLISION_FREQ_RANGE
COLLISION_FREQ_RANGE = 0
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

def retrieveMLResList(metadata, threshold=0, nManRes=None):
    goodMask = ((metadata.flag & ISGOOD) == ISGOOD)
    goodMask = goodMask & (metadata.atten != np.nanmax(metadata.atten))
    if threshold >= 1:
        assert nManRes is not None
        sortedScores = np.sort(metadata.ml_isgood_score[goodMask])[::-1]
        threshold = sortedScores[nManRes-1]
        print 'Using ML score threshold: ', threshold
    goodMask = goodMask & (metadata.ml_isgood_score >= threshold)
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
    #parser.add_argument('-c', '--cut', type=float, default=1.0)
    parser.add_argument('-l', '--lower', default=2.5, type=float)
    parser.add_argument('-u', '--upper', default=2.5, type=float)
    parser.add_argument('-p', '--plotConfusion', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-m', '--match-res', action='store_true', help='If true, resIDs are assumed to not correspond')
    parser.add_argument('-o', '--output-file', default=None, 
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
        saveDir = os.path.dirname(args.manMDFiles[1])
        resIDB, freqB, attenB, goodMaskB = retrieveManResList(mdB) #compare two manual files
        usingML = False
        plotTitle = aFileName + ' vs ' + bFileName
        plotFn = aFileName + '_vs_' + bFileName
    elif args.ml_metadata is not None:
        bFileName = os.path.basename(args.ml_metadata).split('.')[0] + '_ml'
        saveDir = os.path.dirname(args.ml_metadata)
        mdB = SweepMetadata(file=args.ml_metadata)
        resIDB, freqB, attenB, goodMaskB = retrieveMLResList(mdB, args.threshold, np.sum(goodMaskA)) #use ML inference from provided ML file
        usingML = True
        plotTitle = bFileName + ' vs manual'
        plotFn = bFileName
    else: 
        aFileName = 'manual'
        bFileName = 'ml'
        resIDB, freqB, attenB, goodMaskB = retrieveMLResList(mdA, args.threshold, np.sum(goodMaskA)) #use ML inference from provided ML file
        usingML = True
        saveDir = os.path.dirname(args.manMDFiles[0])
        plotTitle = os.path.basename(args.manMDFiles[0]) + ' ml vs manual'
        plotFn = os.path.basename(args.manMDFiles[0])

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
        resIDAMatched, resIDBMatched = matchAttens(resIDA, resIDB, atob)

    else:
        matchedMask = goodMaskA & goodMaskB
        aNotInBMask = goodMaskA & (~matchedMask)
        bNotInAMask = goodMaskB & (~matchedMask)
        attenAMatched = attenA[matchedMask]
        attenBMatched = attenB[matchedMask]
        freqAMatched = freqA[matchedMask]
        freqBMatched = freqB[matchedMask]
        resIDAMatched = resIDA[matchedMask]
        resIDBMatched = resIDB[matchedMask]
        print np.sum(matchedMask), 'resonators matched between files'
        print np.sum(aNotInBMask), 'resonators in', aFileName, 'not found in', bFileName
        print np.sum(bNotInAMask), 'resonators in', bFileName, 'not found in', aFileName

        if args.verbose:
            print 'A not in B ResIDs', resIDA[aNotInBMask]
            print 'B not in A ResIDs', resIDB[bNotInAMask]

    attenDiff = attenBMatched - attenAMatched
    freqDiff = freqBMatched - freqAMatched

    if args.verbose:
        diffAttenMask = np.abs(attenDiff) > 0.5
        resIDPairDiffs = np.vstack((resIDAMatched, resIDBMatched, attenDiff)).T
        print 'ResID discrepancies: A B Diff'
        print np.round(resIDPairDiffs[diffAttenMask]).astype(int)

    print 'mean', np.mean(attenDiff)
    print 'std', np.std(attenDiff)
    print 'fraction within 1 dB', float(np.sum(np.abs(attenDiff) <= 1))/len(attenDiff)
    print 'median', np.median(attenDiff)
    print 'fraction within 1 dB of median', float(np.sum(np.abs(attenDiff - np.median(attenDiff)) <= 1))/len(attenDiff)
        
        

    plt.hist(attenDiff, bins=20, range=(-4.75, 5.25))
    #plt.hist(attenDiff, bins=10, range=(-4.5, 5.5))
    plt.title(plotTitle)
    plt.xlabel('AttenDiff (' + bFileName + ' - ' + aFileName + ')')
    plt.savefig(os.path.join(saveDir, plotFn + '_attenDiff.png'))
    plt.show()

    plt.hist(freqDiff, bins=20, range=(-100.e3, 100.e3))
    plt.xlabel('FreqDiff (Hz; ' + bFileName + ' - ' + aFileName + ')')
    plt.show()

    if args.plotConfusion:
        attenStart = min(np.append(attenAMatched, attenBMatched))
        attenAMatched -= attenStart
        attenBMatched -= attenStart
        attenAMatched = np.round(attenAMatched).astype(int)
        attenBMatched = np.round(attenBMatched).astype(int)
        confImage = np.zeros((max(attenAMatched) + 1, max(attenBMatched) + 1))
        for i in range(len(attenAMatched)):
            confImage[attenAMatched[i], attenBMatched[i]] += 1
    
        plt.imshow(np.transpose(confImage), vmax=55)
        if usingML:
            plt.xlabel('True Atten')
            plt.ylabel('Guess Atten')
        else:
            plt.xlabel(aFileName)
            plt.ylabel(bFileName)

        plt.title(plotTitle)
        plt.colorbar()
        plt.savefig(os.path.join(saveDir, plotFn+'_confusion.png'))
        plt.show()

    if args.output_file:
        mdAvg = SweepMetadata(resid=resIDAMatched, flag=ISGOOD*np.ones(len(resIDAMatched)), mlatten=(attenAMatched + attenBMatched)/2., mlfreq=(freqAMatched + freqBMatched)/2., wsfreq=(freqAMatched + freqBMatched)/2.)
        mdAvg.save(file=args.output_file)




 
