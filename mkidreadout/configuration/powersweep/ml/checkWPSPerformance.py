import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from mkidreadout.configuration.sweepdata import SweepMetadata, ISGOOD
from wpsnn import COLLISION_FREQ_RANGE

def matchResonators(manResIDs, mlResIDs, manFreqs, mlFreqs, maxdf=250.e3):
    manToML = np.zeros((len(manResIDs), 2))
    manToML[:, 0] = np.nan #index of corresponding ML resonator
    manToML[:, 1] = np.inf #df between this and ML resonator

    for i in range(len(manToML)):
        closestMLResInd = np.argmin(np.abs(mlFreqs - manFreqs[i]))
        manToML[i, 0] = closestMLResInd
        manToML[i, 1] = mlFreqs[closestMLResInd] - manFreqs[i]

    duplicateMask = np.diff(manToML[:, 0])==0
    print 'Found', np.sum(duplicateMask), 'duplicates'
    if np.any(duplicateMask):
        duplicateInds = np.where(duplicateMask)[0]
        for ind in duplicateInds:
            toDelete = np.argmax([np.abs(manToML[ind, 1]), np.abs(manToML[ind+1, 1])])
            manToML[ind + toDelete, 0] = np.nan

    foundMLIDs = manToML[~np.isnan(manToML[:, 0]), 0]
    assert np.all(foundMLIDs == np.unique(foundMLIDs))

    tooFarMask = np.abs(manToML[:, 1]) > maxdf
    manToML[tooFarMask, 0] = np.nan

    return manToML

    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Check score triage performance')
    parser.add_argument('mlFile', help='ml metadata file')
    parser.add_argument('manualFile', help='manual metadata file')
    parser.add_argument('-df', '--max-df', type=float, default=250.e3)
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-c', '--cut', type=float)
    parser.add_argument('-l', '--lower', default=2.5, type=float)
    parser.add_argument('-u', '--upper', default=2.5, type=float)
    parser.add_argument('-p', '--plotConfusion', action='store_true')
    args = parser.parse_args()
    
    manualMD = SweepMetadata(file=args.manualFile)
    goodMaskManual = ~np.isnan(manualMD.atten)
    collMaskManual = np.append(np.diff(manualMD.freq) < COLLISION_FREQ_RANGE, True)
    goodMaskManual = goodMaskManual & (~collMaskManual) #humans are unreliable!
    goodMaskManual = goodMaskManual & (manualMD.atten != np.nanmax(manualMD.atten))

    mlMD = SweepMetadata(file=args.mlFile)
    goodMaskML = mlMD.flag == ISGOOD
    goodMaskML = goodMaskML & (mlMD.ml_isgood_score >= args.threshold)

    manResIDs = manualMD.resIDs[goodMaskManual]
    manFreqs = manualMD.freq[goodMaskManual]
    manAttens = manualMD.atten[goodMaskManual]

    sortedInds = np.argsort(manFreqs)
    manResIDs = manResIDs[sortedInds]
    manFreqs = manFreqs[sortedInds]
    manAttens = manAttens[sortedInds]

    mlResIDs = mlMD.resIDs[goodMaskML]
    mlFreqs = mlMD.freq[goodMaskML]
    mlAttens = mlMD.atten[goodMaskML]
    mlScores = mlMD.ml_isgood_score[goodMaskML]

    manToML = matchResonators(manResIDs, mlResIDs, manFreqs, mlFreqs, args.max_df)
    falsePositives = np.empty((0, 2))

    for i, resID in enumerate(mlResIDs):
        if not np.any(i == manToML[:, 0]):
            falsePositives = np.vstack((falsePositives, np.array([resID, mlScores[i]])))

    print 'ML:', np.sum(goodMaskML), 'found resonators above thresh'

    print 'ML found', np.sum(~np.isnan(manToML[:,0])), 'out of', len(manResIDs), \
            'resonators (', 100*float(np.sum(~np.isnan(manToML[:,0])))/len(manResIDs), '%).'
    
    print 'Not found ResIDs: ', manResIDs[np.isnan(manToML[:, 0])]

    print 'False positive ML ResIDs: ', falsePositives
