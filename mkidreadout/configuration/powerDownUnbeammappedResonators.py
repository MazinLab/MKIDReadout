import numpy as np
import os, sys

#todo make part of a frequency comb object

def powerDownBadResonators(resIDList, attenList, badResIDList):
    badInds = []
    for badResID in badResIDList:
        if(np.any(badResID==resIDList)):
            badInds.append(np.where(resIDList==badResID)[0][0])

    badInds = np.array(badInds)
    attenList[badInds] = 99
    return attenList


if __name__=='__main__':
    # usage: python powerDownUnbeammappedResonators.py <freqfile_in_$MKID_DATA_DIR>
    mdd = os.environ['MKID_DATA_DIR']
    beammapFile = '/mnt/data0/Darkness/20180522/Beammap/finalMap_20180524.txt'

    freqFile = os.path.join(mdd,sys.argv[1])
    resIDs, freqs, attens = np.loadtxt(freqFile, unpack=True)
    bmResIDs, bmFlag, x, y = np.loadtxt(beammapFile, unpack=True)

    badResIDs = bmResIDs[np.where(bmFlag!=0)[0]]

    attens = powerDownBadResonators(resIDs, attens, badResIDs)

    np.savetxt(freqFile.split('.')[0] + '_pdubr.txt', np.transpose([resIDs, freqs, attens]), fmt='%4i %10.9e %4i')



    
