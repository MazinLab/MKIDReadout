import numpy as np
import matplotlib.pyplot as plt
import argparse
from mkidreadout.configuration.sweepdata import SweepMetadata

parser = argparse.ArgumentParser(description='Check score triage performance')
parser.add_argument('mlFiles', nargs='+')
parser.add_argument('-m', '--manualFiles', nargs='+')
parser.add_argument('-c', '--cut', type=float)
args = parser.parse_args()

if args.manualFiles is None:
    args.manualFiles = args.mlFiles

assert len(args.mlFiles)==len(args.manualFiles), 'Must specify clickthrough file for every inference file!'
nFiles = len(args.mlFiles)

mlMetadataList = []
manualMetadataList = []
attenDiff = np.array([])
goodScore = np.array([])
badScore = np.array([])
resID = np.array([])

badAttenDiff = np.array([])
badGoodScore = np.array([])

doubleThresh = 500.e3
doubleMask = np.array([], dtype=bool)

for i in range(nFiles):
    mlMetadataList.append(SweepMetadata(file=args.mlFiles[i]))
    manualMetadataList.append(SweepMetadata(file=args.manualFiles[i]))
    goodMask = ~np.isnan(mlMetadataList[i].atten)
    attenDiff = np.append(attenDiff, mlMetadataList[i].mlatten[goodMask] - manualMetadataList[i].atten[goodMask])
    goodScore = np.append(goodScore, mlMetadataList[i].ml_isgood_score[goodMask])
    badScore = np.append(badScore, mlMetadataList[i].ml_isbad_score[goodMask])
    resID = np.append(resID, mlMetadataList[i].resIDs[goodMask])

    badResMask = manualMetadataList[i].atten == np.nanmax(manualMetadataList[i].atten) #bad res were set to max atten
    badAttenDiff = np.append(badAttenDiff, mlMetadataList[i].mlatten[badResMask] - manualMetadataList[i].atten[badResMask])
    badGoodScore = np.append(badGoodScore, mlMetadataList[i].ml_isgood_score[badResMask])

    dbMask = np.diff(manualMetadataList[i].wsfreq)<doubleThresh
    dbMask = np.append(dbMask, False)
    dbMask = dbMask | np.roll(dbMask, 1)
    doubleMask = np.append(doubleMask, dbMask[goodMask])

dbThresh = 2.5
cutMask = goodScore >= args.cut
stdCut = np.std(attenDiff[cutMask])
clickStd = np.std(attenDiff[~cutMask])
nBadClassCut = np.sum(np.abs(attenDiff[cutMask])>=dbThresh)

worstResIDs = resID[np.argsort(np.abs(attenDiff))[::-1]]
attenDiffSorted = attenDiff[np.argsort(np.abs(attenDiff))[::-1]]
attenDiffCut = attenDiff[cutMask]
attenDiffCutSorted = attenDiffCut[np.argsort(np.abs(attenDiffCut))[::-1]]
worstResIDsCut = (resID[cutMask])[np.argsort(np.abs(attenDiffCut))[::-1]]

stdDoubles = np.std(attenDiff[doubleMask])
stdNotDoubles = np.std(attenDiff[~doubleMask])
nBadNotDoubles = np.sum(np.abs(attenDiff[~doubleMask])>=dbThresh)
nBadDoubles = np.sum(np.abs(attenDiff[doubleMask])>=dbThresh)

badResCutMask = badGoodScore >= args.cut

remainMask = goodScore < args.cut
print np.sum(cutMask), '(', 100.*np.sum(cutMask)/len(goodScore), '%)', 'resonators cut, requiring', np.sum(remainMask), 'for clickthrough.'
print 'cut std:', stdCut
print 'click std:', clickStd
print nBadClassCut, '(', 100.*nBadClassCut/np.sum(cutMask), '%)', 'cut resonators misclassified by', dbThresh, 'dB.'
print len(badAttenDiff), 'bad resonators (set to max in clickthrough).', np.sum(badResCutMask), 'cut from clickthrough'
print ''
print np.sum(doubleMask), ' doubles.'
print 'double std:', stdDoubles
print 'not double std:', stdNotDoubles
print nBadNotDoubles, '(', 100.*nBadNotDoubles/np.sum(~doubleMask), '%)', 'cut singles misclassified by', dbThresh, 'dB.'
print nBadDoubles, '(', 100.*nBadDoubles/np.sum(doubleMask), '%)', 'doubles misclassified by', dbThresh, 'dB.'
print 'Worst offenders:'
for i in range(20):
    print worstResIDs[i], attenDiffSorted[i]
print 'Worst cut offenders:'
for i in range(30):
    print worstResIDsCut[i], attenDiffCutSorted[i]

#fig0 = plt.figure()
#fig1 = plt.figure()

#plt.hist(goodScore[cutMask], alpha=0.5, bins=20)
#plt.hist(goodScore[~cutMask], alpha=0.5, bins=20)
plt.hist(attenDiff[cutMask], bins=10, alpha=0.5, range=(-5,5))
plt.hist(attenDiff[~cutMask], bins=10, alpha=0.5, range=(-5,5))
plt.show()
