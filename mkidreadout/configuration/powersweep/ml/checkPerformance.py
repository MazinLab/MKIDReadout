import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from mkidreadout.configuration.sweepdata import SweepMetadata

parser = argparse.ArgumentParser(description='Check score triage performance')
parser.add_argument('mlFiles', nargs='+')
parser.add_argument('-m', '--manualFiles', nargs='+')
parser.add_argument('-c', '--cut', type=float)
parser.add_argument('-l', '--lower', default=2.5, type=float)
parser.add_argument('-u', '--upper', default=2.5, type=float)
parser.add_argument('-p', '--plotConfusion', action='store_true')
args = parser.parse_args()

args.lower = np.abs(args.lower)
args.upper = np.abs(args.upper)

if args.manualFiles is None:
    args.manualFiles = args.mlFiles

assert len(args.mlFiles)==len(args.manualFiles), 'Must specify clickthrough file for every inference file!'
nFiles = len(args.mlFiles)

mlMetadataList = []
manualMetadataList = []
attenDiff = np.array([])
mlAttens = np.array([])
manualAttens = np.array([])
goodScore = np.array([])
badScore = np.array([])
resID = np.array([])

badAttenDiff = np.array([])
badGoodScore = np.array([])

doubleThresh = 200.e3
doubleMask = np.array([], dtype=bool)

for i in range(nFiles):
    mlMetadataList.append(SweepMetadata(file=args.mlFiles[i]))
    manualMetadataList.append(SweepMetadata(file=args.manualFiles[i]))
    goodMask = ~np.isnan(mlMetadataList[i].atten)
    attenDiff = np.append(attenDiff, mlMetadataList[i].mlatten[goodMask] - manualMetadataList[i].atten[goodMask])
    mlAttens = np.append(mlAttens, mlMetadataList[i].mlatten[goodMask])
    manualAttens = np.append(manualAttens, manualMetadataList[i].atten[goodMask])
    goodScore = np.append(goodScore, mlMetadataList[i].ml_isgood_score[goodMask])
    badScore = np.append(badScore, mlMetadataList[i].ml_isbad_score[goodMask])
    resID = np.append(resID, mlMetadataList[i].resIDs[goodMask])

    badResMask = manualMetadataList[i].atten == np.nanmax(manualMetadataList[i].atten) #bad res were set to max atten
    badResMask |= manualMetadataList[i].atten == -1
    badAttenDiff = np.append(badAttenDiff, mlMetadataList[i].mlatten[badResMask] - manualMetadataList[i].atten[badResMask])
    badGoodScore = np.append(badGoodScore, mlMetadataList[i].ml_isgood_score[badResMask])

    dbMask = np.diff(manualMetadataList[i].wsfreq)<doubleThresh
    dbMask = np.append(dbMask, False)
    dbMask = dbMask | np.roll(dbMask, 1)
    doubleMask = np.append(doubleMask, dbMask[goodMask])

cutMask = goodScore >= args.cut
stdCut = np.std(attenDiff[cutMask])
clickStd = np.std(attenDiff[~cutMask])
nBadLowerClassCut = np.sum(attenDiff[cutMask]<=-args.lower)
nBadUpperClassCut = np.sum(attenDiff[cutMask]>=args.upper)

worstResIDs = resID[np.argsort(np.abs(attenDiff))[::-1]]
attenDiffSorted = attenDiff[np.argsort(np.abs(attenDiff))[::-1]]
attenDiffCut = attenDiff[cutMask]
attenDiffCutSorted = attenDiffCut[np.argsort(np.abs(attenDiffCut))[::-1]]
worstResIDsCut = (resID[cutMask])[np.argsort(np.abs(attenDiffCut))[::-1]]

stdDoubles = np.std(attenDiff[doubleMask])
stdNotDoubles = np.std(attenDiff[~doubleMask])
nBadLowerNotDoubles = np.sum(attenDiff[~doubleMask]<=-args.lower)
nBadUpperNotDoubles = np.sum(attenDiff[~doubleMask]>=args.upper)
nBadLowerDoubles = np.sum(attenDiff[doubleMask]<=-args.lower)
nBadUpperDoubles = np.sum(attenDiff[doubleMask]>=args.upper)

badResCutMask = badGoodScore >= args.cut

remainMask = goodScore < args.cut
print np.sum(cutMask), '(', 100.*np.sum(cutMask)/len(goodScore), '%)', 'resonators cut, requiring', np.sum(remainMask), 'for clickthrough.'
print 'cut std:', stdCut
print 'click std:', clickStd
print nBadLowerClassCut, '(', 100.*nBadLowerClassCut/np.sum(cutMask), '%)', 'cut resonators overpowered by', -args.lower, 'dB.'
print nBadUpperClassCut, '(', 100.*nBadUpperClassCut/np.sum(cutMask), '%)', 'cut resonators underpowered by', args.upper, 'dB.'
print len(badAttenDiff), 'bad resonators (set to max in clickthrough).', np.sum(badResCutMask), 'cut from clickthrough'
print ''
print np.sum(doubleMask), ' doubles.'
print 'double std:', stdDoubles
print 'not double std:', stdNotDoubles
print nBadLowerNotDoubles, '(', 100.*nBadLowerNotDoubles/np.sum(~doubleMask), '%)', 'cut singles overpowered by', -args.lower, 'dB.'
print nBadUpperNotDoubles, '(', 100.*nBadUpperNotDoubles/np.sum(~doubleMask), '%)', 'cut singles underpowered by', args.upper, 'dB.'
print nBadLowerDoubles, '(', 100.*nBadLowerDoubles/np.sum(~doubleMask), '%)', 'cut doubles overpowered by', -args.lower, 'dB.'
print nBadUpperDoubles, '(', 100.*nBadUpperDoubles/np.sum(~doubleMask), '%)', 'cut doubles underpowered by', args.upper, 'dB.'
#print 'Worst offenders:'
#for i in range(20):
#    print worstResIDs[i], attenDiffSorted[i]
#print 'Worst cut offenders:'
#for i in range(30):
#    print worstResIDsCut[i], attenDiffCutSorted[i]

#fig0 = plt.figure()
#fig1 = plt.figure()

if args.plotConfusion:
    manualAttens[manualAttens==-1] = max(manualAttens)
    mlAttens[mlAttens==-1] = max(mlAttens)
    attenStart = min(manualAttens)
    manualAttens -= attenStart
    mlAttens -= attenStart
    manualAttens = np.round(manualAttens).astype(int)
    mlAttens = np.round(mlAttens).astype(int)
    confImage = np.zeros((max(manualAttens) - min(manualAttens) + 1, max(manualAttens) - min(manualAttens) + 1))
    for i in range(len(manualAttens)):
        confImage[manualAttens[i], mlAttens[i]] += 1

    plt.imshow(np.transpose(confImage), vmax=30)
    plt.xlabel('True Atten')
    plt.ylabel('Guess Atten')
    title = os.path.basename(args.mlFiles[0]).split('.')[0] #args.mlFiles[0].split('_')[-1].split('.')[0]
    plt.title(title)
    plt.colorbar()
    plt.savefig(os.path.join(os.path.dirname(args.mlFiles[0]), title+'.png'))
    plt.show()


#plt.hist(goodScore[cutMask], alpha=0.5, bins=20)
#plt.hist(goodScore[~cutMask], alpha=0.5, bins=20)
plt.hist(attenDiff[cutMask], bins=20, alpha=0.5, range=(-10,10))
plt.hist(attenDiff[~cutMask], bins=10, alpha=0.5, range=(-5,5))
plt.show()
