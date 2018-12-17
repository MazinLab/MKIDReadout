import numpy as np
import argparse
import glob
from findLOsAndMakeFreqLists import findLOs, modifyTemplarConfigFile
from sweepdata import *

parser = argparse.ArgumentParser(description='Find LOs and convert metadata files to templar lists')
parser.add_argument('metadata_a', help='First metadata file')
parser.add_argument('metadata_b', help='Second metadata file')
parser.add_argument('templarConfig', help='High Templar config file to modify')
args = parser.parse_args()

#fileList = glob.glob(os.path.join(args.metadataDir, '*metadata*txt')

metadataA = SweepMetadata(file=args.metadata_a)
metadataB = SweepMetadata(file=args.metadata_b)

freqSplitIndA = np.where(np.isnan(metadataA.mlfreq))[0][0]
freqSplitIndB = np.where(np.isnan(metadataB.mlfreq))[0][-1] + 1

freqs = metadataA.mlfreq[:freqSplitIndA][metadataA.flag[:freqSplitIndA]==ISGOOD]
freqs = np.append(freqs, metadataB.mlfreq[freqSplitIndB:][metadataB.flag[freqSplitIndB:]==ISGOOD])

loA, loB = findLOs(freqs/1.e9, loRange=0.1)
print 'lo1', loA
print 'lo2', loB

aFile = metadataA.save_templar_freqfile(loA*1.e9)
bFile = metadataB.save_templar_freqfile(loB*1.e9)

modifyTemplarConfigFile(args.templarConfig, [metadataA.feedline, metadataB.feedline], [metadataA.roachnum, metadataB.roachnum], [aFile, bFile], [loA, loB], ['a', 'b'])
