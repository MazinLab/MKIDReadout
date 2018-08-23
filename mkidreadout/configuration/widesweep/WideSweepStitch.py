import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import signal
import Peaks
#from interval import interval, inf, imath
import math
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse


'''
Takes two widesweep-good.txt files that were generated from 
splitting up a widesweep for multiple people to click through
and stitches them back together.

Inputs:
Filename 1 (1st half of -good peaks)
Filename 2 (2nd half)
index = index in widesweep where the files were split.
        This is necessary for autofit to know the correct data
        points in the widesweep file that correspond to each
        peak to fit the full loops.
        
Output:
Stitched together file (user provides file name)
'''

parser = argparse.ArgumentParser(description='Stitch WS Clickthroughs')
parser.add_argument('inputFiles', nargs='+', type=str, help='Input files')
parser.add_argument('-o', '--output', nargs=1, default=['wsStitch.txt'], help='Output file')
args = parser.parse_args()

data1 = np.loadtxt(args.inputFiles[0])
ids1 = data1[:,0]
indices1 = data1[:,1]
freqs1 = data1[:,2]
 
idOffset = (ids1[-1] + 1)%10000

data2 = np.loadtxt(args.inputFiles[1])
ids2 = data2[:,0]
indices2 = data2[:,1]
freqs2 = data2[:,2]

#remove overlapping frequencies
overlaps = np.where(freqs2<=freqs1[-1])[0]
ids2 = np.delete(ids2, overlaps)
indices2 = np.delete(indices2, overlaps)
freqs2 = np.delete(freqs2, overlaps)
ids2 -= ids2[0]%10000

ids2+=idOffset

idsAll = np.append(ids1, ids2)
indicesAll = np.append(indices1, indices2)
freqsAll = np.append(freqs1, freqs2)

np.savetxt(args.output[0], np.transpose([idsAll, indicesAll, freqsAll]), fmt='%8d %12d %16.7f')

