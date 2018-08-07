import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import signal
import Peaks
#from interval import interval, inf, imath
import math
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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

path = os.environ['MKID_DATA_DIR']

file1 = "Varuna_FL3_1-good.txt"
file2 = "Varuna_FL3_2-good.txt"
outfile = "Varuna_FL3-good.txt"
index = 36481 #line number in full ws of freq that is first line in 2nd half, minus 3


fullPath1 = os.path.join(path,file1)
fullPath2 = os.path.join(path,file2)
outPath = os.path.join(path,outfile)

data1 = np.loadtxt(fullPath1)
ids1 = data1[:,0]
indices1 = data1[:,1]
freqs1 = data1[:,2]
 
idOffset = len(ids1)

data2 = np.loadtxt(fullPath2)
ids2 = data2[:,0]
indices2 = data2[:,1]
freqs2 = data2[:,2]

ids2+=idOffset
indices2+=index

gf = open(outPath,'wb')
for i in range(len(ids1)):
    line = "%8d %12d %16.7f\n"%(ids1[i],indices1[i],freqs1[i])
    gf.write(line)
for j in range(len(ids2)):
    line = "%8d %12d %16.7f\n"%(ids2[j],indices2[j],freqs2[j])
    gf.write(line)

gf.close()


