import os, sys, time, struct, traceback
import numpy as np
from tables import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.signal as signal
from scipy import optimize
import scipy.stats as stats
from readDict import readDict
'''
Takes beammap cfg file and gives output beam list with every pixel flagged as failed
'''

if len(sys.argv)<3:
    print "Give <cfg filename> and <FL#> as arguments"
    sys.exit()

try:
    configFileName=str(sys.argv[1])
    configData = readDict()
    configData.read_from_file(configFileName)
except:
    print "Could not find cfg file"
    sys.exit()


# Extract parameters from config file
beammapFormatFile = str(configData['beammapFormatFile'])
imgFileDirectory = str(configData['imgFileDirectory'])
xSweepStartingTimes = np.array(configData['xSweepStartingTimes'], dtype=int)
ySweepStartingTimes = np.array(configData['ySweepStartingTimes'], dtype=int)
xSweepLength = int(configData['xSweepLength'])
ySweepLength = int(configData['ySweepLength'])
pixelStartIndex = int(configData['pixelStartIndex'])
pixelStopIndex = int(configData['pixelStopIndex'])
numRows = int(configData['numRows'])
numCols = int(configData['numCols'])
outputDirectory = str(configData['outputDirectory'])

doubleFilename = str(configData['doubleFilename'])
loadDirectory = str(configData['loadDirectory'])
loadDataFilename = str(configData['loadDataFilename'])
loadDoublesFilename = str(configData['loadDoublesFilename'])

outputFilename = os.path.join(outputDirectory,"Beamlist_FL%s_allBad.txt"%sys.argv[2])

idArray = np.arange(pixelStartIndex,pixelStopIndex)
fArray = np.ones(len(idArray),dtype=int)
pArray = np.zeros(len(idArray),dtype=float)

f=open(outputFilename,'w')
for i in np.arange(len(idArray)):
    f.write('%i\t%i\t%1.1f\t%1.1f\n'%(idArray[i],fArray[i],pArray[i],pArray[i]))
    
f.close()
print outputFilename


