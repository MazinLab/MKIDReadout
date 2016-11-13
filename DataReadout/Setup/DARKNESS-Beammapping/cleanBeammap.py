"""
cleanBeammap.py

Takes rough beammap with overlapping pixels and pixels out of bounds
Reconciles doubles by placing them in nearest empty pix
Places o.o.b pixels in random position.
Outputs final cleaned beammap with ID, flag, x, y; ready to go into dashboard
"""

import numpy as np
import random
import os
from readDict import readDict

# Open config file
configData = readDict()
configData.read_from_file(self.configFileName)

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
numCols = int(self.configData['numCols'])
outputDirectory = str(configData['outputDirectory'])
outputFilename = str(configData['outputFilename'])
doubleFilename = str(configData['doubleFilename'])
loadDirectory = str(configData['loadDirectory'])
loadDataFilename = str(configData['loadDataFilename'])
loadDoublesFilename = str(configData['loadDoublesFilename'])

nRows = 100
nCols = 80

roughBMFile='beammap_rough.txt'
finalBMFile='beammap_final.txt'

roughPath = os.path.join(outputDirectory,roughBMFile)
finalPath = os.path.join(outputDirectory,finalBMFile)

placeUnbeammappedPixels = 0

roughBM = np.loadtxt(roughPath)
ids = roughBM[:,0]
flags = roughBM[:,1]
roughX = roughBM[:,2]
roughY = roughBM[:,3]

noloc = []
onlyX = []
onlyY = []


    grid = np.zeros((nRows,nCols))
    overlaps = []
    gridDicts = []
    locationData = []
    
    
    
    
    
