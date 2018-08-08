"""
clean.py

Takes rough beammap with overlapping pixels and pixels out of bounds
Reconciles doubles by placing them in nearest empty pix
Places o.o.b pixels in random position, changes flag to 1
Places failed pixels randomly.
Outputs final cleaned beammap with ID, flag, x, y; ready to go into dashboard

TODO:
-Fix bug that is leaving resonator 9999 at x,y = 999,999
"""

from __future__ import print_function
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from mkidreadout.utils.arrayPopup import plotArray
from mkidreadout.utils.readDict import readDict
from mkidreadout.configuration.beammap.flags import beamMapFlags

logging.basicConfig()
log = logging.getLogger('beammap.clean')

def isInCorrectFL_mec(i, x):
    #given a resonator ID (i) and x location
    #returns True if  that y-loc is in the correct FL for that resonator ID
    return int(x/14) == (int(i/10000) - 1)

def isInCorrectFL(i, y):
    #given a resonator ID (i) and y location
    #returns True if  that y-loc is in the correct FL for that resonator ID
    if 0<=i and i<=1999: #FL1
        if (y>=0) and (y<=24):
            return True
        else:
            return False
    elif 2000<=i and i<=3999: #FL2
        if (y>=25) and (y<=49):
            return True
        else:
            return False
    elif 4000<=i and i<=5999: #FL3
        if (y>=50) and (y<=74):
            return True
        else:
            return False
    elif 6000<=i and i<=7999: #FL4
        if (y>=75) and (y<=99):
            return True
        else:
            return False
    elif 8000<=i and i<=9999: #FL5
        if (y>=100) and (y<=124):
            return True
        else:
            return False

#Should get these from config file

#update these filenames manually for now
roughBMFile='RawMap.txt'
finalBMFile='finalMap.txt'

#default config file location
configFileName = '/mnt/data0/MKIDReadoout/configuration/clean.txt'

# Open config file
configData = readDict()
configData.read_from_file(configFileName)

# Extract parameters from config file
beammapFormatFile = str(configData['beammapFormatFile'])
imgFileDirectory = str(configData['imgFileDirectory'])
xSweepStartingTimes = np.array(configData['xSweepStartingTimes'], dtype=int)
ySweepStartingTimes = np.array(configData['ySweepStartingTimes'], dtype=int)
xSweepLength = int(configData['xSweepLength'])
ySweepLength = int(configData['ySweepLength'])
pixelStartIndex = int(configData['pixelStartIndex'])
pixelStopIndex = int(configData['pixelStopIndex'])
nRows = int(configData['numRows'])
nCols = int(configData['numCols'])
flipX = bool(configData['flipX'])
outputDirectory = str(configData['outputDirectory'])
outputFilename = str(configData['outputFilename'])
doubleFilename = str(configData['doubleFilename'])
loadDirectory = str(configData['loadDirectory'])
loadDataFilename = str(configData['loadDataFilename'])
loadDoublesFilename = str(configData['loadDoublesFilename'])

instrument = str(configData['instrument'])

#put together full input/output BM paths
roughPath = os.path.join(outputDirectory,roughBMFile)
finalPath = os.path.join(outputDirectory,finalBMFile)

#load location data from rough BM file
roughBM = np.loadtxt(roughPath)
ids = np.array(roughBM[:,0],dtype=np.uint32)
flags = np.array(roughBM[:,1],dtype=np.uint16)
preciseXs = roughBM[:,2]
preciseYs = roughBM[:,3]

#setup arrays for BM cleaning
placeUnbeammappedPixels = 0
noloc = []
onlyX = []
onlyY = []

grid = np.zeros((nRows,nCols))
overlaps = []
gridDicts = []

#define fails as boolean array based on flags. 0 is "good" 1 or more is "fail"
fails = np.array(flags,dtype=np.bool)

#first find out-of-bounds "good" pixels, and reflag as bad
oobMask = ((~fails) & (preciseXs < 0)) | ((~fails) & (preciseXs > nCols)) | ((~fails) & (preciseYs < 0)) | ((~fails) & (preciseYs > nRows))
#print oobMask
for (i,flag,x,y) in zip(ids[oobMask],flags[oobMask],preciseXs[oobMask],preciseYs[oobMask]):
    print("Pixel ", i, " is o.o.b. with x = ", x, " and y = ", y)
flags[oobMask]=1
print("Reflagging: ", flags[oobMask])

#redefine fails now that oob pixels have been properly flagged
fails = np.array(np.logical_and(flags!=beamMapFlags['good'], flags!=beamMapFlags['double']),dtype=np.bool)
#define mask of good pixels
goodMask = (~fails)

#check for solitary "good" pixels that are not in their FL
#if it's more than a one pix away, reflag as a fail "4"
#otherwise push it to inside edge of nearest pixel on its correct FL
#if this causes an overlap, let the overlap code handle it


def improve_feedlines_mec(ids, flags, goodMask, preciseXs,preciseYs):

    global flipx
    flines = np.floor(ids / 10000.0) - 1
    flpos = 9 - flines if flipx else flines.copy()
    #handle X
    wayoutX = mask = (preciseXs < flpos * 14 - 1) | (preciseXs > (flpos + 1) * 14 + 1)
    mask &= goodMask
    log.info('Marking IDs {} (FL {}) at {} as bad(4)', ids[mask], flines[mask], preciseXs[mask])
    flags[mask] = beamMapFlags['wrongFeedline']
    preciseXs[mask], preciseYs[mask] = 999.0, 999.0

    mask = (preciseXs <= flpos * 14) & ~wayoutX
    mask &= goodMask
    newX = flpos * 14 + .01
    log.info('Moving IDs {} (FL {}) from {} to {}', ids[mask], flines[mask], preciseXs[mask], newX[mask])
    preciseXs[mask] = newX[mask]

    mask = (preciseXs >= (flpos + 1) * 14) & ~wayoutX
    mask &= goodMask
    newX = (flpos+1)*14-.01
    log.info('Moving IDs {} (FL {}) from {} to {}', ids[mask], flines[mask], preciseXs[mask], newX[mask])
    preciseXs[mask] = newX[mask]

    #Handle Y
    wayoutY = mask = (preciseYs<-1) | (preciseYs>146)
    mask &= goodMask
    log.info('Marking IDs {} (FL {}) as out of y-range', ids[mask], flines[mask])
    flags[mask] = beamMapFlags['failed']

    mask = ~wayoutY & (preciseYs<0)
    mask &= goodMask
    log.info('IDs {} (FL {}) slightly OOB, moving to 0', ids[mask], flines[mask])
    preciseYs[mask]=0

    mask = (preciseYs>145) & ~wayoutY
    mask &= goodMask
    log.info('IDs {} (FL {}) slightly OOB, moving to 144.9', ids[mask], flines[mask])
    preciseYs[mask]=144.9


def improve_feedlines(ids, flags, goodMask, preciseXs, preciseYs):
    for (i, flag, y) in zip(ids[goodMask], flags[goodMask], preciseYs[goodMask]):
        if 0 <= i and i <= 1999:  # FL1
            if y >= 26:
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Flagged as bad (4)")
                flags[np.where(ids == i)[0]] = 4
                preciseXs[np.where(ids == i)[0]] = 999.0
                preciseYs[np.where(ids == i)[0]] = 999.0
            elif y >= 25:
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Moved to y = 24.99")
                preciseYs[np.where(ids == i)[0]] = 24.99

        elif 2000 <= i and i <= 3999:  # FL2
            if (y >= 51) or (y < 24):
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Flagged as bad (4)")
                flags[np.where(ids == i)[0]] = 4
                preciseXs[np.where(ids == i)[0]] = 999.0
                preciseYs[np.where(ids == i)[0]] = 999.0
            elif y >= 50:
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Moved to y = 49.99")
                preciseYs[np.where(ids == i)[0]] = 49.99
            elif y < 25:
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Moved to y = 25.01")
                preciseYs[np.where(ids == i)[0]] = 25.01

        elif 4000 <= i and i <= 5999:  # FL3
            if (y >= 76) or (y < 49):
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Flagged as bad (4)")
                flags[np.where(ids == i)[0]] = 4
                preciseXs[np.where(ids == i)[0]] = 999.0
                preciseYs[np.where(ids == i)[0]] = 999.0
            elif y >= 75:
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Moved to y = 74.99")
                preciseYs[np.where(ids == i)[0]] = 74.99
            elif y < 50:
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Moved to y = 50.01")
                preciseYs[np.where(ids == i)[0]] = 50.01

        elif 6000 <= i and i <= 7999:  # FL4
            if (y >= 101) or (y < 74):
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Flagged as bad (4)")
                flags[np.where(ids == i)[0]] = 4
                preciseXs[np.where(ids == i)[0]] = 999.0
                preciseYs[np.where(ids == i)[0]] = 999.0
            elif y >= 100:
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Moved to y = 99.99")
                preciseYs[np.where(ids == i)[0]] = 99.99
            elif y < 75:
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Moved to y = 75.01")
                preciseYs[np.where(ids == i)[0]] = 75.01

        elif 8000 <= i and i <= 9999:  # FL5
            if (y < 99):
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Flagged as bad (4)")
                flags[np.where(ids == i)[0]] = 4
                preciseXs[np.where(ids == i)[0]] = 999.0
                preciseYs[np.where(ids == i)[0]] = 999.0
            elif y < 100:
                print("Pixel ", i, " is out of its FL with y = ", y)
                print("Moved to y = 100.01")
                preciseYs[np.where(ids == i)[0]] = 100.01


if instrument == 'mec':
    improve_feedlines_mec(ids, flags, goodMask, preciseXs,preciseYs)
else:
    improve_feedlines(ids, flags, goodMask, preciseXs, preciseYs)


#Define x and y integer position in grid as floor of precise x and y coordinates
xs = np.floor(preciseXs).astype(np.int)
ys = np.floor(preciseYs).astype(np.int)

#redefine fails now that out-of-FL pixels have been re-flagged
fails = (flags!=beamMapFlags['good']) & (flags!=beamMapFlags['double'])
#define mask of good pixels
goodMask = (~fails)

#look through goodMask positions and add each pixel to their corresponding location in the grid
#if a grid position already has 
for (x,y) in zip(xs[goodMask],ys[goodMask]):
    grid[y,x] += 1

#plot initial "good" grid with number of resonators assigned to each location
fig,ax = plt.subplots(1,1)
ax.legend(loc='best')
ylims = ax.get_ylim()
ax.set_ylim(ylims[1],ylims[0])
plotArray(grid,title='Original "good" flagged pixel locations',origin='upper')

#setup dictionary for storing new output data
for i in range(len(ids)):
    resId = ids[i]
    distVector = (preciseXs[i]-(xs[i]+.5),preciseYs[i]-(ys[i]+.5))
    distMag = np.sqrt(distVector[0]**2 + distVector[1]**2)
    gridDicts.append({'x':xs[i],'y':ys[i],'preciseX':preciseXs[i],'preciseY':preciseYs[i],'distMag':distMag,'resId':resId,'fail':flags[i]})

gridDicts = np.array(gridDicts)

#try to clean up overlaps
print('good', np.sum(goodMask))
print('res', np.min(ids[goodMask]), np.max(ids[goodMask]))

for entry in gridDicts:
    x = entry['x']
    y = entry['y']
    if entry['fail']==beamMapFlags['good'] or entry['fail']==beamMapFlags['double']:
        overlapItems = [(item,k) for (k,item) in enumerate(gridDicts) if (item['x'] == entry['x'] and item['y'] == entry['y'])]
        overlapItems,gridIndices = zip(*overlapItems) #unzip
        nPixelsAssigned = len(overlapItems)
        if nPixelsAssigned > 1:
            closestPixelIndex = np.argmin([item['distMag'] for item in overlapItems])
            print(nPixelsAssigned, 'pixels assigned to', (x, y))
            for j in range(len(overlapItems)):
                if j != closestPixelIndex:
                    #print move all the assigned pixels but the closest elsewhere
                    #find the closest open pixel, preferably in 
                    #the direction the distVector is pointing
                    unassignedCoords = np.where(grid==0)
                    unassignedCoords = zip(unassignedCoords[1],unassignedCoords[0]) #x,y
                    unassignedNeighbors = [coord for coord in unassignedCoords if (np.abs(coord[0]-x)<=1 and np.abs(coord[1]-y)<=1 and isInCorrectFL(gridDicts[gridIndices[j]]['resId'],coord[1]))]

                    try:
                        bestUnassignedNeighbor = np.argmin([np.sqrt((coord[0]+.5-overlapItems[j]['preciseX'])**2 + (coord[1]+.5-overlapItems[j]['preciseY'])**2)])
                        newX = unassignedNeighbors[bestUnassignedNeighbor][0]
                        newY = unassignedNeighbors[bestUnassignedNeighbor][1]
                        print('best unassigned neighbor', bestUnassignedNeighbor, newX, newY)
                        gridDicts[gridIndices[j]]['x'] = newX
                        gridDicts[gridIndices[j]]['y'] = newY
                        grid[newY,newX] = 7
                    except IndexError:
                        print('no neighbor could be found')
                        gridDicts[gridIndices[j]]['fail'] = beamMapFlags['duplicatePixel']
                    except:
                        print('error in best-neighbor reassignment')

plotArray(grid,title='Relocated overlaps (moved pixels given 7 value)', origin='upper')

#randomly place failed pixels into empty locations
for entry in gridDicts:
    x = entry['x']
    y = entry['y']
    i = entry['resId']
    if entry['fail']!=beamMapFlags['good'] and entry['fail']!=beamMapFlags['double']:
        unassignedCoords = np.where(grid==0)
        unassignedCoords = zip(unassignedCoords[1],unassignedCoords[0]) #x,y
        unassignedNeighbors = [coord for coord in unassignedCoords if isInCorrectFL(i,coord[1])]
        try:
            newX = unassignedNeighbors[0][0]
            newY = unassignedNeighbors[0][1]
            print('placing failed pixel ', i, newX, newY)
            entry['x'] = newX
            entry['y'] = newY
            grid[newY,newX] = 10
        except IndexError:
            print('no neighbor could be found')
        except:
            print('error in failed-BM pixel reassignment')

plotArray(grid,title='Final BM, 7=overlap fix, 10=random drop', origin='upper')

newLocationData = [[entry['resId'],entry['fail'],entry['x'],entry['y']] for entry in gridDicts]
newLocationData = np.array(newLocationData)

np.savetxt(finalPath,newLocationData,fmt='%d\t%d\t%d\t%d')
print('wrote clean beammap to ', finalPath)
