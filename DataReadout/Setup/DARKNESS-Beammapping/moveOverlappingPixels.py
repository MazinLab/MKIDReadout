import numpy as np
import matplotlib.pyplot as plt
from arrayPopup import plotArray
import os

basePath = '/mnt/data0/Darkness/20161107/beammaps/'
if __name__=='__main__':
    labels = ['fl_all']
    feedlines = [1]

    filenames = ['RawMap.txt']
    outFileName = 'finalMap.txt'
    colors = ['blue']
    yOffsets = [0]

    nRows = 125
    nCols = 80
    grid = np.zeros((nRows,nCols))
    overlaps = []
    gridDicts = []
    locationData = []
    for iLabel,(label,filename) in enumerate(zip(labels,filenames)):
        locationData.append(np.loadtxt(os.path.join(basePath,filename)))

    fig,ax = plt.subplots(1,1)
    for iLabel,(label,filename) in enumerate(zip(labels,filenames)):
        segmentLocationData = locationData[iLabel]
        yOffset = yOffsets[iLabel]
        fails = np.array(segmentLocationData[:,1],dtype=np.bool)
        xs = np.floor(segmentLocationData[:,2])
        ys = np.floor(segmentLocationData[:,3]) + yOffset
        preciseXs = segmentLocationData[:,2]
        preciseYs = segmentLocationData[:,3]+yOffset
        originalPreciseYs = segmentLocationData[:,3]
        goodMask = (~fails) & (preciseXs >= 0) & (preciseXs <=80) & (originalPreciseYs >= 0) & (preciseYs<=125)
        ax.scatter(xs[goodMask],ys[goodMask],label=label,color=colors[iLabel])
        for (x,y) in zip(xs[goodMask],ys[goodMask]):
            if 0 <= x and x < nCols and 0 <= y and y < nRows:
                if grid[y,x] != 0:
                    grid[y,x] = 10
                else:
                    grid[y,x] += iLabel+1
    ax.legend(loc='best')
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[1],ylims[0])

    plotArray(grid,origin='upper')

    #now try to clean up overlapping pixels
    for iLabel,(label,filename) in enumerate(zip(labels,filenames)):
        print iLabel
        yOffset = yOffsets[iLabel]
        resIds = locationData[iLabel][:,0]
        flags = locationData[iLabel][:,1]
        xs = np.floor(locationData[iLabel][:,2])
        ys = np.floor(locationData[iLabel][:,3]) + yOffset
        ys[xs==-1] = -1
        preciseXs = locationData[iLabel][:,2]
        preciseYs = locationData[iLabel][:,3]+yOffset
        fails = np.array(locationData[iLabel][:,1],dtype=np.bool)
        originalPreciseYs = locationData[iLabel][:,3]
        goodMask = (~fails) & (preciseXs >= 0) & (originalPreciseYs >= 0)
        print 'good',np.sum(goodMask)
        print 'res',np.min(resIds[goodMask]),np.max(resIds[goodMask])
        
        for i in range(len(locationData[iLabel])):
            resId = resIds[i]
            if goodMask[i]:
                distVector = (preciseXs[i]-(xs[i]+.5),preciseYs[i]-(ys[i]+.5))
                distMag = np.sqrt(distVector[0]**2 + distVector[1]**2)
                gridDicts.append({'x':xs[i],'y':ys[i],'preciseX':preciseXs[i],'preciseY':preciseYs[i],'distMag':distMag,'segment':iLabel,'resId':resId,'feedline':feedlines[iLabel],'fail':0})

    gridDicts = np.array(gridDicts)

    for entry in gridDicts:
        x = entry['x']
        y = entry['y']
        if x != -1 and y != -1:
            overlapItems = [(item,k) for (k,item) in enumerate(gridDicts) if (item['x'] == entry['x'] and item['y'] == entry['y'])]
            overlapItems,gridIndices = zip(*overlapItems) #unzip
            nPixelsAssigned = len(overlapItems)
            if nPixelsAssigned > 1:
                closestPixelIndex = np.argmin([item['distMag'] for item in overlapItems])
                print nPixelsAssigned, 'pixels assigned to',(x,y)
                for j in range(len(overlapItems)):
                    if j != closestPixelIndex:
                        #print move all the assigned pixels but the closest elsewhere
                        #find the closest open pixel, prefarably in 
                        #the direction the distVector is pointing
                        unassignedCoords = np.where(grid==0)
                        unassignedCoords = zip(unassignedCoords[1],unassignedCoords[0]) #x,y
                        unassignedNeighbors = [coord for coord in unassignedCoords if (np.abs(coord[0]-x)<=1 and np.abs(coord[1]-y)<=1)]

                        try:
                            bestUnassignedNeighbor = np.argmin([np.sqrt((coord[0]+.5-overlapItems[j]['preciseX'])**2 + (coord[1]+.5-overlapItems[j]['preciseY'])**2)])
                            newX = unassignedNeighbors[bestUnassignedNeighbor][0]
                            newY = unassignedNeighbors[bestUnassignedNeighbor][1]
                            print 'best unassigned neighbor',bestUnassignedNeighbor,newX,newY
                            gridDicts[gridIndices[j]]['x'] = newX
                            gridDicts[gridIndices[j]]['y'] = newY
                            grid[newY,newX] = 7
                        except IndexError:
                            print 'no neighbor could be found'
                        except:
                            print 'other error'

    plotArray(grid,origin='upper')
    plt.show()
    print 'next'

    newLocationData = [[entry['resId'],entry['fail'],entry['x'],entry['y']] for entry in gridDicts]
    newLocationData = np.array(newLocationData)
        
    np.savetxt(basePath+outFileName,newLocationData,fmt='%d\t%d\t%d\t%d')
    print 'done!'


