import time
import sys
import numpy as np
import PicomotorClient as pmc

def moveLeft(dx):
    pmc.moveMotor(1,int(dx))
    return

def moveUp(dy):
    pmc.moveMotor(3,int(dy))
    return

#name of cfg file for quick stacker
fileTs = int(time.time())
ditherFile= '/mnt/data0/DarknessPipeline/QuickLook/ditherStack_%s.cfg'%fileTs
date = '20170409'
binDir = '/mnt/data0/ScienceData/'
imgDir = '/mnt/ramdisk/'
outputDir = '/mnt/data0/ProcessedData/seth/imageStacks/PAL2017a/'
darkList = [1491907540,1491907560]
#darkList = [0,0]
darkBool = 'True'

#scale to convert from picomotor counts to darkness pixels (roughly N_pmcounts / Npixels in x direction)
scale = 70000./88.55
theta = 0.066

#time to wait for picomotor move
moveTime = 2 #should scale with size of move, but hardcoded for now

if len(sys.argv)<6:
    print "Syntax: ->python ditherScript.py int(N_x) int(N_y) int(dx in pix) int(dy in pix) int(int time per frame)"
    sys.exit()

nx = int(sys.argv[1])
ny = int(sys.argv[2])
dx = int(sys.argv[3])
dy = int(sys.argv[4])
dt = int(sys.argv[5])
print nx, ny, dx, dy, dt

dx*=scale #convert rough dx and dy in pixels desired to picomotor counts
dy*=scale

startTimeList = []
stopTimeList = []
xPosList = []
yPosList = []
curXPos = 0
curYPos = 0
yMoveSign = 1

pixXPos=0
pixYPos=0

#keep track of how many dither positions we took images at
Npos=0

for i in range(nx+1):
    newXPos = i*dx
    xMove = newXPos-curXPos

    if xMove != 0:
        print "Moving x by %i"%xMove
        #move left
        moveLeft(xMove)
        #wait for move to complete
        time.sleep(moveTime)
        #move up

    curXPos = newXPos
    pixXPos += xMove*np.cos(np.deg2rad(theta))/scale
    pixYPos += -1.0*xMove*np.sin(np.deg2rad(theta))/scale
        
    for j in range(ny+1):
        if yMoveSign == 1:
            newYPos = j*dy
        elif yMoveSign == -1:
            newYPos = (ny-j)*dy
    
        yMove = newYPos - curYPos
        if yMove !=0:
            print "Moving y by %i"%yMove
            #move up/down
            moveUp(yMove)
            #wait for move to complete
            time.sleep(moveTime)

        curYPos = newYPos
        pixYPos += yMove*np.cos(np.deg2rad(theta))/scale
        pixXPos += yMove*np.sin(np.deg2rad(theta))/scale
        #write current x and y positions to list
        yPosList.append(str(int(pixYPos)))
        xPosList.append(str(int(pixXPos)))

        print "Position #%i:"%Npos
        print "Taking data at (%i,%i)"%(curXPos, curYPos)
        #grab time for start time of data at this position
        startTimeList.append(str(int(time.time())))
        #wait for integration time at this position
        time.sleep(dt)
        #get end timestamp here
        stopTimeList.append(str(int(time.time())))
        #add 1 to the number of positions we've taken data at
        Npos+=1

    #reverse Y direction after every Y loop
    yMoveSign*=-1
        
#at end of loop, move back to (0,0)
print "Finished loop, moving to (0,0)..."

if curYPos!=0:
    print "Moving y by %i"%(-1*curYPos)
    moveUp(-1*curYPos)
    time.sleep(moveTime*ny)

if curXPos!=0:
    print "Moving x by %i"%(-1*curXPos)
    moveLeft(-1*curXPos)
    time.sleep(moveTime*nx)


cfg = open(ditherFile,'w')
cfg.write('nPos = %i'%Npos + '\n')
cfg.write('startTimes = '+str(startTimeList) + '\n')
cfg.write('stopTimes = '+str(stopTimeList) + '\n')
cfg.write('xPos = '+str(xPosList) + '\n')
cfg.write('yPos = '+str(yPosList) + '\n')
cfg.write('darkSpan = '+str(darkList) + '\n')
cfg.write('flatSpan = '+str(darkList) + '\n')
cfg.write('divideFlat = False\n')
cfg.write('subtractDark = %s\n'%darkBool)
cfg.write('doHPM = False\n')
cfg.write('coldCut = 0\n')
cfg.write('fitPos = False\n')
cfg.write('upSample = 1\n')
cfg.write('padFraction = 0.4\n')
cfg.write('useImg = True\n')
cfg.write('target = \'dither%s\'\n'%fileTs)
cfg.write('numRows = 125\n')
cfg.write('numCols = 80\n')
cfg.write('date = \'%s\'\n'%str(date))
cfg.write('binDir = \'%s\'\n'%str(binDir))
cfg.write('imgDir = \'%s\'\n'%str(imgDir))
cfg.write('outputDir = \'%s\'\n'%str(outputDir))
cfg.write('refFile = None\n')
cfg.close()
        

