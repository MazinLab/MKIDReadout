
"""
Author: Alex Walter
Date: June 22, 2018

This file is for dithering OAP4 in SCExAO
"""


import numpy as np
import time
import subprocess

remote = 'scexao@133.40.162.192'
#theta_onaxis=4.03
#phi_onaxis=20.83

def thetaStatus():
    command="ssh "+remote+" /home/scexao/bin/devices/oap4 theta status"
    p=subprocess.Popen(command,shell=True,stdout=subprocess.PIPE)
    out,err = p.communicate()
    theta = out.split(',')[0].rsplit(' ',1)[1]
    return float(theta)

def phiStatus():
    command="ssh "+remote+" /home/scexao/bin/devices/oap4 phi status"
    p=subprocess.Popen(command,shell=True,stdout=subprocess.PIPE)
    out,err = p.communicate()
    phi = out.split(',')[0].rsplit(' ',1)[1]
    return float(phi)

def moveTheta(pos):
    command = "ssh "+remote+" '/home/scexao/bin/devices/oap4 theta goto "+str(pos)+"'"
    p=subprocess.Popen(command,shell=True)
    p.communicate()     #This blocks python until the command finishes executing

def movePhi(pos):
    command = "ssh "+remote+" '/home/scexao/bin/devices/oap4 phi goto "+str(pos)+"'"
    p=subprocess.Popen(command,shell=True)
    p.communicate()

#def moveOnAxis():
#    moveTheta(theta_onaxis)
#    movePhi(phi_onaxis)

def moveOnAxis():
    command="ssh "+remote+" /home/scexao/bin/devices/oap4 onaxis"
    p=subprocess.Popen(command,shell=True,stdout=subprocess.PIPE)
    out,err = p.communicate()

def dither(outputPath='.', nSteps=5, startX=4.045, endX=4.005, startY=20.605, endY=21.295, intTime=1.):
    """
    
    INPUT:
        outputPath - Path to save log file in. ie. '/home/data/ScienceData/Subaru/20180622/'. The filename is *timestamp*_dither.log
        nSteps - Number of steps in x and y direction. There will be nSteps**2 points in a grid
        startX - theta direction
        endX - 
        startY - phi direction
        endY - 
        intTime - Time to integrate at each x/y location
    """
    x_list = np.linspace(startX, endX,nSteps)
    y_list = np.linspace(startY, endY,nSteps)
    startTimes=[]
    endTimes=[]
    for x in x_list:
        for y in y_list:
            moveTheta(x)
            movePhi(y)
            startTimes.append(time.time())
            time.sleep(intTime)
            endTimes.append(time.time())
    moveOnAxis()

    cfg = open(outputPath+str(int(np.round(startTimes[0])))+'_dither.log','w')
    cfg.write('startTimes = '+str(startTimes) + '\n')
    cfg.write('endTimes = '+str(endTimes) + '\n')
    cfg.write('xPos = '+str(x_list) + '\n')
    cfg.write('yPos = '+str(y_list) + '\n')
    cfg.write('intTime = '+str(intTime) + '\n')
    cfg.write('nSteps = '+str(nSteps) + '\n')
    cfg.close()


if __name__=='__main__':
    print thetaStatus()
    print phiStatus()
    #dither('/home/data/ScienceData/Subaru/20180625/',intTime=20)
    #moveOnAxis()












