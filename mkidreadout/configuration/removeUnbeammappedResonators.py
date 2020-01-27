"""
Filename:  removeUnbeammappedResonators.py
Author:    Giulia Collura
Date: Sep 27, 2017

#TODO: modernize w/ new data classes, configs, etc.
"""

#Looks at the flag in the Beammap file for each resonator and makes a ps file removing the bad resonators (flag!=0)
#Use: python removeUnbeammappedResonator.py <beammap date [yyyymmdd]> <ps date [yyyymmdd] [optional]>
#If no ps date is given it uses the ps in the $MKID_DATA_DIR folder



import commands
import os
import sys

import numpy as np


#todo make part of a frequency comb object

def removeRes(freqList,beamMapList):
    goodResId=beamMapList[:,0][beamMapList[:,1]==0]
    freqList=[f for f in freqList if f[0] in goodResId]
    return np.array(freqList)


if __name__=="__main__":
    useMsg="Use: python removeUnbeammapedResonators.py <beammap date> <ps folder[optional]>\nExiting"
    basePath='/mnt/data0/Darkness/'
    try:
        beamMapDate=sys.argv[1]
    except:
        print "Error: no beammap date specified.",useMsg
        sys.exit(1)
    
    try:
        path=basePath+sys.argv[2]
    except:
        path = os.environ["MKID_DATA_DIR"]+'/' 
   
    if not os.path.exists(path):
        print "No data dir for the date specified: ", path, "does not exist.",useMsg
        sys.exit(1)
    
    fileList= commands.getoutput('ls ' + path + " | grep ps | grep txt | grep -v clean.cfg")
    if len(fileList)==0:
        print "No power sweeps in", path, useMsg
        sys.exit(1)

    fileList=fileList.split('\n')
    

    beamMapFileName=basePath+beamMapDate+"/Beammap/"+"finalMap_" + beamMapDate + ".txt"
    
    print "Power sweeps found:", fileList
    print "Beam map final map:", beamMapFileName
    
    if not os.path.exists(beamMapFileName):
        print "No beammap final map found in ", beamMapFileName,useMsg 
        sys.exit(1)


    for k,freqF in enumerate(fileList):
        freqFileName=path+"/"+freqF
        print "cleaning file ", k+1, "of", len(fileList), freqFileName
        freqList=np.loadtxt(freqFileName)
        beamMapList=np.loadtxt(beamMapFileName)
        cleanFreqList=removeRes(freqList,beamMapList)
        outFileName=freqFileName.split('.')[0]+"_clean.txt"
        np.savetxt(outFileName,cleanFreqList)
    
    print "done"
