"""
Author: Alex Walter
Date: Nov 14, 2018

A script for reinitializing the ADC DAC board on select roaches

Usage: $python reinitADCDAC.py 220 221 223

"""


import threading
import sys
import ConfigParser
from mkidreadout.channelizer.InitStateMachine import InitStateMachine
import numpy as np


def worker(rNum, cfgFN='initgui.cfg'):
    config = ConfigParser.ConfigParser()
    config.read(cfgFN)
    roach=InitStateMachine(rNum,config)
    print "r"+str(rNum)+ " Connecting"
    roach.connect()
    print "r"+str(rNum)+ " Reinit"
    roach.roachController.reInitADCDACBoard()
    print "r"+str(rNum)+ " ZDOK cal"
    roach.calZdok()
    print "r"+str(rNum)+ " Done"

def reinitADCDAC(roachNums, cfgFN='initgui.cfg'):
    threads = []
    for rNum in roachNums:
        t=threading.Thread(target=worker, args=(rNum,cfgFN,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()    #block until they complete

if __name__ == "__main__":
    
    args = sys.argv[1:]
    roachNums = np.asarray(args, dtype=np.int)
    reinitADCDAC(roachNums)
    

    





