"""
Author: Alex Walter
Date: June 2, 2018

This code sets all the roach attenuators to maximum
Usage: $python maxAttens.py 222 223

"""

import sys
import threading
from mkidreadout.channelizer.Roach2Controls import Roach2Controls

def worker(rNum, params='darknessfpga.param'):
    ip = '10.0.0.'+str(rNum)
    #params = '/home/mecvnc/MKIDReadout/mkidreadout/channelizer/darknessfpga.param'
    roach = Roach2Controls(ip, params, True)
    roach.connect()
    roach.changeAtten(0, 31.75)
    roach.changeAtten(1, 31.75)
    roach.changeAtten(2, 31.75)
    roach.changeAtten(3, 31.75)

def maxAttens(roachNums, params='darknessfpga.param'):
    threads = []
    for rNum in roachNums:
        t=threading.Thread(target=worker, args=(rNum,params,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()    #block until they complete


if __name__=='__main__':

    roachNums=[]
    for arg in sys.argv[1:]:
        roachNums.append(arg)
    maxAttens(roachNums)



