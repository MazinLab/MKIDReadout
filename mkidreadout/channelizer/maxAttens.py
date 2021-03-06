"""
Author: Alex Walter
Date: June 2, 2018

This code sets all the roach attenuators to maximum
Usage: $python maxAttens.py 222 223

"""

import sys
import threading
import time

from mkidreadout.channelizer.Roach2Controls import Roach2Controls


def worker(rNum):
    ip = '10.0.0.'+str(rNum)
    #params = '/home/mecvnc/MKIDReadout/mkidreadout/channelizer/darknessfpga.param'
    roach = Roach2Controls(ip)
    roach.connect()
    roach.changeAtten(0, 31.75)
    roach.changeAtten(1, 31.75)
    roach.changeAtten(2, 31.75)
    roach.changeAtten(3, 31.75)
    del roach
    

def maxAttens(roachNums):
    threads = []
    for rNum in roachNums:
        t=threading.Thread(target=worker, args=(rNum,))
        threads.append(t)
        t.start()
        time.sleep(0.005)#too many files FPGA files opened at once. 
    for t in threads:
        t.join()    #block until they complete
        del t
    del threads
        


if __name__=='__main__':

    roachNums=[]
    for arg in sys.argv[1:]:
        roachNums.append(arg)
    maxAttens(roachNums)



