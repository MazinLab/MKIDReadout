"""
Author: Alex Walter
Date: June 2, 2018

This code sets all the roach attenuators to maximum
Usage: $python maxAttens.py 222 223

"""

import sys
from mkidreadout.channelizer.Roach2Controls import Roach2Controls


if __name__=='__main__':

    for arg in sys.argv[1:]:
        ip = '10.0.0.'+arg
        params = 'darknessfpga.param'
        roach = Roach2Controls(ip, params, True)
        roach.connect()
        roach.changeAtten(0, 31.75)
        roach.changeAtten(1, 31.75)
        roach.changeAtten(2, 31.75)
        roach.changeAtten(3, 31.75)



