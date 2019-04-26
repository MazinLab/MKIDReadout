# TODO Merge with roach2controls main
import sys

from Roach2Controls import Roach2Controls

if __name__=='__main__':
    args = sys.argv[1:]
    for arg in args:
        ip = '10.0.0.'+str(arg)
        roach = Roach2Controls(ip,'darknessfpga.param',True)
        roach.connect()
        roach.setAdcScale(0.9375)
