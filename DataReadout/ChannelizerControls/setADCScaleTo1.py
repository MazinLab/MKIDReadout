from Roach2Controls import Roach2Controls 
import sys

if __name__=='__main__':
    args = sys.argv[1:]
    for arg in args:
        ip = '10.0.0.'+str(arg)
        roach = Roach2Controls(ip,'DarknessFpga_V2.param',True)
        roach.connect()
        roach.setAdcScale(0.9375)
