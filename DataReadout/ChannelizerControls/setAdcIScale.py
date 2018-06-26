from Roach2Controls import Roach2Controls 
import sys

if __name__=='__main__':
    ip = '10.0.0.'+str(sys.argv[1])
    scale = float(sys.argv[2])
    roach = Roach2Controls(ip,'DarknessFpga_V2.param',True)
    roach.connect()
    roach.fpga.write_int('adc_in_i_scale', scale*2**7)
