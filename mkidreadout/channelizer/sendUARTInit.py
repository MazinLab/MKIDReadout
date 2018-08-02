# TODO Merge with roach2controls main

import Roach2Controls as r

roach = r.Roach2Controls('10.0.0.112', 'DarknessFpga_V2.param')
roach.connect()
roach.sendUARTCommand(1)
