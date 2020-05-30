# TODO Merge with roach2controls main

import Roach2Controls as r
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument('roaches', nargs='+')
args = parser.parse_args()

for roach in args.roaches:
    roach = r.Roach2Controls('10.0.0.' + str(roach), 'darknessfpga.param')
    roach.connect()
    roach.sendUARTCommand(1)
