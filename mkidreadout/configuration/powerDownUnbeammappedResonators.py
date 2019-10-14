import os
import sys
import argparse
from glob import glob
import numpy as np
from mkidcore.objects import Beammap
from mkidreadout.configuration.sweepdata import SweepMetadata


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepat', nargs='+', help='Pattern for matching frequency file(s)')
    parser.add_argument('-b', '--beammap', help='Beammap file')
    args = parser.parse_args()

    #freqFiles = glob(args.filepat)
    print 'found: ', args.filepat
    beammap = Beammap(args.beammap, (140, 146))

    for fl in args.filepat:
        md = SweepMetadata(file=fl)
        md.powerDownUnbeammappedRes(beammap)
        md.save(fl.split('.')[0] + '_pdubr.txt')




    
