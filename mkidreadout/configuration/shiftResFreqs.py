import argparse
import os
import glob

from mkidreadout.configuration.sweepdata import SweepMetadata


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata', nargs='+', help='Sweep metadata file pat (argparse will glob for you?)')
    parser.add_argument('-s', '--shift', type=float, default=-30.e3, help='amount to shift freqs (Hz). default: -30 kHz')
    parser.add_argument('-f', '--flag', default='shift', help='flag to be added to outfile name. underscore will be prepended')
    args = parser.parse_args()

    for mdFile in args.metadata:
        md = SweepMetadata(file=mdFile)
        md.freq += args.shift
        fn = os.path.splitext(mdFile)[0] + '_' + args.flag + '.txt'
        md.save(file=fn)
