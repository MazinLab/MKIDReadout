#!/usr/bin/env python
"""
Script for making adjustments to frequency lists (sweep metadata)
after initial ML fitting. Includes finding LOs, clipping, and shifting freqs.

Author: Neelay Fruitwala
"""
import argparse
import os, sys, glob
import numpy as np

import mkidcore.corelog
from mkidcore.corelog import getLogger, create_log

import mkidcore.sweepdata as sd

from clipResAttens import clipHist
from findLOs import findLOs

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata' , help='Sweep metadata file pat (can use same tags as sweep)')
    parser.add_argument('-s', '--sweep', help='Sweep npz file. Can be a pattern specifying\
            multiple files, with tags {roach}, {feedline}, and/or {range}. Required for finding LOs\
            (for now). Sweep/MD files must have at least one tag in common; if this bothers you\
            yell at Neelay')
    parser.add_argument('--freq-shift', type=float, default=0, help='amount to shift freqs (Hz)')
    parser.add_argument('--find-lo', action='store_true', help='runs LO finding script. sweep npz \
            required to determine initial sweep lo')
    parser.add_argument('--clip-atten', action='store_true', help='runs atten clipping GUI')
    parser.add_argument('--config', default=None, help='roach.yml file for storing lo freqs. WILL BE OVERWRITTEN')
    parser.add_argument('--flag', default='proc', help='flag to be added to outfile name. underscore will be prepended')
    args = parser.parse_args()
    getLogger(__name__, setup=True)
    getLogger(__name__).setLevel(mkidcore.corelog.INFO)
    create_log('parse',
               console=True, mpsafe=True, propagate=False,
               fmt='%(asctime)s %(name)s %(funcName)s: %(levelname)s %(message)s ',
               level=mkidcore.corelog.INFO)
    create_log('mkidreadout.configuration.findLOs',
               console=True, mpsafe=True, propagate=False,
               fmt='%(asctime)s %(name)s %(funcName)s: %(levelname)s %(message)s ',
               level=mkidcore.corelog.INFO)

    if args.sweep is not None:
        sweepFiles, mdFiles, paramDicts = sd.matchSweepToMetadataPat(args.sweep, args.metadata)
    else:
        mdFiles, paramDicts = sd.getSweepFilesFromPat(args.metadata)
        if args.find_lo:
            raise Exception('LO finding requires sweep npz files')

    sweepList = []
    mdList = []

    if args.sweep is not None:
        for sweepFile, mdFile in zip(sweepFiles, mdFiles):
            sweepList.append(sd.FreqSweep(sweepFile))
            mdList.append(sd.SweepMetadata(file=mdFile))
    else:
        for mdFile in mdFiles:
            mdList.append(sd.SweepMetadata(file=mdFile))

    # find LOs
    if args.find_lo:
        los = np.nan*np.zeros(len(sweepFiles))
        for i, (sweep, md, paramDict) in enumerate(zip(sweepList, mdList, paramDicts)):
            if np.isnan(los[i]):
                if 'feedline' in paramDict:
                    if isinstance(paramDict['feedline'], int) and paramDict['feedline'] != md.feedline:
                        raise Exception('Filename and metadata feedlines dont match! fn: {}; md: {}'.format(md.feedline, paramDict['feedline']))
                mdb = None
                sweepb = None
                for j in range(i + 1, len(sweepFiles)): #check the rest of the files to see if there is one on same fl
                    if mdList[j].feedline == md.feedline: #we've found the other board
                        mdb = mdList[j]
                        sweepb = sweepList[j]
                        if isinstance(paramDict['feedline'], int) and paramDict['feedline'] != mdb.feedline:
                            raise Exception('Filename and metadata feedlines dont match!')
                        break

                if mdb:
                    getLogger(__name__).info('Finding LOs for: {}; {}'.format(paramDict['roach'], paramDicts[j]['roach']))
                    if md.freq[0] < mdb.freq[0]:
                        los[i], los[j] = findLOs(md.freq, sweep.lo, mdb.freq, sweepb.lo)
                    else:
                        los[j], los[i] = findLOs(mdb.freq, sweepb.lo, md.freq, sweep.lo)

                else:
                    getLogger(__name__).info('Finding LOs for: {}'.format(paramDict['roach']))
                    los[i] = findLOs(md.freq)

        if args.config is not None:
            cfg = mkidcore.config.load(args.config)
            for lo, sweep, md, paramDict in zip(los, sweepList, mdList, paramDicts):
                if isinstance(paramDict['roach'], int) and paramDict['roach'] != 0:
                    rNum = paramDict['roach']
                else:
                    rNum = getRoachNumberFromFile(sweep.file)
                cfg.register('r{}.lo_freq'.format(rNum), float(lo), update=True)

            cfg.save(args.config)

    if args.clip_atten:
        for i,md in enumerate(mdList):
            mdList[i] = clipHist(md)

    if args.freq_shift:
        getLogger(__name__).info('shifting frequencies by {} Hz'.format(args.freq_shift))
        for i, md in enumerate(mdList):
            mdList[i].freq += args.freq_shift

    if args.freq_shift or args.clip_atten: #only save if we modified list
        for md in mdList:
            md.save(os.path.splitext(md.file)[0] + '_' + args.flag + '.txt')
