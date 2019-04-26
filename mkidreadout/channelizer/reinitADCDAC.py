#!/bin/env python
"""
Author: Alex Walter
Date: Nov 14, 2018

A script for reinitializing the ADC DAC board on select roaches

Usage: $python reinitADCDAC.py 220 221 223

"""
from __future__ import print_function
import threading, argparse
from mkidreadout.channelizer.InitStateMachine import InitStateMachine
import mkidreadout.config


def worker(rNum, config):
    roach=InitStateMachine(rNum, config)
    print("r" + str(rNum) + " Connecting")
    roach.connect()
    print("r" + str(rNum) + " Reinit")
    roach.roachController.reInitADCDACBoard()
    print("r" + str(rNum) + " ZDOK cal")
    roach.calZdok()
    print("r" + str(rNum) + " Done")
    del roach


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MKID Reinit ADCDAC')
    parser.add_argument('roaches', nargs='+', type=int, help='Roach numbers')
    parser.add_argument('-c', '--config', default=mkidreadout.config.DEFAULT_INIT_CFGFILE, dest='config',
                        type=str, help='The config file')

    args = parser.parse_args()

    config = mkidreadout.config.load(args.config)

    threads = []
    for n in args.roaches:
        t = threading.Thread(target=worker, args=(n, config))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()  # block until they complete

    

    





