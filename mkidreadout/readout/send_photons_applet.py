#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import time
from datetime import datetime
import threading

import mkidcore.corelog
import mkidcore.instruments
import mkidreadout.config
import mkidreadout.hardware.hsfw
from mkidcore.corelog import create_log, getLogger
from mkidcore.objects import Beammap
from mkidreadout.channelizer.Roach2Controls import Roach2Controls


class MKIDSendPhotonsApplet(threading.Thread):
    def __init__(self, roachNums, config='./dashboard.yml'):
        """
        INPUTS:
            roachNums - List of roach numbers to connect with
            config - the configuration file. See ConfigParser doc for making configuration file
            parent -
        """
        super(threading.Thread, self).__init__()
        self.config = mkidreadout.config.load(config)
        self.sending = False
        self.roaches = []
        self.beammap = None

        # Connect to ROACHES and initialize network port in firmware
        getLogger('Dashboard').info('Connecting roaches and loading beammap...')
        for roachNum in roachNums:
            roach = Roach2Controls(self.config.roaches.get('r{}.ip'.format(roachNum)),
                                   self.config.roaches.fpgaparamfile, num=roachNum,
                                   feedline=self.config.roaches.get('r{}.feedline'.format(roachNum)),
                                   range=self.config.roaches.get('r{}.range'.format(roachNum)),
                                   verbose=False, debug=False)
            if not roach.connect() and not roach.issetup:
                raise RuntimeError('Roach r{} has not been setup.'.format(roachNum))
            roach.setPhotonCapturePort(self.config.packetmaster.captureport)
            self.roaches.append(roach)
        for roach in self.roaches:
            roach.loadCurTimestamp()

    def stop_photon_send(self):
        """
        Tells roaches to stop photon capture
        """
        for roach in self.roaches:
            roach.stopSendingPhotons()
        getLogger('photon_send_control').info('Roaches stopped sending photon packets')

    def start_photon_send(self, beammap):
        """
        Tells roaches to start photon capture

        Have to be careful to set the registers in the correct order in case we are currently in phase capture mode
        """
        self.load_beammap(beammap)
        for roach in self.roaches:
            roach.startSendingPhotons(self.config.packetmaster.ip, self.config.packetmaster.captureport)
            roach.setMaxCountRate(self.config.dashboard.roach_cpslim)
        getLogger('photon_send_control').info('Roaches sending photon packets!')

    def load_beammap(self, beammap):
        """ This function loads the beam map into the roach firmware"""
        for roach in self.roaches:
            ffile = roach.tagfile(self.config.roaches.get('r{}.freqfileroot'.format(roach.num)),
                                  dir=self.config.paths.setup)
            roach.setLOFreq(self.config.roaches.get('r{}.lo_freq'.format(roach.num)))
            roach.loadBeammapCoords(beammap, freqListFile=ffile)
        getLogger('photon_send_control').info('Loaded beam map into roaches')

    def run(self):
        try:
            while True:
                if os.path.exists(self._send_phot_file):
                    if self.sending:
                        time.sleep(1)
                        continue

                    try:
                        with open(self._send_phot_file) as f:
                            beammap_file = f.readline()
                        beammap = Beammap(beammap_file)
                        getLogger('photon_send_control').info('Loaded beammap: %s', beammap_file)
                        self.start_photon_send(beammap)
                        self.sending = True
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        getLogger('photon_send_control').error('Cannot start sending photons due to {}'.format(str(e)))

                elif self.sending:
                    self.stop_photon_send()
                    self.sending = False

        except KeyboardInterrupt:
            getLogger('photon_send_control').info('Shutting down due to keyboard interrupt')
            self.stop_photon_send()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MKID Photon Send Applet')
    parser.add_argument('-a', action='store_true', default=False, dest='all_roaches',
                        help='Run with all roaches for instrument in cfg')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--alla', action='store_true', help='Run with all range A roaches in config')
    group.add_argument('--allb', action='store_true', help='Run with all range B roaches in config')

    parser.add_argument('-r', nargs='+', type=int, help='Roach numbers', dest='roaches')
    parser.add_argument('-c', '--config', default=mkidreadout.config.DEFAULT_DASHBOARD_CFGFILE, dest='config',
                        type=str, help='The config file')
    parser.add_argument('--gencfg', default=False, dest='genconfig', action='store_true',
                        help='generate configs in CWD')

    args = parser.parse_args()

    if args.genconfig:
        mkidreadout.config.generate_default_configs(dashboard=True)
        exit(0)

    config = mkidreadout.config.load(args.config)

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
    create_log('photon_send_control',
               logfile=os.path.join(config.paths.logs, 'dashboard_{}.log'.format(timestamp)),
               console=True, mpsafe=True, propagate=False,
               fmt='%(asctime)s Dashboard %(levelname)s: %(message)s',
               level=mkidcore.corelog.DEBUG)
    create_log('mkidreadout',
               console=True, mpsafe=True, propagate=False,
               fmt='%(asctime)s %(funcName)s: %(levelname)s %(message)s',
               level=mkidcore.corelog.DEBUG)
    create_log('mkidcore',
               console=True, mpsafe=True, propagate=False,
               fmt='%(asctime)s mkidcore.x.%(funcName)s: %(levelname)s %(message)s',
               level='INFO')

    if args.alla:
        roaches = mkidcore.instruments.ROACHESA[config.instrument]
    elif args.allb:
        roaches = mkidcore.instruments.ROACHESB[config.instrument]
    elif args.all_roaches:
        roaches = mkidcore.instruments.ROACHES[config.instrument]
    else:
        roaches = args.roaches

    if not roaches:
        try:
            roaches = config.roaches.in_use
        except AttributeError:
            getLogger('Dashboard').error('No roaches specified')
            exit()
    applet = MKIDSendPhotonsApplet(roaches, config=config)
    applet.run()
