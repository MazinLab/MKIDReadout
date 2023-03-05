#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import threading
import mkidreadout.config
from mkidcore.corelog import create_log, getLogger
from mkidcore.objects import Beammap
from mkidreadout.channelizer.Roach2Controls import Roach2Controls

log = getLogger('send_photons_applet')


def connect_roaches(config):
    roaches = []
    for roachNum in config.roaches.in_use:
        roach = Roach2Controls(config.roaches.get('r{}.ip'.format(roachNum)),
                               config.roaches.fpgaparamfile, num=roachNum,
                               feedline=config.roaches.get('r{}.feedline'.format(roachNum)),
                               range=config.roaches.get('r{}.range'.format(roachNum)),
                               verbose=False, debug=False)
        if not roach.connect() and not roach.issetup:
            log.critical('Roach r{} setup failed'.format(roachNum))
            continue
        roach.setPhotonCapturePort(config.packetmaster.captureport)
        roaches.append(roach)
    for roach in roaches:
        roach.loadCurTimestamp()
    return roaches


def start_photon_send(config, roaches):
    """
    Tells roaches to start photon capture

    Have to be careful to set the registers in the correct order in case we are currently in phase capture mode
    """
    for roach in roaches:
        roach.startSendingPhotons(config.packetmaster.ip, config.packetmaster.captureport)
        roach.setMaxCountRate(config.dashboard.roach_cpslim)


def load_beammap(config, roaches):
    """ This function loads the beam map into the roach firmware"""
    for roach in roaches:
        ffile = roach.tagfile(config.roaches.get('r{}.freqfileroot'.format(roach.num)), dir=config.paths.setup)
        roach.setLOFreq(config.roaches.get('r{}.lo_freq'.format(roach.num)))
        roach.loadBeammapCoords(config.beammap, freqListFile=ffile)


class MKIDSendPhotonsApplet(threading.Thread):
    def __init__(self, send_file):
        """
        INPUTS:
            roachNums - List of roach numbers to connect with
            config - the configuration file. See ConfigParser doc for making configuration file
            parent -
        """
        super(threading.Thread, self).__init__()
        self._send_photons_file = send_file

    def run(self):
        active_cfg_file = ''
        config = None
        roaches = []
        sending = False
        try:
            while True:
                try:
                    with open(self._send_photons_file, 'r') as f:
                        cfg_file = f.readline()
                    if cfg_file != active_cfg_file:
                        config = mkidreadout.config.load(cfg_file)
                        log.info('Loaded {} to send photons'.format(cfg_file))
                        roaches = connect_roaches(config)
                except Exception as e:
                    time.sleep(.25)
                    continue

                while os.path.exists(self._send_photons_file):
                    if not sending:
                        try:
                            load_beammap(config, roaches)
                            start_photon_send(config, roaches)
                            sending = True
                        except Exception as e:
                            log.error('Cannot start sending photons due to {}'.format(e))
                    time.sleep(.5)

                if sending:
                    for roach in roaches:
                        roach.stopSendingPhotons()
                    sending = False

        except Exception as e:
            log.info('Shutting down due to {}'.format(e))
            for roach in roaches:
                roach.stopSendingPhotons()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MKID Photon Send Applet')
    parser.add_argument('-c', default='./SEND_PHOTONS', dest='file', required=True,
                        type=str, help='The send flag file')
    args = parser.parse_args()

    create_log('photon_send_control',
               console=True, mpsafe=True, propagate=False,
               fmt='%(levelname)s: %(message)s', level='DEBUG')
    create_log('mkidreadout', console=True, mpsafe=True, propagate=False,
               fmt='%(funcName)s: %(levelname)s %(message)s', level='DEBUG')
    create_log('mkidcore', console=True, mpsafe=True, propagate=False,
               fmt='mkidcore.x.%(funcName)s: %(levelname)s %(message)s', level='INFO')

    applet = MKIDSendPhotonsApplet(args.file)
    applet.run()
