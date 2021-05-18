"""
Author: Alex Walter
Date: Jul 8, 2016

This class grabs info from the Palomar telescope
"""
from __future__ import print_function

import datetime
import math
import time
from socket import *

import ephem

from mkidcore.corelog import getLogger

__all__ = ['Telescope']

import os
import subprocess

SUBARU = {'client': None, 'time': 0, 'cache': None}


FITS_2_G2CAM = {'AIRMASS': 'FITS.SBR.AIRMASS',
 'ALTITUDE': 'FITS.SBR.ALTITUDE',
 'AUTOGUID': 'FITS.SBR.AUTOGUID',
 'AZIMUTH': 'FITS.SBR.AZIMUTH',
 'D_ADFG': 'AON.RTS.ADFGAIN',
 'D_DMCMTX': 'AON.RTS.DMCMTX',
 'D_DMGAIN': 'AON.RTS.DMGAIN',
 'D_HDFG': 'AON.RTS.HDFGAIN',
 'D_HTTG': 'AON.RTS.HTTGAIN',
 'D_HWADP': 'AON.HWFS.ADC.POS',
 'D_HWADPA': 'AON.HWFS.ADC.PA',
 'D_HWADRA': 'AON.HWFS.ADC.RA',
 'D_HWADST': 'AON.HWFS.ADC.STAT',
 'D_HWAF1': 'AON.HWFS.AFW1',
 'D_HWAF1P': 'AON.HWFS.AFW1.POS',
 'D_HWAF2': 'AON.HWFS.AFW2',
 'D_HWAF2P': 'AON.HWFS.AFW2.POS',
 'D_HWAPDA': 'AON.HWFS.APDAV',
 'D_HWHBS': 'AON.HWFS.HBS',
 'D_HWHBSP': 'AON.HWFS.HBS.POS',
 'D_HWLAF': 'AON.HWFS.LAFW',
 'D_HWLAFP': 'AON.HWFS.LAFW.POS',
 'D_HWLAP': 'AON.HWFS.LGSAP',
 'D_HWLAPP': 'AON.HWFS.LGSAP.POS',
 'D_HWLASH': 'AON.HWFS.LASH',
 'D_HWLAZ': 'AON.HWFS.LAZ',
 'D_HWLAZP': 'AON.HWFS.LAZ.POS',
 'D_HWNAP': 'AON.HWFS.NGSAP',
 'D_HWNAPP': 'AON.HWFS.NGSAP.POS',
 'D_HWPBS': 'AON.HWFS.PBS',
 'D_HWPBSP': 'AON.HWFS.PBS.POS',
 'D_IMR': 'AON.IMR.STAT',
 'D_IMRANG': 'AON.IMR.ANGLE',
 'D_IMRDEC': 'AON.IMR.DEC',
 'D_IMRMOD': 'AON.IMR.MODE',
 'D_IMRPAD': 'AON.IMR.PAD',
 'D_IMRPAP': 'AON.IMR.PAP',
 'D_IMRRA': 'AON.IMR.RA',
 'D_LDFG': 'AON.RTS.LDFGAIN',
 'D_LOOP': 'AON.RTS.LOOP',
 'D_LTTG': 'AON.RTS.LTTGAIN',
 'D_PSUBG': 'AON.RTS.PSUBGAIN',
 'D_STTG': 'AON.RTS.STTGAIN',
 'D_TTCMTX': 'AON.RTS.TTCMTX',
 'D_TTGAIN': 'AON.RTS.TTGAIN',
 'D_TTX': 'AON.TT.TTX',
 'D_TTY': 'AON.TT.TTY',
 'D_VMAP': 'AON.HWFS.VMAP',
 'D_VMAPS': 'AON.HWFS.VMAP.SIZE',
 'D_VMDRV': 'AON.VM.DRIVE',
 'D_VMFREQ': 'AON.VM.FREQ',
 'D_VMPHAS': 'AON.VM.PHASE',
 'D_VMVOLT': 'AON.VM.VOLT',
 'D_WTTC1': 'AON.TT.WTTC1',
 'D_WTTC2': 'AON.TT.WTTC2',
 'D_WTTG': 'AON.RTS.WTTGAIN',
 'DATE-OBS': 'FITS.SBR.DATE-OBS',
 'DEC': 'FITS.SBR.DEC',
 'DOM-HUM': 'FITS.SBR.DOM-HUM',
 'DOM-PRS': 'FITS.SBR.DOM-PRS',
 'DOM-TMP': 'FITS.SBR.DOM-TMP',
 'DOM-WND': 'FITS.SBR.DOM-WND',
 'EQUINOX': 'FITS.SBR.EQUINOX',
 'FOC-POS': 'FITS.CRS.FOC-POS',
 'FOC-VAL': 'FITS.SBR.FOC-VAL',
 'HST': 'FITS.SBR.HST',
 'M2-TIP': 'FITS.SBR.M2-TIP',
 'M2-TYPE': 'FITS.SBR.M2-TYPE',
 'MJD': 'FITS.SBR.MJD',
 'OBJECT': 'FITS.CRS.OBJECT',
 'OBS-ALOC': 'FITS.AON.OBS-ALOC',
 'OBS-MOD': 'FITS.AON.OBS-MOD',
 'OBSERVAT': 'FITS.SBR.OBSERVAT',
 'OBSERVER': 'FITS.AON.OBSERVER',
 'OUT-HUM': 'FITS.SBR.OUT-HUM',
 'OUT-PRS': 'FITS.SBR.OUT-PRS',
 'OUT-TMP': 'FITS.SBR.OUT-TMP',
 'OUT-WND': 'FITS.SBR.OUT-WND',
 'P_RTAGL1': 'WAV.RT_ANG1',
 'P_RTAGL2': 'WAV.RT_ANG2',
 'P_STGPS1': 'WAV.STG1_PS',
 'P_STGPS2': 'WAV.STG2_PS',
 'P_STGPS3': 'WAV.STG3_PS',
 'PROP-ID': 'FITS.AON.PROP-ID',
 'RA': 'FITS.SBR.RA',
 'RET-ANG1': 'WAV.RT_ANG1',
 'RET-ANG2': 'WAV.RT_ANG2',
 'TELESCOP': 'FITS.SBR.TELESCOP',
 'TELFOCUS': 'FITS.SBR.TELFOCUS',
 'UT': 'FITS.SBR.UT',
 'X_BUFPKO': 'SCX.BUFFY.PKO',
 'X_BUFPKP': 'SCX.BUFFY.PKO.POS',
 'X_BUFPUP': 'SCX.BUFFY.PUP',
 'X_CHAPKO': 'SCX.CHARIS.PKO',
 'X_CHAPKP': 'SCX.CHARIS.PKO.POS',
 'X_CHAPKT': 'SCX.CHARIS.PKO.THETA',
 'X_CHAWOL': 'SCX.CHARIS.WOL',
 'X_CHKPUF': 'SCX.CHUCK.PUP.F',
 'X_CHKPUP': 'SCX.CHUCK.PUP',
 'X_CHKPUS': 'SCX.CHUCK.PUP.ST',
 'X_COMPPL': 'SCX.COMPPLATE',
 'X_DICHRO': 'SCX.DICHRO',
 'X_DICHRP': 'SCX.DICHRO.POS',
 'X_FINPKO': 'SCX.FIBINJ.PKO',
 'X_FINPKP': 'SCX.FIBINJ.PKO.POS',
 'X_FIRPKO': 'SCX.FIRST.PKO',
 'X_FIRPKP': 'SCX.FIRST.PKO.POS',
 'X_FPM': 'SCX.FPM',
 'X_FPMF': 'SCX.FPM.F',
 'X_FPMWHL': 'SCX.FPM.WHL',
 'X_FPMX': 'SCX.FPM.X',
 'X_FPMY': 'SCX.FPM.Y',
 'X_FST': 'SCX.FSTOP',
 'X_FSTX': 'SCX.FSTOP.X',
 'X_FSTY': 'SCX.FSTOP.Y',
 'X_GRDAMP': 'SCX.GRID.AMP',
 'X_GRDMOD': 'SCX.GRID.MOD',
 'X_GRDSEP': 'SCX.GRID.SEP',
 'X_GRDST': 'SCX.GRID.STAT',
 'X_HOTSPT': 'SCX.HOTSPOT',
 'X_INTSPH': 'SCX.INTSPH',
 'X_IPIAA': 'SCX.IPIAA',
 'X_IPIPHI': 'SCX.IPIAA.PHI',
 'X_IPITHE': 'SCX.IPIAA.THETA',
 'X_IPIX': 'SCX.IPIAA.X',
 'X_IPIY': 'SCX.IPIAA.Y',
 'X_IRCBLK': 'SCX.IRCAM.BLK',
 'X_IRCFC2': 'SCX.IRCAM.FCS.F2',
 'X_IRCFCP': 'SCX.IRCAM.FCS.F1',
 'X_IRCFCS': 'SCX.IRCAM.FCS',
 'X_IRCFLC': 'SCX.IRCAM.FLC',
 'X_IRCFLP': 'SCX.IRCAM.FLC.POS',
 'X_IRCFLT': 'SCX.IRCAM.FLT',
 'X_IRCHPP': 'SCX.IRCAM.HWP.POS',
 'X_IRCHWP': 'SCX.IRCAM.HWP',
 'X_IRCPUP': 'SCX.IRCAM.PUPIL',
 'X_IRCPUX': 'SCX.IRCAM.PUPIL.X',
 'X_IRCPUY': 'SCX.IRCAM.PUPIL.Y',
 'X_IRCQWP': 'SCX.IRCAM.QWP',
 'X_IRCWOL': 'SCX.IRCAM.WOL',
 'X_LOWBLK': 'SCX.LOWFS.BLK',
 'X_LOWFCS': 'SCX.LOWFS.FCS',
 'X_LOWFRQ': 'SCX.LOWFS.FREQ',
 'X_LOWGN': 'SCX.LOWFS.GAIN',
 'X_LOWLK': 'SCX.LOWFS.LEAK',
 'X_LOWLP': 'SCX.LOWFS.LOOP',
 'X_LOWMOT': 'SCX.LOWFS.MOT',
 'X_LOWNMO': 'SCX.LOWFS.NMO',
 'X_LYOT': 'SCX.LYOT',
 'X_LYOWHL': 'SCX.LYOT.WHL',
 'X_LYOX': 'SCX.LYOT.X',
 'X_LYOY': 'SCX.LYOT.Y',
 'X_MKIPKO': 'SCX.MKIDS.PKO',
 'X_MKIPKP': 'SCX.MKIDS.PKO.POS',
 'X_MKIPKT': 'SCX.MKIDS.PKO.THETA',
 'X_NPS11': 'SCX.NPS.NPS11',
 'X_NPS12': 'SCX.NPS.NPS12',
 'X_NPS13': 'SCX.NPS.NPS13',
 'X_NPS14': 'SCX.NPS.NPS14',
 'X_NPS15': 'SCX.NPS.NPS15',
 'X_NPS16': 'SCX.NPS.NPS16',
 'X_NPS17': 'SCX.NPS.NPS17',
 'X_NPS18': 'SCX.NPS.NPS18',
 'X_NPS21': 'SCX.NPS.NPS21',
 'X_NPS22': 'SCX.NPS.NPS22',
 'X_NPS23': 'SCX.NPS.NPS23',
 'X_NPS24': 'SCX.NPS.NPS24',
 'X_NPS25': 'SCX.NPS.NPS25',
 'X_NPS26': 'SCX.NPS.NPS26',
 'X_NPS27': 'SCX.NPS.NPS27',
 'X_NPS28': 'SCX.NPS.NPS28',
 'X_NPS31': 'SCX.NPS.NPS31',
 'X_NPS32': 'SCX.NPS.NPS32',
 'X_NPS33': 'SCX.NPS.NPS33',
 'X_NPS34': 'SCX.NPS.NPS34',
 'X_NPS35': 'SCX.NPS.NPS35',
 'X_NPS36': 'SCX.NPS.NPS36',
 'X_NPS37': 'SCX.NPS.NPS37',
 'X_NPS38': 'SCX.NPS.NPS38',
 'X_NULPKO': 'SCX.NULL.PKO',
 'X_NULPKP': 'SCX.NULL.PKO.POS',
 'X_OAP1': 'SCX.OAP1',
 'X_OAP1F': 'SCX.OAP1.F',
 'X_OAP1PH': 'SCX.OAP1.PHI',
 'X_OAP1TH': 'SCX.OAP1.THETA',
 'X_OAP4': 'SCX.OAP4',
 'X_OAP4PH': 'SCX.OAP4.PHI',
 'X_OAP4TH': 'SCX.OAP4.THETA',
 'X_PG1PKO': 'SCX.PG1.PKO',
 'X_PI1WHL': 'SCX.PIAA1.WHL',
 'X_PI1X': 'SCX.PIAA1.X',
 'X_PI1Y': 'SCX.PIAA1.Y',
 'X_PI2F': 'SCX.PIAA2.F',
 'X_PI2WHL': 'SCX.PIAA2.WHL',
 'X_PI2X': 'SCX.PIAA2.X',
 'X_PI2Y': 'SCX.PIAA2.Y',
 'X_PIAA1': 'SCX.PIAA1',
 'X_PIAA2': 'SCX.PIAA2',
 'X_POLAR': 'SCX.POLAR',
 'X_POLARP': 'SCX.POLAR.POS',
 'X_PUPIL': 'SCX.PUP',
 'X_PUPWHL': 'SCX.PUP.WHL',
 'X_PUPX': 'SCX.PUP.X',
 'X_PUPY': 'SCX.PUP.Y',
 'X_PYWCAL': 'SCX.PYWFS.CAL',
 'X_PYWCLP': 'SCX.PYWFS.CLOOP',
 'X_PYWCOL': 'SCX.PYWFS.COL',
 'X_PYWDMO': 'SCX.PYWFS.DMOFF',
 'X_PYWFCS': 'SCX.PYWFS.FCS',
 'X_PYWFLT': 'SCX.PYWFS.FLT',
 'X_PYWFPK': 'SCX.PYWFS.FCS.PKO',
 'X_PYWFRQ': 'SCX.PYWFS.FREQ',
 'X_PYWFST': 'SCX.PYWFS.FSTOP',
 'X_PYWFSX': 'SCX.PYWFS.FSTOP.X',
 'X_PYWFSY': 'SCX.PYWFS.FSTOP.Y',
 'X_PYWGN': 'SCX.PYWFS.GAIN',
 'X_PYWLK': 'SCX.PYWFS.LEAK',
 'X_PYWLP': 'SCX.PYWFS.LOOP',
 'X_PYWPKO': 'SCX.PYWFS.PKO',
 'X_PYWPKP': 'SCX.PYWFS.PKO.POS',
 'X_PYWPLP': 'SCX.PYWFS.PLOOP',
 'X_PYWPPX': 'SCX.PYWFS.PUPX',
 'X_PYWPPY': 'SCX.PYWFS.PUPY',
 'X_PYWRAD': 'SCX.PYWFS.RAD',
 'X_RCHFIB': 'SCX.REACH.FIB',
 'X_RCHFIF': 'SCX.REACH.FIB.F',
 'X_RCHFIT': 'SCX.REACH.FIB.THETA',
 'X_RCHFIX': 'SCX.REACH.FIB.X',
 'X_RCHFIY': 'SCX.REACH.FIB.Y',
 'X_RCHOAP': 'SCX.REACH.OAP',
 'X_RCHOPH': 'SCX.REACH.OAP.PHI',
 'X_RCHOTH': 'SCX.REACH.OAP.THETA',
 'X_RCHPKO': 'SCX.REACH.PKO',
 'X_RCHPKP': 'SCX.REACH.PKO.POS',
 'X_RHEPKO': 'SCX.RHEA.PKO',
 'X_RHEPKP': 'SCX.RHEA.PKO.POS',
 'X_SPCFRQ': 'SCX.SPCT.FREQ',
 'X_SPCGN': 'SCX.SPCT.GAIN',
 'X_SPCLP': 'SCX.SPCT.LOOP',
 'X_SRCFFT': 'SCX.SRC.FLUX.FLT',
 'X_SRCFIB': 'SCX.SRC.FIB',
 'X_SRCFIP': 'SCX.SRC.FIB.Y',
 'X_SRCFIR': 'SCX.SRC.FLUX.IRND',
 'X_SRCFIX': 'SCX.SRC.FIB.X',
 'X_SRCFOP': 'SCX.SRC.FLUX.OPTND',
 'X_SRCSEL': 'SCX.SRC.SEL',
 'X_SRCSEP': 'SCX.SRC.SEL.POS',
 'X_STR': 'SCX.STEER',
 'X_STRPHI': 'SCX.STEER.PHI',
 'X_STRTHE': 'SCX.STEER.THETA',
 'X_VAMFST': 'SCX.VAMPIRES.FSTOP',
 'X_VAMFSX': 'SCX.VAMPIRES.FSTOP.X',
 'X_VAMFSY': 'SCX.VAMPIRES.FSTOP.Y',
 'X_ZAPFRQ': 'SCX.ZAP.FREQ',
 'X_ZAPGN': 'SCX.ZAP.GAIN',
 'X_ZAPLP': 'SCX.ZAP.LOOP',
 'X_ZAPMOT': 'SCX.ZAP.MOT',
 'X_ZAPNMO': 'SCX.ZAP.NMO',
 'ZD': 'FITS.SBR.ZD',
 'HA':'FITS.SBR.HA',
 'AZ':'TSCS.AZ',
 'EL':'TSCS.EL'}
G2CAM_TO_FITS = {v:k for k,v in FITS_2_G2CAM.items()}



def get_palomar(host='', user='', password=''):
    #TODO implement
    d = {'FITS.SBR.RA': None, 'FITS.SBR.DEC': None, 'FITS.SBR.EQUINOX': None,
         'FITS.SBR.HA': None, 'FITS.SBR.AIRMASS': None, 'FITS.SBR.UT': None,
         'TSCS.AZ': None, 'TSCS.EL': None, 'STATS.PARALLACTIC': None}
    return {'RA': d['FITS.SBR.RA'], 'DEC': d['FITS.SBR.DEC'], 'HA': d['FITS.SBR.HA'],
            'AIRMASS': d['FITS.SBR.AIRMASS'], 'AZ': d['TSCS.AZ'], 'EL': d['TSCS.EL'], 'TCS-UTC': d['FITS.SBR.UT'],
            'EQUINOX': d['FITS.SBR.EQUINOX'], 'PARALLACTIC': d['STATS.PARALLACTIC']}


def get_subaru(host='', user='', password=''):
    # setup (use the Gen2 host, user name and password you are advised by
    # observatory personnel)
    global SUBARU


    QUERY = {k:None for k in G2CAM_TO_FITS}
    MIN_SUBARU_QUERY_INTERVAL = 5

    try:
        from g2cam.status.client import StatusClient
    except ImportError:
        getLogger(__name__).error('Unable to import g2cam.status.client.StatusClient')
        return {G2CAM_TO_FITS[k]: None for k in QUERY}

    if SUBARU['client'] is None:# or SUBARU['client'].is_disconnected: #TODO is_disconnected isn't an attrib
        try:
            SUBARU['client'] = StatusClient(host=host, username=None if not user else user,
                                            password=None if not password else password)
            SUBARU['client'].connect()
        except Exception:
            getLogger(__name__).error('Unable to connect to Subaru TCS', exc_info=True)

    if time.time() - SUBARU['time'] > MIN_SUBARU_QUERY_INTERVAL:

        if SUBARU['cache'] is None:
            SUBARU['cache'] = QUERY

        try:
            SUBARU['client'].fetch(QUERY)
            SUBARU['time'] = time.time()
            SUBARU['cache'] = QUERY
        except Exception:
            getLogger(__name__).error('Unable to fetch from Subaru TCS', exc_info=True)

    return {G2CAM_TO_FITS[k]: v for k, v in SUBARU['cache'].items()}


def getPalomarSeeing(verbose=False):
    """
    get seeing log from http://nera.palomar.caltech.edu/P18_seeing/current.log

    set Verbose = True if you want debug messages
    read in last line of file and extract seeing value
    return this value
    """
    f = "current.log"
    address = "http://nera.palomar.caltech.edu/P18_seeing/%s" % f
    if verbose:
        getLogger(__name__).debug("Grabbing file from %s", address)
        p = subprocess.Popen("wget %s" % address, shell=True)
    else:
        p = subprocess.Popen("wget --quiet %s" % address, shell=True)
    p.communicate()
    stdin, stdout = os.popen2("tail -1 %s" % f)
    stdin.close()
    line = stdout.readlines()
    stdout.close()

    breakdown = line[0].split('\t')
    seeing = breakdown[4]
    if verbose:
        print(line)
        getLogger(__name__).debug("Seeing = {}. Deleting {}".format(seeing, f))
    os.remove(f)

    return seeing


class Telescope(object):
    def __init__(self, ip="198.202.125.194", port=5004, receivePort=1024, user='', password='NEVER_COMMIT_A_PASS'):
        self.address = (ip, port)
        self.ip = ip
        self.port = port
        self.receivePort = receivePort
        self.password = password
        self.user = user

    def get_header(self):
        return {'RA': '00:00:00.0000', 'DEC': '00:00:00.0000', 'HA': 0.0, 'AIRMASS': 1.0, 'AZ': .0, 'EL': 90,
                'UTCTCS': '01/01/2000 00:00:00.00', 'EQUINOX': 2000.0, 'PARALLACTIC': 0.0}


class NoScope(Telescope):
    def __init__(self, **kwargs):
        super(NoScope, self).__init__(**kwargs)
        self.observatory = 'None'
        self.lat = 0.0
        self.lon = 0.0
        self.latStr = '00.0:00.0:00.0'
        self.lonStr = '0.0:00.0:00.00'
        self.elevation = 0.0

    def get_telescope_position(self, targetName=None):
        if targetName is None or len(targetName) == 0:
            targetName = 'sky'
        return self.get_header()


class Subaru(Telescope):
    def __init__(self, **kwargs):
        super(Subaru, self).__init__(**kwargs)
        self.observatory = 'Subaru'
        self.lat = 19.0 + 49.0/60.0 + 43/3600.0
        self.lon = 155.0 + 28.0/60.0 + 50/3600.0
        self.latStr = '19.0:49.0:43.0'
        self.lonStr = '-155.0:28.0:50.00'
        self.elevation = 4139.0

    def get_header(self):
        return get_subaru(self.ip, self.user, self.password)

    def get_telescope_position(self, targetName='sky'):
        if targetName is None or len(targetName) == 0:
            targetName = 'sky'
        return self.get_header()


class Palomar(Telescope):
    def __init__(self, **kwargs):
        super(Palomar, self).__init__(**kwargs)

        #Palomar's position
        self.observatory = 'Palomar'

        self.lat = 33.0 + 21.0/60.0 + 21.6/3600.0
        self.lon = 116.0 + 51.0/60.0 + 46.8/3600.0
        self.latStr = '33.0:21.0:21.6'
        self.lonStr = '-116.0:51.0:46.80'
        self.elevation = 1706.0

    def getAllTelescopeInfo(self,targetName='sky'):
        telescopeDict = self.get_telescope_position(targetName)
        telescopeDict.update(self.get_telescope_status())
        telescopeDict.update(self.get_parallactic())
        telescopeDict.update(self.get_seeing())
        telescopeDict.update({'telescope':self.observatory, 'obslat':self.lat, 'obslon':self.lon, 'obsalt':self.elevation})
        return telescopeDict

    def sendTelescopeCommand(self, command):
        try:
            self.client_socket = socket(AF_INET, SOCK_STREAM) #Set Up the Socket
            self.client_socket.settimeout(0.2)
            self.client_socket.connect(self.address)
        except:
            getLogger(__name__).error("Connection to TCS at {} failed.".format(self.address))
            return
        response = None
        try:
            self.client_socket.send(command)
            response = self.client_socket.recv(self.receivePort)
        except:
            getLogger(__name__).error('Command "{}" to TCS failed.\n Recieved at {}.'.format(
                                            command, self.receivePort))
        self.client_socket.close()
        return response

    def get_telescope_status(self):
        response = self.sendTelescopeCommand('REQSTAT\r')
        if response is None:
            return {}

        utc, line2, line3, line4, cass_angle = response.split('\n')
        telescope_id, focus, tubelength = line2.split(', ')
        focus_title, focus_val = focus.split(' = ')
        cassAngle_key, cassAngle_val = cass_angle.split(' = ')
        cassAngle_val = cassAngle_val.strip().split('\x00')[0]
        telescopeID_key, telescopeID_val = telescope_id.split(' = ')
        utc = utc.split(' = ')[1]
        tubeLength_key, tubeLength_val = tubelength.split(' = ')
        return {'Status UTC':utc, cassAngle_key: cassAngle_val,
                telescopeID_key: telescopeID_val, tubeLength_key: tubeLength_val,
                focus_title:focus_val}

    def get_parallactic(self):
        response = self.sendTelescopeCommand('?PARALLACTIC\r')
        if response is None:
            return {}
        parallactic = response.split('\n')
        title, par_value = parallactic[0].split('= ')
        return {str(title):float(par_value)}

    def get_telescope_position(self,targetName='sky'):
        if targetName is None or len(targetName)==0: targetName='sky'
        response = self.sendTelescopeCommand('REQPOS\r')
        if response is None:
            return {}

        #split response into status fields
        line1,line2,airmass = response.split('\n')

        utc, lst = line1.split(', ')
        utc_title, junk, utc_day, utc_time = utc.split(' ')
        lst_title, junk, lst_time = lst.split(' ')

        ra, dec, ha = line2.split(', ')
        ra_title, junk, ra_val = ra.split(' ')
        dec_title, junk, dec_val = dec.split(' ')
        ha_title, junk, ha_val = ha.split(' ')

        airmass_title, airmass_val = airmass.split('=  ')


        #calculate alt and az of current target
        target = ephem.readdb(str(targetName)+',f|L,'+str(ra_val)+','+str(dec_val)+',2000')

        lt = time.time()
        dt = datetime.datetime.utcfromtimestamp(lt)

        palomar = ephem.Observer()
        palomar.long, palomar.lat = self.lonStr, self.latStr
        palomar.date = ephem.Date(dt)
        palomar.elevation = self.elevation
        target.compute(palomar)
        alt, az = target.alt*(180./math.pi), target.az*(180/math.pi)

        return {str(utc_title):utc_time, str(lst_title):lst_time, str(ra_title):ra_val,
                str(dec_title):dec_val, str(ha_title):ha_val,str(airmass_title):float(airmass_val[:-1]),
                'alt':alt, 'az':az}

    def get_seeing(self):
        return {'seeing':getPalomarSeeing()}
