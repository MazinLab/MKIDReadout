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

SUBARU = {'client':None, 'time':0, 'cache':None}

def get_palomar(host='', user='', password=''):
    #TODO implement
    d = {'FITS.SBR.RA': None, 'FITS.SBR.DEC': None, 'FITS.SBR.EQUINOX': None,
         'FITS.SBR.HA': None, 'FITS.SBR.AIRMASS': None, 'FITS.SBR.UT': None,
         'TSCS.AZ': None, 'TSCS.EL': None}
    return {'RA': d['FITS.SBR.RA'], 'DEC': d['FITS.SBR.DEC'], 'HA': d['FITS.SBR.HA'],
            'AIRMASS': d['FITS.SBR.AIRMASS'], 'AZ': d['TSCS.AZ'], 'EL': d['TSCS.EL'], 'TCS-UTC': d['FITS.SBR.UT'],
            'EQUINOX': d['FITS.SBR.EQUINOX']}


def get_subaru(host='', user='', password=''):
    # setup (use the Gen2 host, user name and password you are advised by
    # observatory personnel)
    global SUBARU

    QUERY = {'FITS.SBR.RA': None, 'FITS.SBR.DEC': None, 'FITS.SBR.EQUINOX': None,
             'FITS.SBR.HA': None, 'FITS.SBR.AIRMASS': None, 'FITS.SBR.UT': None,
             'TSCS.AZ': None, 'TSCS.EL': None}
    MIN_SUBARU_QUERY_INTERVAL = 5

    try:
        from g2cam.status.client import StatusClient
    except ImportError:
        getLogger(__name__).error('Unable to import g2cam.status.client.StatusClient')
        d = QUERY
        return {'RA': d['FITS.SBR.RA'], 'DEC': d['FITS.SBR.DEC'], 'HA': d['FITS.SBR.HA'],
                'AIRMASS': d['FITS.SBR.AIRMASS'], 'AZ': d['TSCS.AZ'], 'EL': d['TSCS.EL'], 'UTCTCS': d['FITS.SBR.UT'],
                'EQUINOX': d['FITS.SBR.EQUINOX']}

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

    d = SUBARU['cache']

    return {'RA': d['FITS.SBR.RA'], 'DEC': d['FITS.SBR.DEC'], 'HA': d['FITS.SBR.HA'],
            'AIRMASS': d['FITS.SBR.AIRMASS'], 'AZ': d['TSCS.AZ'], 'EL': d['TSCS.EL'], 'TCS-UTC': d['FITS.SBR.UT'],
            'EQUINOX': d['FITS.SBR.EQUINOX']}


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
        return {'RA': '00:00:00.0000', 'DEC': '00:00:00.0000', 'HA': 0.0, 'AIRMASS': 1.0, 'AZ': 45, 'EL': 90,
                'TCS-UTC': '01/01/2000 00:00:00.00', 'EQUINOX': 2000.0}


class Subaru(Telescope):
    def __init__(self, **kwargs):
        super(Subaru, self).__init__(**kwargs)
        self.observatory = 'Subaru Telescope'
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
        self.observatory = 'Palomar 200" Hale Telescope'

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
    


