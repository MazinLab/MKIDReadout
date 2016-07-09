"""
Author: Alex Walter
Date: Jul 8, 2016

This class grabs info from the Palomar telescope
"""
from socket import *
import time
import datetime
import ephem
#from lib.getSeeing import getPalomarSeeing  # From old SDR code
from getSeeing import getPalomarSeeing  # From old SDR code

class Telescope():

    def __init__(self, ipaddress="10.200.2.11", port = 49200, receivePort=1024):
        self.address = (ipaddress, port)
        self.receivePort = receivePort
        self.client_socket = socket(AF_INET, SOCK_STREAM) #Set Up the Socket
        
        #Palomar's position
        observatory = "Palomar"
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
        return telescopeDict

    def sendTelescopeCommand(self, command):
        try:
            self.client_socket.connect(self.address)
        except:
            print "Connection to TCS failed"
            return
        response = None
        try:
            self.client_socket.send(command)
            response = self.client_socket.recv(self.receivePort)
        except:
            print "Command to TCS failed: "+str(command)
        self.client_socket.close()
        return response

    def get_telescope_status(self):
        response = self.sendTelescopeCommand('REQSTAT\r')
        if response is None:
            return {}
            
        utc, line2, line3, line4, cass_angle = response.split('\n')
        telescope_id, focus, tubelength = line2.split(', ')
        focus_title, focus_val = focus.split('= ')
        return {'status_utc':utc, 'cass_angle':cass_angle,
                'id':telescope_id, 'tubelength':tubelength,
                'focus_title':focus_title, 'focus':focus_val}

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
    


