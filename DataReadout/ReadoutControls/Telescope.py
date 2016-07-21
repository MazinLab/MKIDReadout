"""
Author: Alex Walter
Date: Jul 8, 2016

This class grabs info from the Palomar telescope
"""
from socket import *
import time, math
import datetime
from PyQt4 import QtGui
from PyQt4.QtGui import *
from PyQt4 import QtCore
import ephem
#from lib.getSeeing import getPalomarSeeing  # From old SDR code
from lib.getSeeing import getPalomarSeeing  # From old SDR code

class TelescopeWindow(QMainWindow):
    
    def __init__(self, telescope, parent=None):
        """
        INPUTES:
            telescope - Telescope object
        """
        super(QMainWindow, self).__init__(parent)
        self.setWindowTitle("Palomar Telescope")
        self._want_to_close = False
        self.telescope = telescope
        self.create_main_frame()
        updater = QtCore.QTimer(self)
        updater.setInterval(1003)
        updater.timeout.connect(self.updateTelescopeInfo)
        updater.start()
    
    def updateTelescopeInfo(self, target='sky'):
        if not self.isVisible():
            return
        tel_dict = self.telescope.get_telescope_position()
        for key in tel_dict.keys():
            try:
                self.label_dict[key].setText(str(tel_dict[key]))
            except:
                layout = self.main_frame.layout()
                label = QLabel(key)
                label_val = QLabel(str(tel_dict[key]))
                hbox = QHBoxLayout()
                hbox.addWidget(label)
                hbox.addWidget(label_val)
                layout.addLayout(hbox)
                self.main_frame.setLayout(layout)
                self.label_dict[key] = label_val
    
    def create_main_frame(self):
        self.main_frame = QWidget()
        vbox = QVBoxLayout()
        
        def add2layout(vbox, *args):
            hbox = QHBoxLayout()
            for arg in args:
                hbox.addWidget(arg)
            vbox.addLayout(hbox)
        
        label_telescopeStatus = QLabel('Telescope Status')
        font = label_telescopeStatus.font()
        font.setPointSize(24)
        label_telescopeStatus.setFont(font)
        vbox.addWidget(label_telescopeStatus)
        
        tel_dict = self.telescope.get_telescope_position()
        self.label_dict={}
        for key in tel_dict.keys():
            label = QLabel(key)
            label.setMaximumWidth(150)
            label_val = QLabel(str(tel_dict[key]))
            label_val.setMaximumWidth(150)
            add2layout(vbox,label,label_val)
            self.label_dict[key] = label_val
        
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)
    
    def closeEvent(self, event):
        if self._want_to_close:
            self.close()
        else:
            self.hide()
        

class Telescope():

    def __init__(self, ipaddress="198.202.125.194", port = 5004, receivePort=1024):
        self.address = (ipaddress, port)
        self.receivePort = receivePort
        
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
            self.client_socket.connect(self.address)
        except:
            print "Connection to TCS failed"
            print "Telescope IP: ",self.address
            return
        response = None
        try:
            self.client_socket.send(command)
            response = self.client_socket.recv(self.receivePort)
        except:
            print "Command to TCS failed: "+str(command)
            print "Received at: ",self.receivePort
        self.client_socket.close()
        return response

    def get_telescope_status(self):
        response = self.sendTelescopeCommand('REQSTAT\r')
        if response is None:
            return {}
            
        utc, line2, line3, line4, cass_angle = response.split('\n')
        telescope_id, focus, tubelength = line2.split(', ')
        focus_title, focus_val = focus.split(' = ')
        cassAngle_key, cassAngle_val = cass_angle.split(' =  ')
        cassAngle_val = cassAngle_val.split('\x00')[0]
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
    


