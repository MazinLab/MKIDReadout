"""
Author: Alex Walter, Rupert Dodkins
Date: Jul 7, 2016

This class communicates with the Laser box arduino
We can turn the lasers on and off

We also have the Darkness pupil image flipper controlled with this arduino
"""

from socket import *
from PyQt4 import QtCore
#import time



class LaserControl():

    def __init__(self, ipaddress='10.0.0.55', port=8888,receivePort = 4096,verbose=False):
        """
        Class for controlling laser
        """
        self.address = (ipaddress, port)
        self.receivePort = receivePort
        
        self.numLasers = 5+1    # the first number controls the flipper
        
        self.lastToggle='0'*self.numLasers
    
    def toggleLaser(self, toggle,timeLimit=-1):
        """
        Change the laser on or off
        
        INPUTS:
            toggle - String with each character a 1 or 0 describing the on/off state of the laser
                     "10101" means turn lasers 1,3,5 on and 2,4 off. 
            timeLimit - Turn the laser off after this many seconds. if <= 0 then leave on
        """
        print toggle, timeLimit
        #return
        
        assert len(toggle)==self.numLasers
        self.lastToggle = toggle
        client_socket = socket(AF_INET, SOCK_DGRAM) #Set Up the Socket
        client_socket.settimeout(1) #only wait 1 second for a response
        client_socket.sendto(toggle, self.address) #send command to arduino
        if timeLimit>0:
            QtCore.QTimer.singleShot(timeLimit*1000, self.laserOff)
        try:
            rec_data, addr = client_socket.recvfrom(self.receivePort) #Read response from arduino
            if verbose:
                print rec_data #Print the response from Arduino
        except:
            print "No response from Arduino"
        
    def laserOff(self):
        client_socket = socket(AF_INET, SOCK_DGRAM) #Set Up the Socket
        client_socket.settimeout(1) #only wait 1 second for a response
        client_socket.sendto(self.lastToggle[0]+"0"*(self.numLasers-1), address)
        try:
            rec_data, addr = client_socket.recvfrom(self.receivePort) #Read response from arduino
            if verbose:
                print rec_data #Print the response from Arduino
        except:
            print "No response from Arduino"
    
    def laserFlash(self, onTime, offTime, timeLimit):
        raise NotImplementedError


if __name__ == "__main__":
    laser = laserControl()
    toggle = "11100"
    laser.toggleLaser(toggle)
