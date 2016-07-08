"""
Author: Alex Walter, Rupert Dodkins
Date: Jul 7, 2016

This class communicates with the Laser box arduino
We can turn the lasers on and off
"""

from socket import *
from PyQt4 import QtCore
#import time



class laserControl():

    def __init__(self, ipaddress='10.10.10.12', port=8888,receivePort = 4096,verbose=False):
        """
        Class for controlling laser
        """
        self.address = (ipaddress, port)
        self.receivePort = receivePort
        self.client_socket = socket(AF_INET, SOCK_DGRAM) #Set Up the Socket
        #client_socket.settimeout(1) #only wait 1 second for a response
        self.numLasers = 5
    
    def toggleLaser(self, toggle,timeLimit=-1):
        """
        Change the laser on or off
        
        INPUTS:
            toggle - String with each charachter a 1 or 0 describing the on/off state of the laser
                     "10101" means turn lasers 1,3,5 on and 2,4 off. 
            timeLimit - Turn the laser off after this many seconds. if <= 0 then leave on
        """
        assert len(toggle)==self.numLasers
        self.client_socket.sendto(toggle, address) #send command to arduino
        if timeLimit>0:
            QtCore.QTimer.singleShot(timeLimit*1000, self.laserOff)
        try:
            rec_data, addr = self.client_socket.recvfrom(self.receivePort) #Read response from arduino
            if verbose:
                print rec_data #Print the response from Arduino
        except:
            pass
        
     def laserOff(self)
        self.client_socket.sendto("00000", address)
        try:
            rec_data, addr = self.client_socket.recvfrom(self.receivePort) #Read response from arduino
            if verbose:
                print rec_data #Print the response from Arduino
        except:
            pass
    
    def laserFlash(self, onTime, offTime, timeLimit):
        raise NotImplementedError


if __name__ == "__main__":
    laser = laserControl()
    toggle = "11100"
    laser.toggleLaser(toggle)
