"""
Author: Alex Walter, Rupert Dodkins
Date: Jul 7, 2016

This class communicates with the Laser box arduino
We can turn the lasers on and off

We also have the Darkness pupil image flipper controlled with this arduino
"""

from __future__ import print_function

from mkidcore.corelog import getLogger, setup_logging
from mkidcore.safesocket import *


class LaserControl(object):
    def __init__(self, ipaddress='10.0.0.55', port=8888, receivePort=4096):
        """
        Class for controlling laser
        """
        self.address = (ipaddress, port)
        self.receivePort = receivePort
        self.numLasers = 5 + 1  # the first number controls the flipper
        self.lastToggle = '0' * self.numLasers

    @property
    def status(self):
        return 'Status Not Implemented'

    def toggleLaser(self, toggle):
        """
        Change the laser on or off
        
        INPUTS:
            toggle - String with each character a 1 or 0 describing the on/off state of the laser
                     "10101" means turn lasers 1,3,5 on and 2,4 off. 
        """
        if len(toggle) != self.numLasers:
            raise ValueError('Toggle must be a boolean string of length n lasers')

        client_socket = socket(AF_INET, SOCK_DGRAM)  # Set Up the Socket
        client_socket.settimeout(1)  # only wait 1 second for a response
        client_socket.sendto(toggle, self.address)  # send command to arduino
        try:
            rec_data, addr = client_socket.recvfrom(self.receivePort)  # Read response from arduino
            getLogger(__name__).debug(rec_data)  # Print the response from Arduino
            self.lastToggle = toggle
            return True
        except Exception:
            getLogger(__name__).warning("No response from Arduino")
            return False

    def laserOff(self):
        client_socket = socket(AF_INET, SOCK_DGRAM)  # Set Up the Socket
        client_socket.settimeout(1)  # only wait 1 second for a response
        client_socket.sendto(self.lastToggle[0] + "0" * (self.numLasers - 1), self.address)
        try:
            rec_data, addr = client_socket.recvfrom(self.receivePort)  # Read response from arduino
            getLogger(__name__).debug(rec_data)  # Print the response from Arduino
            return True
        except Exception:
            getLogger(__name__).warning("No response from Arduino")
            return False

    def laserFlash(self, onTime, offTime, timeLimit):
        raise NotImplementedError


if __name__ == "__main__":
    setup_logging()
    laser = LaserControl()
    toggle = "11100"
    laser.toggleLaser(toggle)
