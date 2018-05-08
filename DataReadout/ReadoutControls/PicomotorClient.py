'''Written by Isabel Lipartito 04.01.2017

Set of functions for talking to New Focus picomotor controller through
a server running on DARKNESS PC.
'''

import socket
import sys

def moveMotor(motor, nsteps ):
    input=file("params.txt",'r')
    lines=input.readlines()
    input.close()
    TCP_IP=str(lines[0]).rstrip()
    TCP_PORT = int(lines[1])
    BUFFER_SIZE = 10025
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        s.send(str(motor)+' '+str(nsteps))
        flag = s.recv(BUFFER_SIZE)
        s.close()
        print flag.rstrip()
    except socket.error, msg:
        print 'Failed to create socket. Error code: ' + str(msg[0]) + ' , Error message : ' + msg[1]
        sys.exit()
    return 

def getPosition():
    input=file("params.txt",'r')
    lines=input.readlines()
    input.close()
    TCP_IP=str(lines[0]).rstrip()
    TCP_PORT = int(lines[1])
    BUFFER_SIZE = 10025
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        s.send(str(1))
        positions = s.recv(BUFFER_SIZE)
        s.close()
        print positions.rstrip()
    except socket.error, msg:
        print 'Failed to create socket. Error code: ' + str(msg[0]) + ' , Error message : ' + msg[1]
        sys.exit()
    return


if __name__ == "__main__":
  getPosition()
# main()
