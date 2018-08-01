'''Written by Isabel Lipartito 04.01.2017

Set of functions for talking to New Focus picomotor controller through
a server running on DARKNESS PC.
'''

import socket
import sys
import time

def moveMotor(motor, nsteps ):
    input=file("params.txt",'r')
    lines=input.readlines()
    input.close()
    TCP_IP=str(lines[0]).rstrip()
    TCP_PORT = int(lines[1])
    BUFFER_SIZE = 10025
    moveSuccess = False
    try:
        for i in range(10):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((TCP_IP, TCP_PORT))
            messageToSend = str(motor) + ' ' + str(nsteps)
            messageLen = len(messageToSend)
            totalNumSent = 0
            while totalNumSent<messageLen:
                numSent = s.send(messageToSend[totalNumSent:])
                if(numSent==0):
                    raise Exception('broken socket connection')
                totalNumSent += numSent
            flag = s.recv(BUFFER_SIZE)
            flag = flag.rstrip()
            print flag
            s.close()
            if(int(flag)==0):
                moveSuccess = True
                break
            time.sleep(0.5)
        s.close()
    except socket.error, msg:
        print 'Failed to create socket. Error code: ' + str(msg[0]) + ' , Error message : ' + msg[1]
        sys.exit()

    if moveSuccess==False:
        raise Exception('Picomotor move failed after 10 tries')
    return 

def getPosition():
    input=file("params.txt",'r')
    lines=input.readlines()
    input.close()
    TCP_IP=str(lines[0]).rstrip()
    TCP_PORT = int(lines[1])
    BUFFER_SIZE = 10025
    try:
        for i in range(10):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((TCP_IP, TCP_PORT))
            messageToSend = str(1)
            messageLen = len(messageToSend)
            totalNumSent = 0
            while totalNumSent<messageLen:
                numSent = s.send(messageToSend[totalNumSent:])
                if(numSent==0):
                    raise Exception('broken socket connection')
                totalNumSent += numSent
            positions = s.recv(BUFFER_SIZE)
            positions = positions.rstrip()
            print positions
            s.close()
            if(len(positions)>1):
                moveSuccess = True
                break
            time.sleep(0.5)
    except socket.error, msg:
        print 'Failed to create socket. Error code: ' + str(msg[0]) + ' , Error message : ' + msg[1]
        sys.exit()
    if moveSuccess==False:
        raise Exception('Picomotor move failed after 10 tries')
    return


if __name__ == "__main__":
  getPosition()
# main()
