"""
Author: Alex Walter
Date: Aug 29, 2018

This code is for controlling/dithering the Newport Conex-AG-M100D stage in the MEC fore-optics

"""
from __future__ import print_function

import time
from threading import RLock, Thread

import numpy as np
import requests
import serial
from flask_restful import Api, Resource, reqparse

import mkidcore
from mkidcore.corelog import create_log, getLogger

TIMEOUT = 2.0               # timeout for request post
CONEX_COM_PORT = "COM9"
CONEX_SERVER_PORT = 50001

class Conex():
    """
    This class actually talks with the conex mount over the pyserial connection
    """
    def __init__(self, port="COM9", baudrate=912600, bytesize=serial.EIGHTBITS,
                 stopbits=serial.STOPBITS_ONE, timeout=1., xonxoff=True, controllerNum=1):
      
        self._rlock = RLock()
        # Generally 1. This is for when you daisy chain multiple conex stages together
        self.ctrlN = controllerNum

        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.stopbits = stopbits
        self.timeout = timeout
        self.xonxoff = xonxoff
        self.u_lowerLimit = -np.inf
        self.v_lowerLimit = -np.inf
        self.u_upperLimit = np.inf
        self.v_upperLimit = np.inf

        self._started = 0
        try:
            self._device = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=self.bytesize,stopbits=self.stopbits, timeout=self.timeout, xonxoff=self.xonxoff)
            #self._started = 1
            time.sleep(0.1)  # wait for port to open

            q = [self.query(q) for q in ('SLU?','SLV?','SRU?','SRV?')]
            f = lambda x: float(x[4:-2])
            self.u_lowerLimit = f(q[0])
            self.v_lowerLimit = f(q[1])
            self.u_upperLimit = f(q[2])
            self.v_upperLimit = f(q[3])
        except serial.SerialException:
            getLogger('conex').debug("Could not connect to Conex Mount", exc_info=True)
            raise

        getLogger('conex').debug("Connected to Conex Mount")

    def close(self):
        try:
            getLogger('conex').debug("Disconnecting Serial port")
            self._device.close()
        except: pass

    @property
    def status(self):
        """
        Get the status
        syntax: "xxTSabcdef\r\n"
                xx - the controller number
                TS - the command
                abcd - error code
                ef - status
        
        """
        status = self.query('TS')
        getLogger('conex').debug("Status: " + status[7:-2])
        getLogger('conex').debug("Err State: " + status[3:-4])
        return status.strip()

    def write(self, command):
        """
        Send an ascii command to the conex controller
        Format is: xxCCnn\r\n
                   xx is the controller number (usually 1 unless you daisy chain multiple conex stages)
                   CC is the command
                   nn is the option input for the command
                   \r\n is the terminator character letting Conex know you've finished sending the command

        This is a blocking function. It will try to write to the conex until it 
        succeeds or the write_timeout is reached

        INPUT:
            command - The two letter ascii command (CC above)
        """
        with self._rlock:
            self._device.write('{}{}\r\n'.format(self.ctrlN, command).encode())

    def read(self, bufferSize=1):
        """
        read from the port buffer

        This is a blocking function. It will try to read until
        it sees the end line characters \r\n or it times out.

        INPUT:
            bufferSize - size of return string in bytes
                         If you don't know the exact size, leave as 1. 
                         It'll keep adding onto the buffer until it reaches the \r\n characters
        """
        with self._rlock:
            bufferSize = abs(bufferSize)
            if not bufferSize:
                return ''
            ret = self._device.read(bufferSize).decode()    #blocks until at least buffersize bytes in receive buffer (or timeout)
            if not ret:
                return ret
            while ret[-2:]  != '\r\n':  # reads until end of line
                #getLogger('conex').debug('Got "{}"'.format(new))
                new = self._device.read(self._device.in_waiting).decode()
                ret += new
            return ret

    def query(self, command, bufferSize=1):
        """
        Send an ascii command to the conex controller
        Read the output

        INPUTS:
            command - see write()
            bufferSize - see read()
        OUTPUTS:
            the return buffer
        """
        with self._rlock:
            self._device.reset_output_buffer()  # abandon any command it's currently sending
            self._device.reset_input_buffer()  # clear the input buffer
            self.write(command)
            self._device.flush()  # wait for command to finish writing
            return self.read(bufferSize)

    @property
    def limits(self):
        """
        The hardware limit for u, v in degrees

        OUTPUT:
            dictionary with keys (umin, umax, vmin, vmax)
        """
        return dict(umin=self.u_lowerLimit, vmin=self.v_lowerLimit,
                    umax=self.u_upperLimit, vmax=self.v_upperLimit)
    
    def ready(self):
        """
        Checks the status of the conex

        returns True if it's ready for another command (like move)
        returns false if not ready
        If there was an error it raises an IOError
        """
        with self._rlock:
            self._device.flush()  # wait for last command to finish sending
            self._device.reset_input_buffer()  # clear any return values in the buffer
            self.write('TS')
            # self.conex.flush()
            status = self._device.read(11)  # blocking read until 11 bytes are returned

            err = status[3:7]  # hex error code
            state = status[7:9]  # decimal state
            if err == '0020':
                raise IOError('Motion Time out')
            elif int(err, 16) > 0:
                raise IOError('Unknown Err - ' + err)

            return int(state) in (32, 33, 34, 35, 36)

    def move(self, pos, blocking=False, timeout=5.):
        """
        Move mirror to new position

        INPUTS:
            pos - [U, V] tuple position in degrees (Conex truncates this at 3 decimal places)
            blocking - If True then don't return until move is complete
            timeout - error out if it takes too long to complete move
                      ignored if not blocking
                      For some reason it needs a long time even though the move itself is fast
        """
        with self._rlock:
            if not self.inBounds(pos):
                raise IOError('Target position outside of limits. Aborted move.')
            self.write('PAU' + str(pos[0]))
            self._device.flush()  # wait until write command finishes sending
            self.write('PAV' + str(pos[1]))  # Conex is capable of moving both axes at once!
            if blocking: self._device.flush()
        if blocking:    #don't lock the resource while waiting for move to finish. This allows the stop() command
            t = time.time()
            while not self.ready():
                if time.time() - t > timeout:
                    status = self.query('TS')
                    raise IOError("Move Abs timed out. Status: " + status[:-2])
                time.sleep(0.0001)

    def stop(self):
        """
        Aborts connex mount move
        """
        with self._rlock:
            self.write('ST')

    def inBounds(self, target):
        """
        Check that the target is within the movement limits

        INPUT:
            target - [U,V] tuple position in degrees
        """
        return (self.u_lowerLimit <= target[0] <= self.u_upperLimit and
                self.v_lowerLimit <= target[1] <= self.v_upperLimit)

    def home(self,blocking=False):
        self.move((0, 0),blocking=blocking)

    def position(self):
        """ Returns the pos [U,V] in degrees """
        return float(self.query('TPU')[4:-2]), float(self.query('TPV')[4:-2])


class ConexManager():
    """
    This class manages the Conex() object

    It implements a thread safe dither routine among other things

    The self.state attribute is a tuple
        self.state[0] was the last state
        self.state[1] is the current state

    Posible states:
        'Unknown', 'Offline', 'Idle', 'Stopped', 'Moving ...', 'Dither ...', 'Error ...'
    """
    def __init__(self, port):
        self.conex = Conex(port=port)
        self._completed_dithers = []    # list of completed dithers
        self._movement_thread = None    #thread for moving/dithering
        self._halt_dither = True
        self._rlock = RLock()
        self._startedMove = 0           # number of times start_move was called (not dither). Reset in queryMove and start_dither
        self._completedMoves = 0        # number of moves completed (not dither)

        self.state=('Unknown','Unknown')
        try:
            if self.conex.ready(): self._updateState('Idle')
        except: pass
        self.cur_status = self.status()
        



    def queryMove(self):
        """
        Checks to see if move completed

        It should be thread safe. Even if you hit the move button several times really fast

        OUTPUTS:
            dictionary {'completed':True/False, 'status':self.cur_status}
        """
        if self._completedMoves>0:  # don't lock if no moves completed. Reading is thread safe
            with self._rlock:       # if at least one move completed then lock
                if self._completedMoves>0:      # need to check again for thread safety (maybe started two moves but only 1 completed)
                    self._completedMoves-=1
                    self._startedMove-=1
                    self._startedMove = max(0, self._startedMove)
                    return {'completed':True, 'status':self.cur_status}
        return {'completed':False, 'status':self.cur_status}

    def queryDither(self):
        """
        returns the dictionary containing information about an ongoing or completed dither

        keys:
            status - The current status of the conex manager
            estTime - ?? Not implemented right now
            dither - A dictionary containing the oldest completed dither that hasn't been popped yet
                     see dither() output
                     If no completed dithers then None
            completed - True or False
        """
        dith=None
        estTime=0
        completed=False
        if len(self._completed_dithers)>0:  # Reading is thread safe
            with self._rlock:   # only lock if at least one dither completed
                try:
                    dith = self._completed_dithers.pop(0)
                    completed=True
                except IndexError: pass
        if dith is None:    # check if a dither was popped
            estTime = time.time()+1 #estimated unix time of when dither will complete
        return {'status':self.cur_status, 'estTime': estTime, 'dither':dith, 'completed':completed}

    def _updateState(self, newState):
        with self._rlock:
            self.state = (self.state[1], newState)

    def status(self):
        pos = (np.NaN, np.NaN)
        status=''
        try:
            status = self.conex.status
            pos = self.conex.position()
            #getLogger('ConexManager').debug("Conex: {} @ pos {}".format(status,pos))
        except (IOError, serial.SerialException):
            getLogger('ConexManager').error('Unable to get conex status', exc_info=True)
            self._halt_dither = True
            self._updateState('Offline')
        return {'state':self.state, 'pos':pos, 'conexstatus':status, 'limits':self.conex.limits}


    def stop(self, wait=False):
        """
        stops the current movement or dither

        if wait is False then it forcibly writes to the conex to tell it to stop motion
        
        after that it waits for the movement thread to finish
        """
        getLogger('ConexManager').debug('stopping conex')
        
        if self._movement_thread is not None and self._movement_thread.is_alive():
            with self._rlock:
                self._halt_dither = True
                if not wait:
                    self.conex.stop()  # puts conex in ready state so that _movement thread will finish
                self._updateState('Stopped')
                self.cur_status=self.status()
            self._movement_thread.join()    # not in rlock
            with self._rlock:
                self.cur_status=self.status()   # could change in other thread
        else:
            with self._rlock:
                self.cur_status = self.status()
        return self.cur_status

    def start_dither(self,dither_dict):
        """
        Starts dither in a new thread
        """
        getLogger('ConexManager').debug('starting dither')
        self.stop(wait=False)   # stop whatever we were doing before (including a previous dither)
        with self._rlock:
            self.cur_status =self.status()
            if self.cur_status['state'] == 'Offline': return False
            self._halt_dither = False
            self._startedMove = 0
        self._preDitherPos = self.cur_status['pos']
        self._movement_thread = Thread(target=self.dither, args=(dither_dict,), name="Dithering thread")
        self._movement_thread.daemon = True
        self._movement_thread.start()
        return True

    def dither(self,dither_dict):
        """
        INPUTS:
            dither_dict - dictionary with keys:
                        startx: (float) start x loc in conex degrees
                        endx: (float) end x loc
                        starty: (float) start y loc
                        endy: (float) end y loc
                        n: (int) number of steps in square grid
                        t: (float) dwell time in seconds
                        subStep: (float) degrees to offset for subgrid pattern
                        subT: (float) dwell time for subgrid

                        subStep and subT are optional

        appends a dictionary to the self._completed_dithers attribute
            keys - same as dither_dict but additionally
                   it has keys (xlocs, ylocs, startTimes, endTimes)

        """
        x_list = np.linspace(dither_dict['startx'], dither_dict['endx'], dither_dict['n'])
        y_list = np.linspace(dither_dict['starty'], dither_dict['endy'], dither_dict['n'])
        
        subDither = 'subStep' in dither_dict.keys() and dither_dict['subStep']>0 and \
                    'subT' in dither_dict.keys() and dither_dict['subT']>0

        x_locs = []
        y_locs = []
        startTimes = []
        endTimes = []
        for x in x_list:
            for y in y_list:
                startTime, endTime = self._dither_move(x,y,dither_dict['t'])
                if startTime is not None:
                    x_locs.append(self.cur_status['pos'][0])
                    y_locs.append(self.cur_status['pos'][1])
                    startTimes.append(startTime)
                    endTimes.append(endTime)
                if self._halt_dither: break

                #do sub dither if neccessary
                if subDither:
                    x_sub = [-dither_dict['subStep'], 0, dither_dict['subStep'], 0]
                    y_sub = [0, dither_dict['subStep'], 0, -dither_dict['subStep']]
                    for i in range(len(x_sub)):
                        if self.conex.inBounds((x+x_sub[i], y+y_sub[i])):
                            startTime, endTime = self._dither_move(x+x_sub[i],y+y_sub[i],dither_dict['subT'])
                            if startTime is not None:
                                x_locs.append(self.cur_status['pos'][0])
                                y_locs.append(self.cur_status['pos'][1])
                                startTimes.append(startTime)
                                endTimes.append(endTime)
                        if self._halt_dither: break
                if self._halt_dither: break
            if self._halt_dither: break

        #Dither has completed (or was stopped prematurely)
        if not self._halt_dither:       #no errors and not stopped
            self.move(*self._preDitherPos)
            with self._rlock:
                if not self._halt_dither:       # still no errors nor stopped
                    self._updateState("Idle")
                self.cur_status=self.status()

        dith = dither_dict.copy()
        dith['xlocs'] = x_locs  #could be empty if errored out or stopped too soon
        dith['ylocs'] = y_locs
        dith['startTimes'] = startTimes
        dith['endTimes'] = endTimes
        with self._rlock:
            self._completed_dithers.append(dith)
            
    def _dither_move(self,x,y,t):
        """
            Helper function for dither()

            The state after this function call will be one of:
                "error: ..." - If there there was an error during the move
                "processing" - If everything worked
        """
        polltime=0.1    #wait for dwell time but have to check if stop was pressed periodically
        self.move(x,y)
        if self._halt_dither: return None, None    # Stopped or error during move
        self._updateState("Dither dwell for {:.1f} seconds".format(t))
        #dwell at position
        startTime=time.time()
        dwell_until = startTime+t
        endTime=time.time()
        with self._rlock:
            self.cur_status = self.status()
        while self._halt_dither == False and endTime<dwell_until:
            sleep = min(polltime, dwell_until-endTime)
            time.sleep(max(sleep,0))
            endTime=time.time()
        return startTime, endTime

    def start_move(self, x, y):
        """
        Starts move in new thread
        """
        self.stop(wait=False)    # If the user wants to move, then forcibly stop whatever we were doing before (indcluding dithers)
        with self._rlock:
            self.cur_status =self.status()
            if self.cur_status['state'] == 'Offline': return False
            self._startedMove+=1
        #getLogger('ConexManager').error("Starting move to {:.2f}, {:.2f}".format(x,y))
        self._movement_thread = Thread(target=self.move, args=(x, y,),
                                       name='Move to ({}, {})'.format(x,y))
        self._movement_thread.daemon = True
        self._movement_thread.start()
        
        return True

    def move(self, x, y):
        """
        Tells conex to move and collects errors
        """
        self._updateState('Moving to {:.2f}, {:.2f}'.format(x, y))
        try:
            self.conex.move((x, y),blocking=True)   #block until conex is done moving (or stopped)
            if self._startedMove>0: self._updateState('Idle')
            getLogger('ConexManager').debug('moved to ({}, {})'.format(x,y))
        except (IOError, serial.SerialException) as e:              # on timeout it raise IOError
            self._updateState('Error: move to {:.2f}, {:.2f} failed'.format(x, y))
            self._halt_dither = True
            getLogger('ConexManager').error('Error on move', exc_info=True)
        except:                                                     # I dont think this should happen??
            self._updateState('Error: move to {:.2f}, {:.2f} failed'.format(x, y))
            self._halt_dither = True
            getLogger('ConexManager').error('Unexpected error on move', exc_info=True)
        if self._startedMove>0:
            with self._rlock:
                self.cur_status=self.status()
                self._completedMoves+=1
                

class ConexAPI(Resource):
    """
    class that defines the possible commands from the POST request
    """

    def post(self):
        """
        argument data is sent in a json blob dictionary

        Arguments need to include the command type:
            ie. status', 'move', 'dither', 'stop', 'queryMove', 'queryDither'
        Additional arguments are passed to the function as required
        """
        parser = reqparse.RequestParser()
        choices=('status', 'move', 'dither', 'stop', 'queryMove', 'queryDither')
        parser.add_argument('command', type=str, required=True, choices=choices,
                                   help='Action', location='json')      
        args=parser.parse_args()

        if args.command == 'stop':
            getLogger('ConexManager').info('Stopping')
            ret=conex_manager.stop()
        elif args.command == 'status':
            getLogger('ConexManager').info('Status')
            ret=conex_manager.status()
        elif args.command == 'move':
            getLogger('ConexManager').info('Moving')
            parser.add_argument('x', type=float, required=True,
                                   help='X angle', location='json')
            parser.add_argument('y', type=float, required=True,
                                   help='Y angle', location='json')
            args=parser.parse_args()
            ret = conex_manager.start_move(args.x, args.y)
        elif args.command == 'dither':
            parser.add_argument('dither_dict', type=dict, required=True,
                                   help='dither dict', location='json')
            args=parser.parse_args()
            getLogger('ConexManager').info('Dithering')
            ret = conex_manager.start_dither(args.dither_dict)
        elif args.command == 'queryDither':
            #getLogger('ConexManager').info('Query Dither')
            ret = conex_manager.queryDither()
        elif args.command == 'queryMove':
            #getLogger('ConexManager').info('Query Move')
            ret = conex_manager.queryMove()
        if args.command in choices:
            return ret, 200
        else:
            getLogger('ConexManager').error('Unknown command: '+str(args.command))
            return None, 400

def dither(dither_dict,address='http://localhost:50001', timeout=TIMEOUT):
    """
    Client side function: Tells conex mount to start a dither

    INPUTS:
        dither_dict - see conex_manager.dither()

    Returns:
        HTTP status code
    """
    req = {'command':'dither','dither_dict':dither_dict}
    r=requests.post(address+'/conex', json=req, timeout=timeout)
    return r.json()

def move(x,y,address='http://localhost:50001', timeout=TIMEOUT):
    """
    Client side function: Tells conex mount to move to position x,y (in degrees)

    Returns:
        HTTP status code
    """
    req={'command':'move', 'x':x, 'y':y}
    r=requests.post(address+'/conex', json=req, timeout=timeout)
    return r.json()

def stop(address='http://localhost:50001', timeout=TIMEOUT):
    """
    Client side function: Tells conex mount to stop all movement

    Returns:
        dictionary: {'state':self.state, 'pos':pos, 'conexstatus':status, 'limits':self.conex.limits}
        HTTP status code
    """
    r=requests.post(address+'/conex', json={'command':'stop'}, timeout=timeout)
    return r.json()

def status(address='http://localhost:50001', timeout=TIMEOUT):
    """
    Client side function: Gets current status of connex mount

    Returns:
        dictionary: {'state':self.state, 'pos':pos, 'conexstatus':status, 'limits':self.conex.limits}
        HTTP status code
    """
    r = requests.post(address+'/conex', json={'command':'status'}, timeout=timeout)
    return r.json()

def queryDither(address='http://localhost:50001', timeout=TIMEOUT):
    """
    Client side function: Checks if the latest dither has completed

    Returns:
        dictionary: {'completed':True/False, 'status':self.cur_status, 'estTime': estTime, 'dither':dith}
                    dith is a dictionary containing the last dither info (or None if not completed yet)
                    status is a dictionary like {'state':self.state, 'pos':pos, 'conexstatus':status, 'limits':self.conex.limits}
        HTTP status code
    """
    r = requests.post(address+'/conex', json={'command':'queryDither'}, timeout=timeout)
    return r.json()
def queryMove(address='http://localhost:50001', timeout=TIMEOUT):
    """
    Client side function: Checks if the latest move has completed

    Returns:
        dictionary: {'completed': True/False, 'status':self.cur_status}
                    status is a dictionary like {'state':self.state, 'pos':pos, 'conexstatus':status, 'limits':self.conex.limits}
        HTTP status code
    """
    r = requests.post(address+'/conex', json={'command':'queryMove'}, timeout=timeout)
    return r.json()


if __name__=='__main__':
    from flask import Flask

    create_log('ConexManager',
                console=True, mpsafe=True, propagate=False,
                fmt='%(asctime)s %(levelname)s %(message)s',
                level=mkidcore.corelog.DEBUG)
    create_log('conex',
                console=True, mpsafe=True, propagate=False,
                fmt='%(asctime)s %(levelname)s %(message)s',
                level=mkidcore.corelog.DEBUG)

    app = Flask(__name__, static_url_path="")
    flasklog = getLogger('werkzeug')
    flasklog.setLevel(mkidcore.corelog.ERROR)
    api = Api(app)
    api.add_resource(ConexAPI, '/conex', endpoint='conex')

    conex_manager = ConexManager(port=CONEX_COM_PORT)
    app.run(host='0.0.0.0', debug=False, port=CONEX_SERVER_PORT)
    conex_manager.conex.close()
    time.sleep(0.1) # wait for conex to close



