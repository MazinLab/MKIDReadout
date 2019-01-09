"""
Author: Alex Walter
Date: Aug 29, 2018

This code is for dithering the Newport Conex-AG-M100D stage in the MEC fore-optics
"""
from __future__ import print_function
import serial
import time
import numpy as np
from mkidcore.corelog import getLogger
from threading import RLock, Thread
import itertools
import argparse
from flask_restful import Api, Resource, reqparse, fields, marshal
import requests


TIMEOUT = .25
CONEX_PORT = 50001


# from flask_httpauth import HTTPBasicAuth
# auth = HTTPBasicAuth()
#
# @auth.get_password
# def get_password(username):
#     if username == 'mec':
#         return 'python'
#     return None
#
#
# @auth.error_handler
# def unauthorized():
#     # return 403 instead of 401 to prevent browsers from displaying the default auth dialog
#     return make_response(jsonify({'message': 'Unauthorized access'}), 403)


dither_fields = {
    'nSteps': fields.Fixed,
    'start': {'x':fields.Float,'y':fields.Float},
    'end': {'x':fields.Float,'y':fields.Float},
    'intTime': fields.Float,
    'uri': fields.Url('dither')
}

dither_path_fields = {'path': fields.List(fields.List(fields.Float)), 'dither':fields.String,
                   'start': fields.List(fields.Float), 'end': fields.List(fields.Float)}
conex_fields = {
    'conexstatus': fields.String,
    'xpos': fields.Float,
    'ypos': fields.Float,
    'state': fields.String,
    'limits': fields.Nested({'umin': fields.Float, 'umax': fields.Float,
                             'vmin': fields.Float, 'vmax': fields.Float}),
    'last_dither': fields.Nested({'path': fields.List(fields.List(fields.Float)),
                                  'dither': fields.String,
                                  'start': fields.List(fields.Float),
                                  'end': fields.List(fields.Float)})
}


class Conex(object):
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
            self._init()
        except serial.SerialException:
            pass

    def _init(self):
        try:
            self._device = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=self.bytesize,
                                         stopbits=self.stopbits, timeout=self.timeout, xonxoff=self.xonxoff)
            self._started = 1
            time.sleep(0.1)  # wait for port to open
            self.write('ID?')
            self.status

            try:
                q = [self.query(q) for q in ('SLU?','SLV?','SRU?','SRV?')]
                f = lambda x: float(x[4:-2])
                self.u_lowerLimit = f(q[0])
                self.v_lowerLimit = f(q[1])
                self.u_upperLimit = f(q[2])
                self.v_upperLimit = f(q[3])
            except ValueError:
                raise
            self.close()
        except serial.SerialException:
            getLogger('conex').debug("Could not connect to Conex Mount", exc_info=True)
            raise

        getLogger('conex').debug("Connected to Conex Mount")
        self._started = 2

    def close(self):
        self._device.close()

    def open(self):
        if not self._started:
            self._init()
        if self._started==1:
            return
        if self._device.is_open:
            return
        self._device.open()
        self._device.write('*IDN?\r\n'.encode())
        self.read()

    @property
    def status(self):
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
        INPUT:
            command - The two letter ascii command (CC above)
        """
        with self._rlock:
            self.open()
            self._device.write('{}{}\r\n'.format(self.ctrlN, command).encode())

    def read(self, bufferSize=1):
        """
        read from the port buffer

        This is a blocking function. It will try to read until
        it sees the end line characters \r\n or it times out.
        """
        with self._rlock:
            bufferSize = abs(bufferSize)
            if not bufferSize:
                return ''
            ret = new = self._device.read(bufferSize).decode()
            if not ret:
                return ret
            while ret[-2:]  != '\r\n':
                #getLogger('conex').debug('Got "{}"'.format(new))
                new = self._device.read(self._device.in_waiting).decode()
                ret += new
            return ret

    def query(self, command, bufferSize=1):
        """
        Send an ascii command to the conex controller
        Read the output
        """
        with self._rlock:
            self.open()
            self._device.reset_output_buffer()  # abandon any command it's currently sending
            self._device.reset_input_buffer()  # clear the input buffer
            self.write(command)
            self._device.flush()  # wait for command to finish writing
            return self.read(bufferSize)

    @property
    def limits(self):
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
        """
        with self._rlock:
            if not self.inBounds(pos):
                raise IOError('Target position outside of limits. Aborted move.')
            self.write('PAU' + str(pos[0]))
            self._device.flush()  # wait until write command finishes sending
            self.write('PAV' + str(pos[1]))  # Conex is capable of moving both axes at once!
            if blocking:
                t = time.time()
                while not self.ready():
                    if time.time() - t > timeout:
                        status = self.query('TS')
                        raise IOError("Move Abs timed out. Status: " + status[:-2])
                    time.sleep(0.0001)

    def inBounds(self, target):
        """
        Check that the target is within the movement limits

        INPUT:
            target - [U,V] tuple position in degrees
        """
        return (self.u_lowerLimit <= target[0] <= self.u_upperLimit and
                self.v_lowerLimit <= target[1] <= self.v_upperLimit)

    def home(self):
        self.move((0, 0))

    def position(self):
        """ Returns the pos [U,V] in degrees """
        return float(self.query('TPU')[4:-2]), float(self.query('TPV')[4:-2])


class ConexDummy(object):
    def __init__(self, port="COM9", baudrate=912600, bytesize=serial.EIGHTBITS,
                 stopbits=serial.STOPBITS_ONE, timeout=1., xonxoff=True, controllerNum=1):

        self._rlock = RLock()
        # Generally 1. This is for when you daisy chain multiple conex stages together
        self.ctrlN = controllerNum
        time.sleep(0.1)  # wait for port to open

        self.u_lowerLimit = -1000
        self.v_lowerLimit = -1000
        self.u_upperLimit = 1000
        self.v_upperLimit = 1000

        getLogger('conex').debug("Connected to Conex Mount")

    def close(self):
        pass

    def open(self):
        pass

    @property
    def status(self):
        status = self.query('TS')
        getLogger('conex').debug("Status: " + status[7:-2])
        getLogger('conex').debug("Err State: " + status[3:-4])
        return status

    def write(self, command):
        """
        Send an ascii command to the conex controller
        Format is: xxCCnn\r\n
                   xx is the controller number (usually 1 unless you daisy chain multiple conex stages)
                   CC is the command
                   nn is the option input for the command
                   \r\n is the terminator character letting Conex know you've finished sending the command
        INPUT:
            command - The two letter ascii command (CC above)
        """
        with self._rlock:
            1+1

    def read(self, bufferSize=1):
        """
        read from the port buffer

        This is a blocking function. It will try to read until
        it sees the end line characters \r\n or it times out.
        """
        with self._rlock:
            return '0'*bufferSize

    def query(self, command, bufferSize=0):
        """
        Send an ascii command to the conex controller
        Read the output
        """
        with self._rlock:
            self.write(command)
            return self.read(bufferSize)

    def ready(self):
        """
        Checks the status of the conex

        returns True if it's ready for another command (like move)
        returns false if not ready
        If there was an error it raises an IOError
        """
        with self._rlock:
            self.write('TS')
            status = '00000003200'
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
        """
        with self._rlock:
            if not self.inBounds(pos):
                raise IOError('Target position outside of limits. Aborted move.')
            self.write('PAU' + str(pos[0]))
            self.write('PAV' + str(pos[1]))  # Conex is capable of moving both axes at once!
            if blocking:
                t = time.time()
                while not self.ready():
                    if time.time() - t > timeout:
                        status = self.query('TS')
                        raise IOError("Move Abs timed out. Status: " + status[:-2])
                    time.sleep(0.0001)

    def inBounds(self, target):
        """
        Check that the target is within the movement limits

        INPUT:
            target - [U,V] tuple position in degrees
        """
        return (self.u_lowerLimit <= target[0] <= self.u_upperLimit and
                self.v_lowerLimit <= target[1] <= self.v_upperLimit)

    def home(self):
        self.move((0, 0))

    def position(self):
        """ Returns the pos [U,V] in degrees """
        return (3.14,3.14)


class ConexStatus(object):
    def __init__(self, state='offline', pos=(np.NaN, np.NaN), conexstatus='',
                 dither=None, limits=None):
        """
            state is a ConexManager state : 'idle' 'offline' 'processing' 'stopped/stopping'
                                         'moving to {}, {}'
                                         'move to {:.2f}, {:.2f} failed'
            pos is the (x,y) pos (nan nan) if issues
            limits are the conex limits (a dict umin umax vmin vmax)
            dither is a dither result or None
            conexstatus is the conex result of TS or ''
        """
        self.state = state
        self.pos = pos
        self.conexstatus = conexstatus
        self.last_dither = dither
        self.limits = limits

    @property
    def xpos(self):
        return self.pos[0]

    @property
    def ypos(self):
        return self.pos[1]

    @property
    def running(self):
        return 'moving' in self.state or 'processing' in self.state

    @property
    def haserrors(self):
        return 'error' in self.state

    @property
    def offline(self):
        return 'offline' in self.state

    def __str__(self):
        return self.state

    def print(self):
        print(self.state)
        print(self.pos)
        print(self.conexstatus)
        print(str(self.last_dither))

    # def __eq__(self, o):
    #     return (self.state == o.state and
    #             self.pos == o.pos and
    #             self.conexstatus == o.conexstatus and
    #             self.limits == o.limits and
    #             self.)


class DitherPath(object):
    def __init__(self, dither, start_t, end_t, path):
        self.dither = dither
        self.start = start_t
        self.end = end_t
        self.path = path

    def __str__(self):
        return '\n'.join(map(str, (self.dither, self.start, self.end, self.path)))




class Dither(object):
    def __init__(self, nSteps=5, start=(-.76, -0.76), end=(0.0, 0.76), intTime=.1):
        self.nSteps = nSteps
        self.start = start  # x,y
        self.end = end
        self.intTime = intTime
        self.id = None

    def __str__(self):
        return ('{s[0]:.2f}, {s[1]:.2f} -> {e[0]:.2f}, {e[1]:.2f}, '
                '{n} steps {t} seconds').format(n=self.nSteps, s=self.start,
                                                e=self.end, t=self.intTime)


class ConexManager(object):
    def __init__(self, port):
        self.conex = Conex(port=port)
        self._dither_result = None
        self._active_dither = None
        self._movement_thread = None
        self._halt = False
        self.state = 'idle'

    def status(self):
        """
        returns a conex status object:
            state is a ConexManager state : 'idle' 'offline' 'processing' 'stopped/stopping'
                                         'moving to {}, {}'
                                         'move to {:.2f}, {:.2f} failed'
            pos is the (x,y) pos (nan nan) if issues
            limits are the conex limits (a dict umin umax vmin vmax)
            dither is a dither result or None
            conexstatus is the conex result of TS or ''
        """
        status = ''
        pos = (np.NaN, np.NaN)
        try:
            status = self.conex.status
            pos = self.conex.position()
            getLogger('ConexManager').debug("Conex: {} @ pos {}".format(status,pos))
        except (IOError, serial.SerialException):
            getLogger('ConexManager').error('Unable to get conex status', exc_info=True)
            self._halt = True
            self.state = 'offline'
        return ConexStatus(state=self.state, pos=pos, conexstatus=status,
                           dither=self._dither_result, limits=self.conex.limits)

    def start_dither(self, id):
        """id is either a key into dithers or a dictionary for creating a dither """
        self.state = 'processing'
        dither = dithers[id] if id in dithers else Dither(**id)
        self.stop(wait=True)
        self._movement_thread = Thread(target=self.dither, args=(dither,), name=str(dither))
        self._movement_thread.daemon = True
        self._movement_thread.start()
        return self.status()

    def start_move(self, x, y):
        self.state = 'processing'
        getLogger('ConexManager').error("Starting move to {:.2f}, {:.2f}".format(x,y))
        self.stop(wait=True)
        self._movement_thread = Thread(target=self.move, args=(x, y,),
                                       name='Move to ({}, {})'.format(x,y))
        self._movement_thread.daemon = True
        self._movement_thread.start()
        return self.status()

    def stop(self, wait=True):
        getLogger('ConnexManager').info('stopping')
        self._halt = True
        if wait and self._movement_thread is not None:
            self._movement_thread.join()
        self.state = 'stopped'

    def move(self, x, y):
        self.state = 'moving to {:.2f}, {:.2f}'.format(x, y)
        try:
            self.conex.move((x, y))
            self.wait_on_conex()
            self.state = 'idle'
            return True
        except (IOError, serial.SerialException) as e:
            self.state = 'error: move to {:.2f}, {:.2f} failed'.format(x, y)
            getLogger('ConexManager').error('Error on move', exc_info=True)
            return False

    def dither(self, dither, return_to_start=False):
        """
        Do a dither

        INPUT:
            outputPath - Path to save log file in. ie. '/home/data/ScienceData/Subaru/20180622/'. The filename is *timestamp*_dither.log
            nSteps - Number of steps in each x and y direction. There will be nSteps**2 points in a grid
            startX - theta direction
            endX -
            startY - phi direction
            endY -
            intTime - Time to integrate at each x/y location
        """
        self._active_dither = dither
        self._halt = False
        self.state = 'dithering: {}'.format(str(dither))
        estOverhead = 35. / 25.
        runtime = dither.nSteps ** 2. * (dither.intTime + estOverhead)
        getLogger('DitherManager').info("Est Time: {:.1f} s".format(runtime))

        x_list = np.linspace(dither.start[0], dither.end[0], dither.nSteps)
        y_list = np.linspace(dither.start[1], dither.end[1], dither.nSteps)
        startTimes = []
        endTimes = []
        pos = []
        ts = time.time()

        try:
            startPos = self.conex.position()
            self._dither_result = None

            for x, y in itertools.product(x_list, y_list):

                if not self._move((x,y)):
                    self._dither_result = DitherPath(dither, startTimes, endTimes, pos)
                    return False

                startTimes.append(time.time())
                u, v = self.conex.position()

                pos.append((u,v))
                getLogger('ConexManager').info('({}, {}) -> ({}, {})'.format(x, y, u, v))

                intTime = dither.intTime - (time.time()-startTimes[-1])
                while intTime>0 and not self._halt:
                    time.sleep(min(intTime, .25))
                    intTime -= .25

                endTimes.append(time.time())

            self._dither_result = DitherPath(dither, startTimes, endTimes, pos)

            if return_to_start:
                if not self._move(startPos):
                    return False

            totalTime = time.time() - ts
            overhead = totalTime - dither.nSteps ** 2. * dither.intTime
            getLogger('ConexManager').info("Total dither time: {:.1f}s".format(totalTime))
            getLogger('ConexManager').debug("Dither overhead: {:.1f}s".format(overhead))

            self.state = 'idle'
            return True

        except (IOError, serial.SerialException) as e:
            getLogger('ConexManager').error("Dither failed ", exc_info=True)
            #TODO do we kill the conex object here?
            self.state = self.state = 'dither {} failed'.format(str(dither))
            return False
        finally:
            self._active_dither = None

    def _move(self, pos):
        if self._halt:
            return False
        self.conex.move(pos)
        self.wait_on_conex()
        if self._halt:
            return False
        return True

    def wait_on_conex(self, timeout=15):
        """wait until conex in position, timeout or _halt gets set"""
        movestart = time.time()
        while not self.conex.ready():
            if self._halt:
                return False  #TODO NB this doesn't actually halt the stage, just quits waiting on it
            if time.time() - movestart > timeout:
                raise IOError('Move timeout, unable to achieve position')
            time.sleep(.1)
        return True


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s must be positive" % value)
    return ivalue


class MoveAPI(Resource):
    # decorators = [auth.login_required]

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('x', type=float, required=True,
                                   help='Desired X position', location='json')
        self.reqparse.add_argument('y', type=float, required=True,
                                   help='Desired Y position', location='json')
        super(MoveAPI, self).__init__()

    def get(self):
        return marshal(conex_manager.status(), conex_fields)

    def post(self):
        args = self.reqparse.parse_args()
        getLogger('ConexAPI').debug('API request to move to {:.2f}, {:.2f}'.format(args.x,args.y))
        conex_manager.start_move(args.x, args.y)
        return self.get(), 201


class ConexAPI(Resource):
    # decorators = [auth.login_required]

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('command', type=str, required=True, choices=('stop',),
                                   help='Action', location='json')
        super(ConexAPI, self).__init__()

    def get(self):
        stat = conex_manager.status()
        stat.print()
        return marshal(stat, conex_fields)

    def post(self):
        args = self.reqparse.parse_args()
        if args.command == 'stop':
            conex_manager.stop()
        return self.get(), 201


class DitherAPI(Resource):
    # decorators = [auth.login_required]

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('id', type=str, required=False, location='json',
                                   choices=dithers.keys())
        self.reqparse.add_argument('dither', type=dict, required=False, location='json')

        self.dither_parser = reqparse.RequestParser()
        self.dither_parser.add_argument('startx', type=float, required=True, location='dither')
        self.dither_parser.add_argument('starty', type=float, required=True, location='dither')
        self.dither_parser.add_argument('n', type=check_positive, required=True, location='dither')
        self.dither_parser.add_argument('t', type=check_positive, required=True, location='dither')
        self.dither_parser.add_argument('endx', type=float, required=True, location='dither')
        self.dither_parser.add_argument('endy', type=float, required=True, location='dither')
        super(DitherAPI, self).__init__()

    def get(self):
        return marshal(conex_manager.status(), conex_fields)

    def post(self):
        self.reqparse.args[0].choices = dithers.keys()
        args = self.reqparse.parse_args()
        if args.dither is not None:
            dither = self.dither_parser.parse_args(req=args)
        else:
            dither = args.id

        conex_manager.start_dither(dither)
        return self.get(), 201


#NB make_response(jsonify({'message':'Invalid dither ID'}), 400)

dithers = {"0": Dither()}


def dither(id='default', start=None, end=None, n=1, t=1, address='http://localhost:50001', timeout=TIMEOUT):
    """ Do a Dither by ID or full settings. Uses full settings if start is not None
    returns a requests result object
    """
    if start is not None:
        req = {'startx': start[0], 'starty': start[1], 'endx': end[0], 'endy': end[1],
               'n': n, 't': t}
    else:
        req = {'id': id}
    try:
        r = requests.post(address+'/dither', json=req, timeout=timeout)
        j = r.json()
        ret = ConexStatus(state=j['state'], pos=(j['xpos'], j['ypos']), conexstatus=j['conexstatus'],
                          dither=DitherPath(j['last_dither']['dither'], j['last_dither']['start'],
                                            j['last_dither']['end'], j['last_dither']['path']))
    except requests.ConnectionError:
        ret = ConexStatus(state='error: unable to connect')

    return ret


def move(x, y, address='http://localhost:50001', timeout=TIMEOUT):
    try:
        r = requests.post(address+'/move', json={'x': x, 'y': y}, timeout=timeout)
        j = r.json()
        ret=ConexStatus(state=j['state'], pos=(j['xpos'],j['ypos']), conexstatus=j['conexstatus'],
                        dither=DitherPath(j['last_dither']['dither'], j['last_dither']['start'],
                                          j['last_dither']['end'], j['last_dither']['path']))
    except requests.ConnectionError:
        ret = ConexStatus(state='error: unable to connect')
    return ret


def status(address='http://localhost:50001', timeout=TIMEOUT):
    try:
        r = requests.get(address + '/conex', timeout=timeout)
        j = r.json()
        ret=ConexStatus(state=j['state'], pos=(j['xpos'],j['ypos']), conexstatus=j['conexstatus'],
                        dither=DitherPath(j['last_dither']['dither'], j['last_dither']['start'],
                                          j['last_dither']['end'], j['last_dither']['path']))
    except requests.ConnectionError:
        ret = ConexStatus(state='error: unable to connect')
    return ret


def stop(address='http://localhost:50001', timeout=TIMEOUT):
    try:
        r = requests.post(address + '/conex', json={'command': 'stop'}, timeout=timeout)
        j = r.json()
        ret = ConexStatus(state=j['state'], pos=(j['xpos'], j['ypos']), conexstatus=j['conexstatus'],
                          dither=DitherPath(j['last_dither']['dither'], j['last_dither']['start'],
                                            j['last_dither']['end'], j['last_dither']['path']))
    except requests.ConnectionError:
        ret = ConexStatus(state='error: unable to connect')

    return ret


if __name__=='__main__':
    from flask import Flask, jsonify, make_response

    app = Flask(__name__, static_url_path="")
    api = Api(app)
    api.add_resource(MoveAPI, '/move', endpoint='move')
    api.add_resource(DitherAPI, '/dither', endpoint='dither')
    api.add_resource(ConexAPI, '/conex', endpoint='conex')

    conex_manager = ConexManager(port='COM9')
    app.run(host='0.0.0.0', debug=True, port=CONEX_PORT)
