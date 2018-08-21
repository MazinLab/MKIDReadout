"""
Dim msg
msg = "Starting Test..." & vbNewLine
'Create an instance of the FilterWheels Class
'this provides you with a means of accessing all of the FilterWheels attached to the PC Set FiltWheelsObj= WScript.CreateObject ("OptecHID_FilterWheelAPI.FilterWheels" )
'Get an ArrayList of FilterWheel Objects representing the attached devices
Set ListOfDevices=FiltWheelsObj.FilterWheelList
'Create a Reference to the first item in the list.
Set FW = ListOfDevices(0)
'Home the device
FW.HomeDevice
'Change to various positions
FW.CurrentPosition = 4
FW.CurrentPosition = 1
FW.CurrentPosition = 2
FW.CurrentPosition = 3
FW.CurrentPosition = 5
'Get the Number of Filters from the device
msg = msg & "This filter wheel has " & FW.NumberOfFilters & " filters" & vbNewLine 'Get the WheelID from the device (Make sure to cast it using chr())
msg = msg & "This filter wheel ID is " & chr(FW.WheelID) & vbNewLine
'Get the Firmware version from the device
msg = msg & "Firmware Version = " & FW.FirmwareVersion & vbNewLine
msg = msg & "Test Complete"
"""

import os, errno, sys, re
from mkidcore.corelog import getLogger
import mkidcore.safesocket as socket
from win32com.client import Dispatch
import threading
import time
import traceback
import os
from datetime import datetime


HSFWERRORS = {0: 'No error has occurred. (cleared state)',
              1: 'The 12VDC power has been disconnected from the device.',
              2: 'The device stalled during a Home of Move procedure.',
              3: 'An invalid parameter was received during communication with the host(PC).',
              4: 'An attempt to Home the device was made while the device was already in motion.',
              5: 'An attempt to change filter positions was made while the device was already in motion.',
              6: 'An attempt to change filter positions was made before the device had been homed.'}


# self.mccdaq = ct.windll.LoadLibrary(MCCDAQLIB)
# self.lib = ct.CDLL(EPOS2Shutter.LIB_PATH)
# self.lib.VCS_ResetDevice(self.dev_handle, self.node_id, ct.byref(err))
# err.value

log = getLogger('HSFW')  #TODO this isn't best practice but i don't think it will matter here



def start_server(port):
    """
    starts a server: the server will receive commands
    from the client (control computer) and respond adequately

    Args:
        port: port to run on

    Raises:
    """

    global global_KILL_SERVER

    # get IP address
    host = socket.gethostbyname(socket.gethostname())

    # create a socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # bind socket to local host and port
    try:
        sock.bind((host, port))
    except socket.error as err:
        msg='Bind failed. Error Code: {} Message: {}'.format(err[0], err[1])
        log.critical(msg)
        sys.exit()

    log.info('Socket bound')

    # start listening on socket
    sock.listen(1)

    # it's a server, just let it run
    while not global_KILL_SERVER:
        # wait to accept a connection - blocking call
        conn, addr = sock.accept()
        log.info('Connected with ' + addr[0] + ':' + str(addr[1]))
        start_new_thread(connection_handler, (conn,), name='ClientThread')
    sock.close()


def connection_handler(conn):
    # main loop
    global global_KILL_SERVER
    while True:
        try:
            # may raise error: [Errno 10054] An existing connection was forcibly closed by the remote host
            # if a network fault occues
            data = conn.recv(1024).strip()

            if data == 'exit':
                #TODO closeout filter
                conn.sendall('exiting')  # confirm stop to control
                global_KILL_SERVER = True
                break

            try:
                fnum = int(data)
                if abs(fnum) not in (1,2,3,4,5,6):
                    raise ValueError('Filter must be 1-6!')
                result = HSFWERRORS[setfilterbynumber(abs(fnum), home=fnum < 0)]
                myprint(result, the_socket=conn)
            except ValueError as e:
                myprint('bad command:  {}'.format(e), the_socket=conn)

        except Exception as e:
            msg = 'Server Connection Loop Exception {}: \n'.format(e) + traceback.format_exc()
            log.error(msg)
            conn.sendall(msg)
            myprint(msg, the_socket=conn)
            try:
                enum = e.errno
            except AttributeError:
                enum = 0
            if enum in (errno.EBADF, errno.ECONNRESET):
                # TODO closeout filter
                break

    log.info('Closing connection')
    myprint('Closing connection', the_socket=conn)
    conn.close()



def myprint(the_message, the_socket=None, timestamp=True, warning=False, error=False, queue=None, concat=None):
    """
    Sends information in order to be displayed in the GUI

    Arguments:
        the_message: message to display
    Optional keywords:
        the_socket: if provided, it is the socket used by the server to send
                    data back to the client. if not provided (default), this
                    function is used by the client to display data.
        timestamp (boolean): if True (default), a timestamp is added before
            the message
        error (boolean): if True (not default), the word 'ERROR: ' is added
            before the message
        warning (boolean): if True (not default), the word 'WARNING: ' is added
            before the message unless error is True
        textbox (Tkinter.Text): if provided, the message will be displayed in a
            Tkinter.Text widget
        concat (string): if provided, the_message is added to the concat string
        nolog: if False (default, only used if textbox is provided), it will
            also write the command in the log file
    """
    if error:
        the_message = 'ERROR: '+the_message
    elif warning:
        the_message = 'WARNING: '+the_message
    if timestamp:
        datestr = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        the_message = datestr+' > '+the_message
    if the_socket:
        try:
            the_socket.sendall(the_message)
        except:
            getLogger(__name__).error(the_message)
        time.sleep(0.1)
    elif concat != None:
        print the_message
        concat += the_message
        return concat
    elif queue:
        for line in the_message.splitlines():
            line = re.sub('(\n|\r)','<br>', line).rstrip().rstrip('<br>')
            if line:
                queue.put(line)


class Thread(object):
    """
    Class to start and stop each of the threads
    """
    def __init__(self, func, name, log, daemon=False):
        """
        Initialization
        """
        self._func = func
        self._name = name
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._log = log
        self._daemon = daemon
        self._thread = None

    @property
    def _is_running(self):
        """
        Is the thread running?
        """
        return not self._stop_event.is_set()

    def start(self, args, kwargs={}, main=False):
        """
        Starting the thread
        """
        if not self._is_running:
            self._log.info('Starting thread: ' + self._name)
            self._stop_event.clear()
            kwargs['stop_event'] = self._stop_event
            if main:
                self._func(*args, **kwargs)
            else:
                self._thread = start_new_thread(self._func, args, kwargs=kwargs,
                                                log=self._log, name=self._name,
                                                daemon=self._daemon)
        else:
            self._log.info('Thread arleady running: ' + self._name)

    def stop(self):
        """
        Starting the thread
        """
        if self._is_running:
            self._log.info('Ending thread: ' + self._name)
            self._stop_event.set()
        else:
            self._log.info('Cannot stop non-running thread: ' + self._name)


def start_new_thread(target, args, kwargs={}, log=None, name='',
                     timestamp=True, socket=None, daemon=False):
    """
    Starts a new thread with target function. If not successful, error is
    logged.
    Args:
        target (function): Function to execute in the new thread.
        args (tuple): Tuple with arguments of the function.
        kwargs (dictionary, optional): dictionary of keyword arguments for the
            function. Defaults to {}.
        queue (Queue object, optional): Queue to send error messages.
            Defaults to None.
        name (str, optional): Thread name. Defaults to ''.

    Returns:

    Raises:
    """
    try:
        thread = threading.Thread(target=target, args=args, name=name,
                                  kwargs=kwargs)
        if daemon:
            thread.daemon = True
        thread.start()
        return thread
    except Exception:
        errmsg = traceback.format_exc()
        if log:
            log.error(errmsg)
        else:
            myprint(errmsg, timestamp=timestamp, the_socket=socket)



def setfilterbynumber(num, home=False):

    if num not in (1,2,3,4,5,6):
        return

    fwheels = Dispatch("OptecHID_FilterWheelAPI.FilterWheels")
    wheel = fwheels.FilterWheelList[0]
    #wheel.FirmwareVersion

    if home:
        wheel.HomeDevice()  #HomeDevice_Async()
    wheel.CurrentPosition = num

    return wheel.ErrorState




def connect(host, port, verbose=True):
    """Starts a client socket (control computer). This will connect to the host
    and return the socket if successful

    Args:
        host (str): the server name or ip address.
        port (int): the port to use
    Returns:
        returns a tuple with a socket connection and a message to display
    Raises:
    """
    if verbose:
        log.info('Trying to connect with ' + host)

    try:
        sock = socket.create_connection((host, port), timeout=10)
    except socket.error:
        if verbose:
            log.error('Connection with ' + host + ' failed')
        return None

    if verbose:
        log.info('Connected with ' + host)

    # NB I'm uneasy about this timeout but it does not seem to cause issues
    sock.settimeout(None)

    return sock



def setfilter(fnum, home=False, host='localhost:50000'):
    host,port=host.split(':')
    conn = connect(host, port, '', port)
    try:
        conn.sendall('{}\n'.format(-fnum if home else fnum))
        data = conn.recv(1024).strip()
        #TODO log response and return response
        conn.close()
    except Exception as e:
        msg = 'Cannot send command to filter server "{}"\n'
        log.error(msg.format, exec_info=True)
        conn.close()
        raise e


if __name__ == '__main__':

    if os.platform=='Linux':
        sys.exit(1)
    else:
        start_server(50000)