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
import errno, sys
from mkidcore.corelog import getLogger
import mkidcore.safesocket as socket
import traceback
import platform
import threading
import argparse
import mkidcore.corelog

HSFWERRORS = {0: 'No error has occurred. (cleared state)',
              1: 'The 12VDC power has been disconnected from the device.',
              2: 'The device stalled during a Home of Move procedure.',
              3: 'An invalid parameter was received during communication with the host(PC).',
              4: 'An attempt to Home the device was made while the device was already in motion.',
              5: 'An attempt to change filter positions was made while the device was already in motion.',
              6: 'An attempt to change filter positions was made before the device had been homed.'}

# Get-ChildItem HKLM:\Software\Classes -ErrorAction SilentlyContinue | Where-Object {$_.PSChildName -match '^\w+\.\w+$' -and (Test-Path -Path "$($_.PSPath)\CLSID")} | Select-Object -ExpandProperty PSChildName
# OptecHIDTools.DeviceChangeNotifier
# OptecHIDTools.HIDMonitor
# OptecHIDTools.HID_API_Wrapers
# OptecHIDTools.ReadWrite_API_Wrappers
# OptecHIDTools.Setup_API_Wrappers
# OptecHIDTools.Win32Errors
# OptecHID_FilterWheelAPI.FilterWheels
# from win32com.client import Dispatch
# fwheels = Dispatch("OptecHID_FilterWheelAPI.FilterWheels")

# self.mccdaq = ct.windll.LoadLibrary(MCCDAQLIB)
# self.lib = ct.CDLL(EPOS2Shutter.LIB_PATH)
# self.lib.VCS_ResetDevice(self.dev_handle, self.node_id, ct.byref(err))
# err.value

HSFW_PORT = 50000
NFILTERS=NUM_FILTERS=5
global_KILL_SERVER = False


def start_server(port, log=None):
    """
    starts a server: the server will receive commands
    from the client (control computer) and respond adequately

    Args:
        port: port to run on

    Raises:
    """

    global global_KILL_SERVER

    if log is None:
        log = getLogger(__name__)

    # get IP address
    host = socket.gethostbyname(socket.gethostname())

    # create a socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # bind socket to local host and port
    try:
        sock.bind((host, port))
    except socket.error:
        log.critical('Bind Failed: ', exc_info=True)
        sys.exit()

    log.info('Socket bound')

    # start listening on socket
    sock.listen(1)

    # it's a server, just let it run
    while not global_KILL_SERVER:
        # wait to accept a connection - blocking call
        conn, addr = sock.accept()
        log.info('Connected with ' + addr[0] + ':' + str(addr[1]))
        try:
            thread = threading.Thread(target=connection_handler, args=(conn,), name='ClientThread')
            thread.daemon = True
            thread.start()
        except Exception:
            getLogger(__name__).critical(exc_info=True)
    sock.close()


def connection_handler(conn):
    # main loop
    global global_KILL_SERVER
    while True:
        try:
            # may raise error: [Errno 10054] An existing connection was forcibly closed by the remote host
            # if a network fault occues
            data = conn.recv(1024).decode('utf-8').strip()

            if 'exit' in data:
                #TODO closeout filter
                conn.sendall('exiting'.encode('utf-8'))  # confirm stop to control
                global_KILL_SERVER = True
                break

            if '?' in data:
                conn.sendall('{}'.format(_getfilter()).encode('utf-8'))
                continue

            try:
                fnum = int(data)
                if abs(fnum) not in (1,2,3,4,5,6):
                    raise ValueError('Filter must be 1-6!')
                result = HSFWERRORS[_setfilter(abs(fnum), home=fnum < 0)]
                conn.sendall(result.encode('utf-8'))
            except ValueError as e:
                conn.sendall('bad command:  {}'.format(e).encode('utf-8'))
        except Exception as e:
            try:
                enum = e.errno
            except AttributeError:
                enum = 0
            if enum == errno.WSAECONNABORTED:
                getLogger(__name__).info('Connection Closed')
            else:
                msg = 'Server Connection Loop Exception {}: \n'.format(e) + traceback.format_exc()
                getLogger(__name__).error(msg)
                conn.sendall(msg.encode('utf-8'))
            if enum in (errno.EBADF, errno.ECONNRESET, errno.WSAECONNABORTED):
                # TODO closeout filter
                break

    try:
        conn.sendall('Closing connection'.encode('utf-8'))
        getLogger(__name__).info('Closing connection')
    except Exception:
        pass
    conn.close()


def connect(host, port, timeout=10.0):
    """Starts a client socket (control computer). This will connect to the host
    and return the socket if successful

    Args:
        host (str): the server name or ip address.
        port (int): the port to use
    Returns:
        returns a socket connection or None
    Raises:
    """
    log = getLogger(__name__)
    log.debug('Trying to connect with ' + host)
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except socket.error:
        log.error('Connection with ' + host + ' failed', exc_info=True)
        return None
    log.debug('Connected with ' + host)
    sock.settimeout(timeout)
    return sock


def _setfilter(num, home=False):
    from win32com.client import Dispatch
    import pythoncom, pywintypes
    pythoncom.CoInitialize()
    if num not in (1,2,3,4,5,6):
        return
    try:
        fwheels = Dispatch("OptecHID_FilterWheelAPI.FilterWheels")
        wheel = fwheels.FilterWheelList[0]
        getLogger(__name__).debug('Filter Wheel FW: {}'.format(wheel.FirmwareVersion))
        if home:
            getLogger(__name__).info('Homing...')
            wheel.HomeDevice  # or use HomeDevice_Async() for non blocking
            getLogger(__name__).info('Homing complete.')
        getLogger(__name__).info('Setting postion to {}'.format(num))
        wheel.CurrentPosition = num
        return wheel.ErrorState
    except pywintypes.com_error:
        return 'Error: Unable to communicate. Check filter wheel is connected. (comerror)'
        getLogger(__name__).error('Windows COM error. Filter probably disconnected', exc_info=True)
    except Exception:
        error = traceback.format_exc()
        getLogger(__name__).error('Caught error', exc_info=True)
        return error


def _getfilter():
    from win32com.client import Dispatch
    import pythoncom, pywintypes
    pythoncom.CoInitialize()
    try:
        fwheels = Dispatch("OptecHID_FilterWheelAPI.FilterWheels")
        wheel = fwheels.FilterWheelList[0]
        return wheel.CurrentPosition
        #return wheel.ErrorState
    except pywintypes.com_error:
        return 'Error: Unable to communicate. Check filter wheel is connected. (comerror)'
        getLogger(__name__).error('Windows COM error. Filter probably disconnected', exc_info=True)
    except Exception:
        error = traceback.format_exc()
        getLogger(__name__).error('Caught error', exc_info=True)
        return error


def getfilter(host='localhost:50000', timeout=.01):
    host, port = host.split(':')
    conn = connect(host, port, timeout=timeout)
    try:
        conn.sendall('?\n'.encode('utf-8'))
        data = conn.recv(2048).decode('utf-8').strip()
        getLogger(__name__).info("Response: {}".format(data))
        conn.close()
        print(data)
        if data.lower().startswith('error'):
            getLogger(__name__).error(data)
            return data
        return int(data)
    except AttributeError:
        msg = 'Cannot connect to filter server'
        getLogger(__name__).error(msg)
        return 'Error: ' +msg
    except Exception as e:
        msg = 'Cannot get status of filter server'
        getLogger(__name__).error(msg, exc_info=True)
        try:
            conn.close()
        except Exception as e:
            getLogger(__name__).error('error:', exc_info=True)
        return 'Error: '+str(e)


def setfilter(fnum, home=False, host='localhost:50000', killserver=False, timeout=2):

    if killserver:
        try:
            host, port = host.split(':')
            conn = connect(host, port, timeout=timeout)
            conn.sendall('exit\n'.encode('utf-8'))
            conn.sendall('exit\n'.encode('utf-8'))
            data = conn.recv(2048).decode('utf-8').strip()
            conn.close()
            return data
        except Exception:
            getLogger(__name__).error('error:', exc_info=True)
            return False


    try:
        fnum = int(fnum)
        if not 1 <= fnum <= NUM_FILTERS:
            raise TypeError
    except TypeError:
        raise ValueError('Not a number between 1-{}'.format(NUM_FILTERS))

    host, port = host.split(':')


    try:
        conn = connect(host, port, timeout=timeout)
        conn.sendall('{}\n'.format(-fnum if home else fnum).encode('utf-8'))
        data = conn.recv(2048).strip()
        getLogger(__name__).info("Response: {}".format(data))
        conn.close()
    except AttributeError:
        msg = 'Cannot connect to filter server'
        getLogger(__name__).error(msg)
        return False
    except Exception as e:
        msg = 'Cannot send command to filter server "Filter: {}"'
        getLogger(__name__).error(msg.format(fnum), exc_info=True)
        try:
            conn.close()
        except Exception:
            getLogger(__name__).error('error:', exc_info=True)
        return False

    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HSFW Server')
    parser.add_argument('--port', type=int, default=HSFW_PORT, help="Port on which to listen")
    args = parser.parse_args()

    if platform.system == 'Linux':
        print('Server application only supported on windows.')
        sys.exit(1)

    # import here so that we don't need the windows modules on a unix machine
    from win32com.client import Dispatch
    import pythoncom, pywintypes

    mkidcore.corelog.create_log('__main__', console=True, mpsafe=True, propagate=False,
                                fmt='%(asctime)s HSFW SERVER:%(levelname)s %(message)s')
    mkidcore.corelog.create_log('mkidcore', console=True, mpsafe=True, propagate=False,
                                fmt='%(asctime)s %(levelname)s %(message)s')
    mkidcore.corelog.create_log('mkidreadout', console=True, mpsafe=True, propagate=False,
                                fmt='%(asctime)s %(levelname)s %(message)s')
    start_server(args.port)
