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
from win32com.client import Dispatch
import traceback
import platform
from mkidcore.threads import start_new_thread, print2socket

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

global_KILL_SERVER = False


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
        msg = 'Bind failed. Error Code: {} Message: {}'.format(err[0], err[1])
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
                print2socket(result, the_socket=conn)
            except ValueError as e:
                print2socket('bad command:  {}'.format(e), the_socket=conn)

        except Exception as e:
            msg = 'Server Connection Loop Exception {}: \n'.format(e) + traceback.format_exc()
            log.error(msg)
            conn.sendall(msg)
            print2socket(msg, the_socket=conn)
            try:
                enum = e.errno
            except AttributeError:
                enum = 0
            if enum in (errno.EBADF, errno.ECONNRESET):
                # TODO closeout filter
                break

    log.info('Closing connection')
    print2socket('Closing connection', the_socket=conn)
    conn.close()

def setfilterbynumber(num, home=False):
    if num not in (1,2,3,4,5,6):
        return
    try:
        fwheels = Dispatch("OptecHID_FilterWheelAPI.FilterWheels")
        wheel = fwheels.FilterWheelList[0]
        #wheel.FirmwareVersion

        if home:
            wheel.HomeDevice()  #HomeDevice_Async()
        wheel.CurrentPosition = num
        return wheel.ErrorState
    except Exception:
        error = traceback.format_exc()
        getLogger(__name__).error('Caught error', exc_info=True)
        return error


def connect(host, port, verbose=True):
    """Starts a client socket (control computer). This will connect to the host
    and return the socket if successful

    Args:
        host (str): the server name or ip address.
        port (int): the port to use
        verbose (True): log stuff
    Returns:
        returns a socket connection or None
    Raises:
    """
    if verbose:
        log.info('Trying to connect with ' + host)
    try:
        sock = socket.create_connection((host, port), timeout=10)
    except socket.error:
        if verbose:
            log.error('Connection with ' + host + ' failed', exc_info=True)
        return None
    if verbose:
        log.info('Connected with ' + host)
    # NB I'm uneasy about this timeout but it does not seem to cause issues
    sock.settimeout(None)
    return sock


def setfilter(fnum, home=False, host='localhost:50000'):
    host,port=host.split(':')
    conn = connect(host, port)
    try:
        conn.sendall('{}\n'.format(-fnum if home else fnum))
        data = conn.recv(2048).strip()
        getLogger('HSFW').info("Response: {}".format(data))
        conn.close()
        return data
    except AttributeError:
        msg = 'Cannot connect to filter server\n'
        getLogger('HSFW').error(msg)
        return msg
    except Exception as e:
        msg = 'Cannot send command to filter server "Filter: {}"\n'
        getLogger('HSFW').error(msg.format(fnum), exc_info=True)
        try:
            conn.close()
        except Exception :
            getLogger('HSFW').error('error:', exc_info=True)
        raise e


if __name__ == '__main__':

    if platform.system=='Linux':
        sys.exit(1)
    else:
        start_server(50000)