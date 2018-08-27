import serial
import time
from mkidcore.corelog import getLogger

# Should Modify:
# Add method to class change serial timeout
# Implement error checking at end of each write (TE)
# Add functions for editing software limits
# Consider implementing configuration function which changes values in config state
# Provide a build mode which does not print
# Imporved time performance and outputs as returned values
# Can use buildFLG to supress prints and take it in as arg
# Can keep printDev() for printing if really needed
# Throws actual errors instead of print lines

DEVICE_NAME='/dev/connexxy'

def move(x,y, home=False, close=False):

    if need_to_connect:
        xy=ConexDevice(DEVICE_NAME)

        xy.open()

    if home:
        xy.home()

    xy.moveTo()

    if close:
        xy.close()


class ConexDevice(object):
    """Class for controlling the Newport Conex Stages (Linear Stage only for now)"""
    DELAY = .05  # Number of seconds to wait after writing a message
    MVTIMEOUT = 600  # (MVTIMEOUTxDELAY)= number of seconds device will

    # wait before declaring a move timeout error

    def __init__(self, devnm, baud=57600):
        """Create serial object and instantiate instance variables
        *devnm should be a string like '/dev/ttyUSB0'
        baud = the baudrate for comms. 57600 for Serial, 921600 for USB
        """
        # Create Serial Object for Communication (keep closed though)
        self.ser = serial.Serial()
        self.ser.baudrate = baud
        self.ser.port = devnm
        self.ser.timeout = 0.5

        # Other Instance Variables
        self.SN = 'DevNotOpenedYet'  # Device serial number
        # self.SN also serves as flag to check if device has been opened
        self.TY = 'DevNotOpenedYet'  # Device type
        self.FW = 'DevNotOpenedYet'  # Device Revision Information
        self.MXPS = 'DevNotOpenedYet'  # Device maximum Position (mm)
        self.MNPS = 'DevNotOpenedYet'  # Device minimum Position (mm)

    # :::::::::::::::::::::::PORT MANAGEMENT FUNCTIONS::::::::::::::::::::::::::
    def open(self):
        """Opens connection to device
        Also queries the device to obtain basic information
            This serves to confirm communication
        *Does not reopen device if already open
        """
        # Open port if not already open
        if self.ser.isOpen():
            getLogger(__name__).info('(SN:%s) is already open' % self.SN)
        else:
            getLogger(__name__).info('Connecting to : %s...' % self.ser.name)
            self.ser.open()

            # Send 'ID?' command to synchronize comms with device:
            # The first message sent after the device is powered up is
            # automatically ignored by the device. I did not want to send
            # '1RS' since this would reset the device everytime open() is called
            self.write('1ID?')
            self.readAll()  # clear read buffer

            # Request Device Information
            self.reqInfo()

            # Request Software Limits
            self.reqLim()

            getLogger(__name__).info('Device is a   : %s \n' % self.TY +
                  'Serial Number : %s \n' % self.SN +
                  'Frameware vs. : %s \n' % self.FW)

    def close(self):
        """Closes the device connection
        """
        self.ser.close()

    # :::::::::::::::::::::::WRITE/READ FUNCTIONS:::::::::::::::::::::::::::::::
    def write(self, MSG, append=None):
        """Formats a string, 'MSG' and sends it to the device
        MSG should be the message as string (ex. '1ID?')
            Should NOT include CR\LF
        Can append data (including numbers) to the end of the MSG
        *Data requests using 'write' should be followed by a read
            Otherwise unread items in buffer may cause problems
        **This function is useful for sending messages that do not
            have a dedicated function yet.
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        msg = MSG.encode()  # convert to bytes

        # convert 'append' and append to end
        if append != None:
            MSG = MSG + str(append)

        MSG = MSG + '\r\n'
        msg = MSG.encode()

        # Send message using pySerial (serial)
        self.ser.write(msg)

    def readAll(self):
        """Returns the full read buffer
        Also serves as a 'flush' function to clear buffer itself
            Useful for debugging reads to ensure read data is as expected
        Returns the read data as bytes in bytearray
            Does NOT strip() CR\LF at end of messages
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        time.sleep(self.DELAY)  # Give device time to respond
        rd = self.ser.readlines()

        return rd

    def read(self):
        """Reads a sinlge line from the readbuffer
        Strips the CR\LF and decodes it into a string
        *DOES NOT CHECK IF PORT IS OPEN
            Thus, performing read with port closed could throw errors
        Returns:
            str result of single-line read
        """
        # Does not check if port is open to avoid slow-downs from checking
        # if port is open repeatedly when back-to-back reads are performed
        return self.ser.readline().strip().decode()

    # :::::::::::::::::::::::DEVICE PRINT FUNCTION::::::::::::::::::::::::::
    def printDev(self):
        """Prints the device properties with labels and formatting
        Also updates instance variables if possible
        Very useful for debugging
        Works even if the port is closed
        *If port is closed, returns instance variables
        *If port is open, re-reads and returns all variables
        """
        # Exit if values have not been instantiated
        if self.SN == 'DevNotOpenedYet':
            getLogger(__name__).info('ERROR::: Variables not instantiated yet:\n' +
                  '  open() must be called at least once before printDev()')
            return

        # Create second part of message for use when device is open
        msgOut = ''

        getLogger(__name__).info('printDev() Result:_____')

        # Check if port is open and format output accordingly
        if self.ser.isOpen():
            # Check values available in all states____________________
            self.write('1TB')  # get command error string
            rd = self.read()
            msgOut = msgOut + '  Last Error       : ' + rd[3:]

            self.write('1TH')  # get target position
            rd = self.read()
            msgOut = msgOut + '\n  Target Position  : ' + rd[3:]

            self.write('1TP')  # get current position
            rd = self.read()
            msgOut = msgOut + '\n  Current Position : ' + rd[3:]

            self.write('1VE')  # get controller revision information
            rd = self.read()
            self.FW = rd[15:]
            msgOut = msgOut + '\n  Revision vs.     : ' + rd[4:]

            # Check controller state:_________________________________
            self.write('1TS')  # get positioner error and controller state
            rd = self.read()
            msgOut = msgOut + '\n  Positioner Error : ' + rd[3:7]
            msgOut = msgOut + '\n  Current State    : ' + rd[7:]

            # Create flag to mark if device was in 'ready' state
            flg = False
            # Ensure the device is in state where ZT is allowed
            if (self.isMoving(rd) or self.isHoming(rd)):  # rd[-2:] in ('28', '1E'):
                # Device is moving or Homing
                getLogger(__name__).info('\nWARNING::: Device Moving or Homing:\n' +
                      '  printDev() cannot display all values\n')

                msgOut = '  Device Type      : ' + self.TY + '' \
                                                             '\n  Serial Number    : ' + self.SN + '' \
                                                                                                   '\n  Software Min Lim.: ' + str(
                    self.MNPS) + '' \
                                 '\n  Software Max Lim.: ' + str(self.MXPS) + '\n' + msgOut
                getLogger(__name__).info(msgOut)
                return
            elif self.isReady(rd):
                # Device in 'ready' state; change to disable for ZT command
                getLogger(__name__).info('\nWARNING::: Device in \'Ready\' State:\n' +
                      '  Temporarily changed to Disable for ZT read\n')
                self.write('1MM0')  # Enter disable state
                time.sleep(self.DELAY)
                flg = True
            elif not self.isReferenced():
                # Device in 'not referenced' state; warn user about inaccuracies
                getLogger(__name__).info('\nWARNING::: Device \'Not Referenced\':\n' +
                      '  Current Position is unreliable; call home() to correct\n')

                # Request remaining device information
            self.write('1ZT')  # Get all controller parameters
            rd = self.readAll()

            # Process and format ZT read result:______________________
            self.SN = rd[1][18:25].decode()  # Update SN
            self.TY = rd[1][3:14].decode()  # Update TY
            self.MNPS = float(rd[10][3:].strip().decode())  # Update MNPS
            self.MXPS = float(rd[11][3:].strip().decode())  # Update MXPS
            msgOut = '  Device Type      : ' + self.TY + '' \
                                                         '\n  Serial Number    : ' + self.SN + '' \
                                                                                               '\n  Software Min Lim.: ' + str(
                self.MNPS) + '' \
                             '\n  Software Max Lim.: ' + str(self.MXPS) + '\n' + msgOut + '' \
                                                                                          '\n  Deadband         : ' + \
                     rd[5][3:].strip().decode() + '' \
                                                  '\n  Home Search Type : ' + rd[8][3:].strip().decode() + '' \
                                                                                                           '\n  Interpol. Factor : ' + \
                     rd[6][3:].strip().decode() + '' \
                                                  '\n  Integral Gain    : ' + rd[3][3:].strip().decode() + '' \
                                                                                                           '\n  Proportional Gain: ' + \
                     rd[4][3:].strip().decode() + '' \
                                                  '\n  L-P Filter  Freq.: ' + rd[2][3:].strip().decode() + '' \
                                                                                                           '\n  RS-485 Address   : ' + \
                     rd[9][3:].strip().decode() + '' \
                                                  '\n  Encoder Incr. Val: ' + rd[7][3:].strip().decode()
            getLogger(__name__).info(msgOut)

            # Return device to 'ready' state if needed
            if flg:
                self.write('1MM1')  # Leave disable state
                getLogger(__name__).info('\nWARNING::: Device returned to \'Ready\' State.\n')
        else:
            getLogger(__name__).info('\nWARNING::: Device is not open\n' +
                  '  returning last instance variables\n')
            msgOut = '  Device Type      : ' + self.TY + '' \
                                                         '\n  Serial Number    : ' + self.SN + '' \
                                                                                               '\n  Software Min Lim.: ' + str(
                self.MNPS) + '' \
                             '\n  Software Max Lim.: ' + str(self.MXPS)
            getLogger(__name__).info(msgOut)

    #:::::::::::::::::::::::STATE CHANGE FUNCTIONS:::::::::::::::::::::::::::::
    def home(self, isBlocking=False):
        """Homes the device
        isBlocking = True will block execution until homing completes
        reset()s then home()s when called in isReady() or isMoving() state
        Returns:
            if isBlocking is set True:
                boolean True if device reports it isReady() at end of home()
                boolean False if device does not report isReady at end of home()
            Nothing when isBlocking is set False
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        if self.isReady() or self.isMoving():
            # Reset device to allow home()
            self.reset()

        self.write('1OR')  # execute home search

        # Check for errors
        erFlg, erCd = self.isError()
        if erFlg:
            getLogger(__name__).info('DEV ERROR::: Device returned error:\n' +
                  '  ' + self.errorStr(erCd))
            return

        # Wait for move to complete when isBlocking is set
        if isBlocking:
            tmtItr = 0;  # Iteration counter for timeout
            while self.isHoming():
                if tmtItr > self.MVTIMEOUT:
                    # exit loop in case of timout
                    getLogger(__name__).info('ERROR::: home() timed out\n' +
                          '  Call isError()/errStr() to get device error.\n' +
                          '  Timeout is set to %f s' % (self.DELAY * self.MVTIMEOUT))
                    break
                time.sleep(self.DELAY)
                tmtItr += 1
            return self.isReady()

    def reset(self):
        """Reset the device
        Returns:
            boolean True if device reports it is in the Not Referenced State
            boolean False otherwise
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        self.write('1RS')  # execute home search
        time.sleep(10 * self.DELAY)  # reset takes time to execute

        # Check for errors
        erFlg, erCd = self.isError()
        if erFlg:
            getLogger(__name__).info('DEV ERROR::: Device returned error:\n' +
                  '  ' + self.errorStr(erCd))
            return

        return not self.isReferenced()

    def stop(self):
        """Stop all motion on the device
        If device is homing, it will return to Not Referenced State
        If device is performing a move, it will return to Ready State
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        self.write('1ST')  # execute home search

        # Check for errors
        erFlg, erCd = self.isError()
        if erFlg:
            getLogger(__name__).info('DEV ERROR::: Device returned error:\n' +
                  '  ' + self.errorStr(erCd))

    def enable(self):
        """Enables the device (set 'Ready' state)
        Returns boolean TRUE if device reports that it is now in READY state
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        self.write('1MM1')  # leave disable state

        # Check for errors
        erFlg, erCd = self.isError()
        if erFlg:
            getLogger(__name__).info('DEV ERROR::: Device returned error:\n' +
                  '  ' + self.errorStr(erCd))
            if erCd == 'H':
                getLogger(__name__).info('ERROR::: Device likely not homed:\n' +
                      '  solution: call home()')
            return False

        # Confirm that device is in 'Ready' state
        return self.isReady()

    def disable(self):
        """Disables the device (set 'Disable' state)
        Returns boolean TRUE if device reports that it is now in DISABLED state
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        self.write('1MM0')  # enter disable state

        # Check for errors
        erFlg, erCd = self.isError()
        if erFlg:
            getLogger(__name__).info('DEV ERROR::: Device returned error:\n' +
                  '  ' + self.errorStr(erCd))
            if erCd == 'H':
                getLogger(__name__).info('ERROR::: Device likely not homed:\n' +
                      '  solution: call home()')
            return False

        # Confirm that device is in 'Ready' state
        return self.isDisable()

        #:::::::::::::::::::::::STATE CHECK FUNCTIONS::::::::::::::::::::::::::::::

    def isReady(self, rd=None):
        """Checks that the device is in 'Ready' state
        *'rd' can be provided to avoid re-checking device status
            rd should be the stripped, decoded return of a TS command
        Returns:
            boolean TRUE if device is in 'Ready' state
            boolean False otherwise
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        # Request status if 'rd' not provided
        if rd == None:
            # Check if device is in 'Ready' state
            self.write('1TS')  # get positioner error and controller state
            rd = self.read()

        # Check for ready cases
        if rd[-2:] in ('32', '33', '34'):
            return True
        else:
            return False

    def isDisable(self, rd=None):
        """Checks that the device is in 'Disable' state
        *'rd' can be provided to avoid re-checking device status
            rd should be the stripped, decoded return of a TS command
        Returns:
            boolean TRUE if device is in 'Disable' state
            boolean False otherwise
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        # Request status if 'rd' not provided
        if rd == None:
            # Check if device is in 'Disable' state
            self.write('1TS')  # get positioner error and controller state
            rd = self.read()

        # Check for disable cases
        if rd[-2:] in ('3C', '3D'):
            return True
        else:
            return False

    def isReferenced(self, rd=None):
        """Checks that the device is in a 'Referenced' state
                *'a Referenced' state is any state OTHER THAN not referenced
        *'rd' can be provided to avoid re-checking device status
            rd should be the stripped, decoded return of a TS command
        Returns:
            boolean TRUE if device is in a 'Referenced' state
            boolean False otherwise
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        # Request status if 'rd' not provided
        if rd == None:
            # Check if device is in a 'Referenced' state
            self.write('1TS')  # get positioner error and controller state
            rd = self.read()

        # Check for not referenced cases
        if not ('0' in rd[-2:]):
            return True
        else:
            return False

    def isConfiguration(self, rd=None):
        """Checks that the device is in 'Configuration' state
        *'rd' can be provided to avoid re-checking device status
            rd should be the stripped, decoded return of a TS command
        Returns:
            boolean TRUE if device is in 'Configuration' state
            boolean False otherwise
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        # Request status if 'rd' not provided
        if rd == None:
            # Check if device is in 'Configuration' state
            self.write('1TS')  # get positioner error and controller state
            rd = self.read()

        # Check for configuration case
        if (rd[-2:] in '14'):
            return True
        else:
            return False

    def isHoming(self, rd=None):
        """Checks that the device is in 'Homing' state
        *'rd' can be provided to avoid re-checking device status
            rd should be the stripped, decoded return of a TS command
        Returns:
            boolean TRUE if device is in 'Homing' state
            boolean False otherwise
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        # Request status if 'rd' not provided
        if rd == None:
            # Check if device is in 'Homing' state
            self.write('1TS')  # get positioner error and controller state
            rd = self.read()

        # Check for homing case
        if (rd[-2:] in '1E'):
            return True
        else:
            return False

    def isMoving(self, rd=None):
        """Checks that the device is in 'Moving' state
        *'rd' can be provided to avoid re-checking device status
            rd should be the stripped, decoded return of a TS command
        Returns:
            boolean TRUE if device is in 'Moving' state
            boolean False otherwise
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        # Request status if 'rd' not provided
        if rd == None:
            # Check if device is in 'Moving' state
            self.write('1TS')  # get positioner error and controller state
            rd = self.read()

        # Check for moving case
        if (rd[-2:] in '28'):
            return True
        else:
            return False

    #:::::::::::::::::::::::MOVE FUNCTIONS:::::::::::::::::::::::::::::::::::::
    def moveAbs(self, newPOS, isBlocking=False):
        """Moves device to newPOS(mm)
        Waits for move to end when isBlocking is True
            returns final position (as a float) at end of move
        if isBlocking is ommitted or False, nothing is returned
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        self.write('1PA', newPOS)  # move absolute

        # Check for errors
        erFlg, erCd = self.isError()
        if erFlg:
            getLogger(__name__).info('DEV ERROR::: Device returned error:\n' +
                  '  ' + self.errorStr(erCd))
            if erCd == 'C':
                getLogger(__name__).info('ERROR::: Desired position likely beyond limits\n' +
                      '  Must be within [%0.4f and %0.4f]' % (self.MNPS, self.MXPS))
            return

        # Wait for move to complete when isBlocking is set
        if isBlocking:
            tmtItr = 0;  # Iteration counter for timeout
            while self.isMoving():
                if tmtItr > self.MVTIMEOUT:
                    # exit loop in case of timout
                    getLogger(__name__).info('ERROR::: moveAbs() timed out\n' +
                          '  Call isError()/errStr() to get device error.\n' +
                          '  Timeout is set to %f s' % (self.DELAY * self.MVTIMEOUT))
                    break
                time.sleep(self.DELAY)
                tmtItr += 1
            return self.reqPosAct()

    def moveRel(self, relMOV, isBlocking=False):
        """Moves device relMOV mm from current position
        Waits for move to end when isBlocking is True
            returns final position (as a float) at end of move
        if isBlocking is ommitted or False, nothing is returned
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        self.write('1PR', relMOV)  # move absolute

        # Check for errors
        erFlg, erCd = self.isError()
        if erFlg:
            getLogger(__name__).info('DEV ERROR::: Device returned error:\n' +
                  '  ' + self.errorStr(erCd))
            if erCd == 'C':
                getLogger(__name__).info('ERROR::: Desired position likely beyond limits\n' +
                      '  Must be within [%0.4f and %0.4f]' % (self.MNPS, self.MXPS))
            return

        # Wait for move to complete when isBlocking is set
        if isBlocking:
            tmtItr = 0;  # Iteration counter for timeout
            while self.isMoving():
                if tmtItr > self.MVTIMEOUT:
                    # exit loop in case of timout
                    getLogger(__name__).info('ERROR::: moveRel() timed out\n' +
                          '  Call isError()/errStr() to get device error.\n' +
                          '  Timeout is set to %f s' % (self.DELAY * self.MVTIMEOUT))
                    break
                time.sleep(self.DELAY)
                tmtItr += 1
            return self.reqPosAct()

    #:::::::::::::::::::::::ERROR CHECK FUNCTIONS::::::::::::::::::::::::::::::
    def isError(self):
        # Uses TE to reduce read time
        """Checks for device errors
        Returns:
            boolean True/False to mark if an error occurred
            str with error code returned by device
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        # Read error
        self.write('1TE')  # get command error string
        rd = self.read()

        # Check if error occurred and return accoridingly
        erCd = rd[3:]
        erFlg = False
        if erCd != '@':
            # error occurred
            erFlg = True
        return erFlg, erCd

    def errorStr(self, erCd):
        """Translates the error code ,'erCd', to a readable string
        Returns:
            str with text describing the error code
            *If device is not open(), the code itself is returned
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('WARNING::: Device must be open to translate string\n' +
                  '  solution: call open()')
            return erCd

        # Send error code to device for translation
        self.write('1TB', erCd)  # get command error string
        rd = self.read()

        if rd[3:4] != erCd:
            getLogger(__name__).info('ERRORR::: Device did not recognize provided error code')
            return 'Unrecognized Error Provided:  ' + erCd
        else:
            return rd[3:]

    #:::::::::::::::::::::::POSITION CHECK FUNCTIONS:::::::::::::::::::::::::::
    def reqPosSet(self):
        """Requests the target position
        Returns:
            Target position in mm as reported by device
            -9999 if error occurred
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        self.write('1TH')  # get target position
        rd = self.read()

        # Check for errors
        erFlg, erCd = self.isError()
        if erFlg:
            print('DEV ERROR::: Device returned error:\n' +
                  '  ' + self.errorStr(erCd))
            return -9999

        return float(rd[3:])

    def reqPosAct(self):
        """Requests the current position
        Returns:
            Actual position in mm as reported by device
            -9999 if error occurred
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        self.write('1TP')  # get current position
        rd = self.read()

        # Check for errors
        erFlg, erCd = self.isError()
        if erFlg:
            getLogger(__name__).info('DEV ERROR::: Device returned error:\n' +
                  '  ' + self.errorStr(erCd))
            return -9999

        return float(rd[3:])

    def reqInfo(self):
        """Reads device information and updates variables
        *These values usually don't change so accessing them from the
            instance variable is more efficient than repeating a call to
            reqInfo()
        *To simply display values, use devPrint()
        Returns:
            Serial number, device number, revision version
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        # Request and read device information
        self.write('1ID?')  # Get stage identifier
        rd = self.read()

        # Format and set SN and TY instance variables
        self.SN = rd[18:25]
        self.TY = rd[3:14]

        # Request and read revision information
        self.write('1VE')  # Get controller revision information
        rd = self.read()

        # Format and set FW instance variable
        self.FW = rd[15:]

    def reqLim(self):
        """Reads device software limits and updates variables
        *These values usually don't change so accessing them from the
            instance variable is more efficient than repeating a call to
            reqLim()
        *To simply display the values, use devPrint()
        Returns: Min Position, Max Position
        """
        # Check if port is open
        if not self.ser.isOpen():
            getLogger(__name__).info('ERROR::: Device must be open\n' +
                  '  solution: call open()')
            return

        # Request and read lower limit
        self.write('1SL?')  # Get negative software limit
        rd = self.read()

        # Format and set MNPS instance variable
        self.MNPS = float(rd[3:])

        # Request and read upper limit
        self.write('1SR?')  # Get positive software limit
        rd = self.read()

        # Format and set MXPS instance variable
        self.MXPS = float(rd[3:])

        return (self.MNPS, self.MXPS)
