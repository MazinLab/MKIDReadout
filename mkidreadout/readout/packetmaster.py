from mkidcore.config import yaml, yaml_object
import tempfile
import psutil
import subprocess
import select
import threading
import os
from mkidcore.corelog import getLogger

DEFAULT_CAPTURE_PORT = 50000  #This should be whatever is hardcoded in packetmaster -JB

@yaml_object(yaml)
class PacketMasterConfig(object):
    _template = ('{rdsk}\n'
                 '{nrow} {ncol}\n'
                 '{nuller:.0f}\n'
                 '{nroach}\n')

    def __init__(self, ramdisk='/mnt/ramdisk/', nrow=80, ncol=125, nuller=False, nroach=1):
        self.ramdisk = ramdisk
        self.nrows=nrow
        self.ncols=ncol
        self.nuller=nuller
        self.nroach=nroach


class Packetmaster(object):
    # TODO overall log configuration must configure for a 'packetmaster' log
    def __init__(self, nroaches, detinfo=(100,100), nuller=False, ramdisk=None,
                 binary='', resume=False, captureport=DEFAULT_CAPTURE_PORT, start=True):
        self.ramdisk = ramdisk
        self.nroaches = nroaches
        if os.path.isfile(binary):
            self.binary_path = binary
        else:
            self.binary_path = os.path.join(os.path.dirname(__file__), 'packetmaster', 'packetmaster')
        self.detector = detinfo
        self.captureport = captureport
        self.nuller = nuller
        self.log = getLogger(__name__)

        self.log.debug('Using "{}" binary'.format(self.binary_path))

        packetmasters = [p.pid for p in psutil.process_iter(attrs=['pid','name'])
                         if 'packetmaster' in p.name()]
        if len(packetmasters)>1:
            self.log.critical('Multiple instances of packetmaster running. Aborting.')
            raise RuntimeError('Multiple instances of packetmaster')

        self._process = psutil.Process(packetmasters[0]) if packetmasters else None

        if self._process is not None:
            if resume:
                try:
                    connections = [x for x in self._process.get_connections()
                                   if x.status == psutil.CONN_LISTEN]
                    self.captureport = connections[0].laddr.port
                except Exception:
                    self.log.debug('Unable to determine listening port: ', exc_info=True)

                self.log.warning('Reusing existing packetmaster instance, logging will not work')
            else:
                self.log.warning('Killing existing packetmaster instance.')
                self._process.kill()
                self._process = None

        self._pmmonitorthread = None

        if start:
            self._start()

    @property
    def is_running(self):
        #TODO note this returns true even if _process.status() == psutil.STATUS_ZOMBIE
        try:
            return self._process.is_running()
        except AttributeError:
            return False

    def _logline(self, source, data):
        if source == 'stdout':
            self.log.info(data)
        else:
            self.log.error(data)

    def _monitor(self):
        source = {'stdout': self._process.stdout, 'stderr': self._process.stderr}

        def doselect(timeout=1):
            readable, _, _ = select.select([self._process.stdout, self._process.stderr], [],
                                           [self._process.stdout, self._process.stderr], timeout)
            for r in readable:
                try:
                    self._logline(source[r], r.readline())
                except:
                    self.log.debug('Caught in monitor', exc_info=True)

        while True:
            if not self.is_running:
                break
            doselect()
        doselect(0)

    def _start(self):
        if self.is_running:
            return

        self.log.info('Starting packetmaster...')

        with tempfile.NamedTemporaryFile('w', suffix='.cfg', delete=False) as tfile:
            tfile.write(self.ramdisk + '\n')
            tfile.write('{} {}\n'.format(self.detector[1], self.detector[0])) #TODO is this column row order correct
            tfile.write(str(self.nuller) + '\n')
            tfile.write(str(self.nroaches))

        self._process = psutil.Popen((self.binary_path, tfile.name), stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     shell=False, cwd=None, env=None, creationflags=0)

        self._pmmonitorthread = threading.Thread(target=self._monitor, name='Packetmaster IO Handler')
        self._pmmonitorthread.daemon = True
        self._pmmonitorthread.start()

    def startobs(self, datadir):
        sfile = os.path.join(self.ramdisk, 'START_tmp')
        self.log.debug("Starting packet save. Start file loc: %s", sfile[:-4])
        with open(sfile, 'w') as f:
            f.write(datadir)
        os.rename(sfile, sfile[:-4])  # prevent race condition

    def stopobs(self):
        self.log.debug("Stopping packet save.")
        open(os.path.join(self.ramdisk, 'STOP'), 'w').close()

    def quit(self):
        open(os.path.join(self.ramdisk, 'QUIT'), 'w').close()   # tell packetmaster to end
