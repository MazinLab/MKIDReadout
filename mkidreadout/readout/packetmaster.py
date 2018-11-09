from mkidcore.config import yaml, yaml_object
import tempfile
import psutil
import subprocess
import select
import threading
from mkidcore.corelog import getLogger

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
                 binary='./packetmaster', resume=False):
        self.ramdisk = ramdisk
        self.nroaches = nroaches
        self.binary_path = binary
        self.detector = detinfo
        self.nuller = nuller

        self.log = getLogger('packetmaster')

        packetmasters = [p.pid for p in psutil.process_iter(attrs=['pid','name'])
                         if 'packetmaster' in p.name]
        if len(packetmasters)>1:
            self.log.critical('Multiple instances of packetmaster running. Aborting.')
            raise RuntimeError('Multiple instances of packetmaster')

        self._process = psutil.Process(packetmasters[0]) if packetmasters else None

        if self._process is not None:
            if resume:
                self.log.warning('Reusing existing packetmaster instance, logging will not work')
            else:
                self.log.warning('Killing existing packetmaster instance.')
                self._process.kill()
                self._process = None

        self._pmmonitorthread = None

        self.start()

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

        self._pmmonitorthread = threading.Thread(self._monitor, name='Packetmaster IO Handler')
        self._pmmonitorthread.daemon = True
