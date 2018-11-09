from mkidcore.config import yaml, yaml_object

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
    def __init__(self, roaches, detinfo=(100,100), nuller=False, ramdisk=None, binary='./packetmaster'):
        self.ramdisk = ramdisk
        self.roaches = roaches
        self.binary_path = binary
        self.detector = detinfo
        self.nuller = nuller
        self.log = getLogger('packetmaster')
        self._process = None

        self._pmmonitorthread = Thread()

    @property
    def is_running(self):
        return self._process is not None and self._process.is_running()

    def _handle_pmpipe(self):
        if not self.is_running:
            return
        try:
            _, stdo,stde = self._process.communicate()
            self.log.error(stde)
            self.log.info(stdo)
        except PIPEERRORS:
            self.log.error('IPC Pipe Error', exc_info=True)

    def start(self):
        if self.is_running:
            return

        self.log.info('Starting packetmaster...')

        self._pmmonitor.start()

        with tempfile as cfg():
            cfg.write(self.ramdisk + '\n')
            cfg.write('{} {}\n'.format(self.detector[1], self.detector[0])) #TODO is this column row order correct
            cfg.write(str(self.nuller) + '\n')
            cfg.write(str(len(self.roaches)))

        # command = "sudo nice -n -10 %s >> %s"%(packetMaster_path, packetMasterLog_path)
        psutil.Process(self.binary_path,args=tempfile.name, nice=-10)
        # QtCore.QTimer.singleShot(50,partial(subprocess.Popen,command,shell=True))

