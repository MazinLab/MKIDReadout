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

