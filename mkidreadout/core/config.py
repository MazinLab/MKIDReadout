
from collections import OrderedDict
from pydoc import locate
import ruamel.yaml
import atexit
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser

RESERVED = ('._c', '._lc', '._t', '._a')

yaml = ruamel.yaml.YAML()

class ConfigManager(object):
    def __init__(self, file = 'settings.ini'):
        self._dirty = True
        self._dict = OrderedDict()
        self._file = file
        self._cfg = None

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def get(self, key, type=None):
        self.registered(key, error=True)
        type = self._dict.get(key+'._t', type)
        return locate(type)(self._dict[key]) if type is not None else self._dict[key]

    def registered(self, key, error=False):
        if key not in self._dict:
            if error:
                raise KeyError("Setting '{}' is not registered.")
            return False
        return True

    def keyisvalid(self, key, error=False):
        if key.endswith(RESERVED):
            if error:
                raise KeyError("Setting keys may not end with '{}'.".format(RESERVED))
            else:
                return False
        return True

    def update(self, key, value, comment=None, longcomment=None):
        self.registered(key, error=True)

        #TODO add checking against allowed

        if (self(key) != value or
                (comment is not None and self._dict[key+'._c'] != comment) or
                (longcomment is not None and self._dict[key+'._lc'] != longcomment)):

            self._dict[key]=value
            self._dirty = True
            if comment is not None:
                self._dict[key+'._c'] = comment
            if longcomment is not None:
                self._dict[key+'._lc'] = longcomment

    def register(self, key, initialvalue, type=None, allowed=None, comment='', longcomment='', update=False):
        """Registers a key, true iff the key was not already registered."""
        self.keyisvalid(key, error=True)
        ret = self.registered(key)
        if ret and not update:
            return ret
        self._dict[key] = initialvalue
        self._dirty = True
        self._dict[key + '._c'] = comment
        self._dict[key + '._lc'] = longcomment
        if type is not None:
            self._dict[key + '._t'] = type
        if allowed is not None:
            self._dict[key + '._a'] = allowed
        return ret

    def save(self, file=None):
        yaml.dump(self._cfg, file)

    def load(self, file=None):
        self._cfg = yaml.load(file)


def importoldconfig(cfgfile, namespace):
    """
    Load an old config such that settings are accessible by namespace.section.key
    Sections are coerced to lowercase and spaces are replaced with underscores.
    The default section keys are accessible as namespace.key. If called on successive
    configfiles any collisions will be handled silently by adopting the value
    of the most recently loaded config file. All settings are imported as strings.
    """
    cp = ConfigParser()
    cp.read(cfgfile)
    ns = namespace if namespace.endswith('.') else namespace + '.'
    for k, v in cp.items('DEFAULT'):
        config.register((ns+k.lower()).replace(' ','_'), v, update=True)
    for s in cp.sections():
        ns = namespace + s.lower() + '.'
        for k, v in cp.items(s):
            config.register((ns + k.lower()).replace(' ','_'), v, update=True)


config = ConfigManager()
atexit.register(config.save)





