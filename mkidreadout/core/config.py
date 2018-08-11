import ruamel.yaml
from mkidreadout.core import caller_name
from mkidpipeline.core.corelog import getLogger, setup_logging
try:
    import ConfigParser as configparser
except ImportError:
    import configparser

RESERVED = ('._c', '._lc', '._t', '._a')

yaml = ruamel.yaml.YAML()

setup_logging()


@ruamel.yaml.yaml_object(yaml)
class ConfigDict(dict):
    yaml_tag = u'!configdict'

    def __init__(self, *args):
        """ If initialized with a list of tuples cannonization is not enforced on values"""
        if args:
            self.update([(cannonizekey(k), v) for k, v in args[0]])

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_mapping(cls.yaml_tag, dict(node))

    @classmethod
    def from_yaml(cls, loader, node):
        # loader = ruamel.yaml.constructor.RoundTripConstructor
        # node = MappingNode(tag=u'!MyDict', value=[(ScalarNode(tag=u'tag:yaml.org,2002:str', value=u'initgui'),....
        # cls = <class '__main__.MyDict'>
        d = loader.construct_pairs(node)  #WTH this one line took half a day to get right
        return cls(d)

    def __getattr__(self, key):
        k1, _, krest = key.partition('.')
        return self[k1][krest] if krest else self[k1]

    def __contains__(self, k):
        k1, _, krest = k.partition('.')
        if super(ConfigDict, self).__contains__(k1):
            if krest:
                return krest in self[k1]
            return True
        else:
            return False

    def registered(self, key, error=False):
        if key not in self:
            if error:
                raise KeyError("Setting '{}' is not registered.".format(key))
            return False
        return True

    def keyisvalid(self, key, error=False):
        if key.endswith(RESERVED):
            if error:
                raise KeyError("Setting keys may not end with '{}'.".format(RESERVED))
            else:
                return False
        return True

    def updateSetting(self, key, value, comment=None, longcomment=None):
        self.registered(key, error=True)
        self._update(key, value, comment=comment, longcomment=longcomment)

    def _update(self, key, value, comment=None, longcomment=None):
        k1, _, krest = key.partition('.')

        if krest:
            self[k1]._update(krest, value, comment=comment, longcomment=longcomment)
        else:
            # TODO add checking against allowed
            self[k1] = value
            if comment is not None:
                self._dict[k1+'._c'] = comment
            if longcomment is not None:
                self._dict[k1+'._lc'] = longcomment

    def _register(self, key, initialvalue, type=None, allowed=None, comment='', longcomment=''):
        k1, _, krest = key.partition('.')
        if krest:
            cd = self.get(k1, ConfigDict())
            cd._register(krest, initialvalue, type=type, allowed=allowed, comment=comment, longcomment=longcomment)
            self[k1] = cd
            # getLogger('MKIDConfig').debug('registering {}.{}={}'.format(k1,krest, initialvalue))
        else:
            # getLogger('MKIDConfig').debug('registering {}={}'.format(k1, initialvalue))
            self[k1] = initialvalue
            if comment:
                self[key + '._c'] = comment
            if longcomment:
                self[key + '._lc'] = longcomment
            if type is not None:
                self[key + '._t'] = type
            if allowed is not None:
                self[key + '._a'] = allowed
        return self

    def register(self, key, initialvalue, type=None, allowed=None, comment='', longcomment='', update=False):
        """Registers a key, true iff the key was registered. Does not update an existing key unless
        update is True."""
        self.keyisvalid(key, error=True)
        ret = not self.registered(key)
        if not ret and not update:
            return ret
        self._register(key, initialvalue, type=type, allowed=allowed, comment=comment, longcomment=longcomment)
        return ret

    def todict(self):
        ret = dict(self)
        for k,v in ret.items():
            if isinstance(v, ConfigDict):
                ret[k] = v.todict()
        return ret

    def save(self, file):
        yaml.dump(self, file)

    def registerfromconfigparser(self, cp, namespace=None):
        """loads all data in the config parser object, overwriting any that already exist"""
        if namespace is None:
            namespace = caller_name().lower()
            getLogger('MKIDConfig').debug('Assuming namespace "{}"'.format(namespace))
        namespace = namespace if namespace.endswith('.') else (namespace + '.' if namespace else '')
        for k, v in cp.items('DEFAULT'):
            self.register(cannonizekey(namespace+k), cannonizevalue(v), update=True)
        for s in cp.sections():
            ns = namespace + s + '.'
            for k, v in cp.items(s):
                self.register(cannonizekey(ns + k), cannonizevalue(v), update=True)

    def registerfromkvlist(self, kv, namespace=None):
        """loads all data in the keyvalue iterable, overwriting any that already exist"""
        if namespace is None:
            namespace = caller_name().lower()
            getLogger('MKIDConfig').debug('Assuming namespace "{}"'.format(namespace))
        namespace = namespace if namespace.endswith('.') else (namespace + '.' if namespace else '')
        for k, v in kv:
            self.register(cannonizekey(namespace + k), cannonizevalue(v), update=True)
        return self


def cannonizekey(k):
    """Enforce cannonicity of config keys lowercase, no spaces (replace with underscore)"""
    return k.strip().lower().replace(' ', '_')


def cannonizevalue(v):
    try:
        v=dequote(v)
    except:
        pass
    try:
        if '.' in v:
            return float(v)
    except:
        pass
    try:
        return int(v)
    except:
        pass
    return v


def dequote(v):
    """Change strings like "'foo'" to "foo"."""
    if (v[0] == v[-1]) and v.startswith(("'", '"')):
        return v[1:-1]
    else:
        return v


def importoldconfig(config, cfgfile, namespace=None):
    """
    Load an old config such that settings are accessible by namespace.section.key
    Sections are coerced to lowercase and spaces are replaced with underscores.
    The default section keys are accessible as namespace.key. If called on successive
    configfiles any collisions will be handled silently by adopting the value
    of the most recently loaded config file. All settings are imported as strings.
    """
    cp = configparser.ConfigParser()
    try:
        cp.read(cfgfile)
    except configparser.MissingSectionHeaderError:
        #  Some files aren't configparser dicts, pretend they have a DEFUALTS section only
        with open(cfgfile, 'r') as f:
            data = f.readlines()

        for l in (l for l in data if l and l[0]!='#'):
            k, _, v =l.partition('=')
            if not k.strip():
                continue
            cp.set('DEFAULT', k.strip(), v.strip())

    config.registerfromconfigparser(cp, namespace)


config = ConfigDict()

def load(file):
    with open(file,'r') as f:
        return yaml.load(f)

# #---------------------
#
# from glob import glob
# import os
# import StringIO
#
# cfiles = glob('/Users/one/ucsb/MKIDPipelineProject/data/*.cfg')
#
#
# for cf in cfiles:
#     importoldconfig(config, cf, os.path.basename(cf)[:-4])
#
# cp = configparser.ConfigParser(); cp.read(cfiles[-1])
# rs=[ConfigDict().registerfromkvlist(cp.items(rname),'') for rname in cp.sections() if 'Roach' in rname]
# config.register('templarconf.roaches', rs)
#
# out = StringIO.StringIO()
# yaml.dump(config, out)
#
# x = yaml.load(out.getvalue())
# # # #
# # # x['templarconf.roaches'][0].roachnum
