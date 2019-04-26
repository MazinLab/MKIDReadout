import os
from shutil import copy2 as copy2

from pkg_resources import resource_filename

import mkidcore.config
from mkidcore.config import DEFAULT_BMAP_CFGFILES

DEFAULT_TEMPLAR_CFGFILE = resource_filename('mkidreadout', os.path.join('config', 'hightemplar.yml'))
DEFAULT_DASHBOARD_CFGFILE = resource_filename('mkidreadout', os.path.join('config', 'dashboard.yml'))
DEFAULT_ROACH_CFGFILE = resource_filename('mkidreadout', os.path.join('config', 'roach.yml'))
DEFAULT_INIT_CFGFILE = DEFAULT_ROACH_CFGFILE

load = mkidcore.config.load  # ensure import doesn't get optimized out by an IDE


default_log_dir = './logs'


def tagfile(f, tag='', nounderscore=False, droppath=False):
    if tag and not nounderscore:
        tag = '_'+tag
    f, ext = os.path.splitext(f)
    if droppath:
        f=os.path.basename(f)
    return '{}{}{}'.format(f, tag, ext)


def generate_default_configs(instrument='mec', dir='./', init=False, templar=False,
                             dashboard=False):
    if templar or dashboard or init:
        copy2(DEFAULT_ROACH_CFGFILE, os.path.join(dir, tagfile(DEFAULT_ROACH_CFGFILE, 'generated', droppath=True)))
    if templar:
        copy2(DEFAULT_TEMPLAR_CFGFILE, os.path.join(dir, tagfile(DEFAULT_TEMPLAR_CFGFILE, 'generated', droppath=True)))
    if dashboard:
        copy2(DEFAULT_DASHBOARD_CFGFILE,
              os.path.join(dir, tagfile(DEFAULT_DASHBOARD_CFGFILE, 'generated', droppath=True)))
        copy2(DEFAULT_BMAP_CFGFILES[instrument],
              os.path.join(dir, tagfile(DEFAULT_BMAP_CFGFILES[instrument], 'generated', droppath=True)))

tagstr = mkidcore.config.tagstr