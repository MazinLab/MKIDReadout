import re
from mkidcore.corelog import getLogger
import os

MEC_FEEDLINE_INFO = dict(num=10, width=14, length=146)
DARKNESS_FEEDLINE_INFO = dict(num=5, width=25, length=80)

DEFAULT_ARRAY_SIZES = {'mec': (140, 146), 'darkness': (80, 125)}


MEC_NUM_FL_MAP = {236: '1a', 237: '1b', #238: '5a', 239: '5b',
                  220: '6a', 221: '6b', 222: '7a', 223: '7b', 232: '8a',
                  233: '8b', 228: '9a', 229: '9b', 224: '10a', 225: '10b'}

#TODO FLs are arbitrary as instrument isn't installed
DARKNESS_NUM_FL_MAP = {112: '1a', 114: '1b', 115: '5a', 116: '5b',
                       117: '6a', 118: '6b', 119: '7a', 120: '7b', 121: '8a',
                       122: '8b'}

MEC_FL_NUM_MAP = {v: str(k) for k, v in MEC_NUM_FL_MAP.items()}

DARKNESS_FL_NUM_MAP = {v: str(k) for k, v in MEC_NUM_FL_MAP.items()}

ROACHES = {'mec': MEC_NUM_FL_MAP.keys(), 'darkness': DARKNESS_NUM_FL_MAP.keys(),
           'MEC': MEC_NUM_FL_MAP.keys(), 'DARKNESS': DARKNESS_NUM_FL_MAP.keys()}


def roachnum(fl, band, instrument='MEC'):
    if instrument.lower() == 'darkness':
        DARKNESS_FL_NUM_MAP['{}{}'.format(fl, band)]
    elif instrument.lower() == 'mec':
        return MEC_FL_NUM_MAP['{}{}'.format(fl, band)]


def guessFeedline(filename):
    # TODO generaize and find a home for this function
    try:
        flNum = int(re.search('fl\d', filename, re.IGNORECASE).group()[-1])
    except AttributeError:
        try:
            ip = int(os.path.splitext(filename)[0][-3:])
            flNum = int(MEC_NUM_FL_MAP[ip][0])
        except (KeyError, ValueError, IndexError):
            getLogger(__name__).warning('Could not guess feedline from filename {}.')
            raise ValueError('Unable to guess feedline')

    getLogger(__name__).debug('Guessing FL{} for filename {}'.format(flNum, os.path.basename(filename)))
    return flNum
