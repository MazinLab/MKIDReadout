import re
import os
import glob
import logging
import numpy as np

try:
    import progressbar as pb
    HAS_PB = True
except ImportError:
    HAS_PB = False

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def load_default_template(config):
    template = np.loadtxt(config.filter.default_template)
    min_index = np.argmin(template)
    start = min_index - config.filter.offset
    stop = start + config.filter.npulse
    default_template = template[start:stop]
    if default_template.size != config.filter.npulse:  # slicing can return a different sized array
        raise ValueError("The default template is not the right size. The 'npulse' parameter may be too large.")
    return default_template[start:stop]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def natural_sort_key(s, _re=re.compile(r'(\d*\.\d+|\d+)')):
    return [float(text) if is_number(s) else text for text in _re.split(s)]


def res_id_from_file_name(file_name):
    return file_name.split("_")[2][5:]


def get_file_list(directory):
    # get all npz files in the directory
    file_list = glob.glob(os.path.join(directory, "snap_*_resID*_*-*.npz"))
    file_list = file_list.sort(key=natural_sort_key, reverse=True)  # high numbers first

    _, indices = np.unique(["_".join(name.split("_")[:2]) for name in file_list], return_index=True)
    file_list = list(np.array(file_list)[indices])  # remove duplicate res IDs with older time stamps
    return file_list


def map_async_progress(pool, func, iterable, callback=None, error_callback=None, progress=True):
    # setup progress bar
    progress = progress and HAS_PB
    if progress:
        ii = 0
        pbar = setup_progress(iterable)

        def update(*args):
            if callback is not None:
                callback(*args)
            global ii
            ii += 1
            pbar.update(ii)
    else:
        update = callback
    # add jobs to pool
    results = MapResult()
    for ii in iterable:
        results.append(pool.apply_async(func, ii, callback=update, error_callback=error_callback))
    return results


def map_progress(pool, *args, **kwargs):
    results = map_async_progress(pool, *args, **kwargs)
    pool.join()
    return results.get()


class MapResult(list):
    def get(self, *args, **kwargs):
        return [r.get(*args, **kwargs) for r in self]


def setup_progress(iterable):
    percentage = pb.Percentage()
    bar = pb.Bar()
    timer = pb.Timer()
    eta = pb.ETA()
    pbar = pb.ProgressBar(widgets=[percentage, bar, '  (', timer, ') ', eta, ' '], max_value=len(iterable)).start()
    return pbar
