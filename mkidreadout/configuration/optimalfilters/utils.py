from __future__ import division
import re
import os
import glob
import logging
import numpy as np
import scipy as sp
import pkg_resources as pkg
import multiprocessing as mp

try:
    import progressbar as pb
    HAS_PB = True
except ImportError:
    HAS_PB = False

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def load_fallback_template(config):
    if config.fallback_template == "default":
        file_name = pkg.resource_filename(__name__, "template_15us.txt")
    else:
        file_name = config.fallback_template
    template = np.loadtxt(file_name)
    min_index = np.argmin(template)
    start = min_index - config.offset
    stop = start + config.ntemplate
    fallback_template = template[start:stop]
    check_template(config, fallback_template)
    return fallback_template


def check_template(config, template):
    if template.size != config.ntemplate:  # slicing can return a different sized array
        raise ValueError("The fallback template is not the right size. The 'ntemplate' parameter may be too large.")
    if np.argmin(template) != config.offset:
        raise ValueError("The fallback template peak is not at the right 'offset' index.")
    if np.min(template) != -1:
        raise ValueError("The fallback template must have a peak height of -1.")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def natural_sort_key(s, _re=re.compile(r'(\d*\.\d+|\d+)')):
    return [float(text) if is_number(s) else text for text in _re.split(s)]


def res_id_from_file_name(file_name):
    return int(file_name.split("_")[-2][5:])


def get_file_list(directory):
    # get all npz files in the directory
    file_list = glob.glob(os.path.join(directory, "*_resID*_*-*.npz"))
    # sort them with higher numbers first (newest times on top)
    file_list = file_list.sort(key=natural_sort_key, reverse=True)
    # find and remove older files with duplicate names (excluding the timestamp)
    _, indices = np.unique(["_".join(name.split("_")[:-1]) for name in file_list], return_index=True)
    file_list = list(np.array(file_list)[indices])
    return file_list


def map_async_progress(pool, func, iterable, callback=None, progress=True):
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
        results.append(pool.apply_async(func, (ii,), callback=update))
    return results


def map_progress(pool, *args, **kwargs):
    results = map_async_progress(pool, *args, **kwargs)
    pool.join()
    return results.get()


class MapResult(list):
    def get(self, *args, **kwargs):
        results = []
        for r in self:
            try:
                results.append(r.get(*args, **kwargs))
            except mp.TimeoutError:
                results.append(None)
        return results


def setup_progress(iterable):
    percentage = pb.Percentage()
    bar = pb.Bar()
    timer = pb.Timer()
    eta = pb.ETA()
    pbar = pb.ProgressBar(widgets=[percentage, bar, '  (', timer, ') ', eta, ' '], max_value=len(iterable)).start()
    return pbar


def covariance_from_psd(psd, size=None, dt=1.):
    autocovariance = np.real(np.fft.irfft(psd / 2.) / dt)  # divide by 2 for single sided PSD
    if size is not None:
        autocovariance = autocovariance[:size]
    covariance = sp.linalg.toeplitz(autocovariance)
    return covariance


def filter_cutoff(filter_, cutoff):
    """
    This function addresses a problem encountered when generating filters from
    oversampled data. The high frequency content of the filters can be
    artificially large due to poor estimation of the noise and template.

    In this case it is useful to remove frequencies above the cutoff from the
    filter. Only use this function when addressing the above issue and if the
    majority of the signal frequencies are < cutoff.

    It is best to avoid this procedure if possible since removing the high
    frequency content will artificially make the filter periodic, throws away
    some information in the signal, and may negatively influence some of the
    intended filter properties.
    """
    freq = np.fft.rfftfreq(filter_.shape[0], d=1)
    filter_fft = np.fft.rfft(filter_, axis=0)
    filter_fft[freq > cutoff, ...] = 0
    filter_ = np.fft.irfft(filter_fft, filter_.shape[0], axis=0)
    return filter_


def process_time_stream(time_stream):
    """Process the time stream object and compute it's filter."""
    time_stream.make_pulses()
    time_stream.make_noise()
    time_stream.make_template()
    time_stream.make_filter()
    # print(time stream.index)
    from time import sleep
    sleep(.01)
    return time_stream
