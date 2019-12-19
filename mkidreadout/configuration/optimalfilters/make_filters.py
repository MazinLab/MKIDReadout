from __future__ import division
import os
import signal
import pickle
import logging
import argparse
import numpy as np
import scipy as sp
import scipy.signal
import multiprocessing as mp
from astropy.stats import mad_std
from skimage.restoration import unwrap_phase


import mkidcore.config
import mkidcore.objects  # must be imported for beam map to load from yaml
from mkidcore.pixelflags import filters as flags

import mkidreadout.configuration.optimalfilters.utils as utils
import mkidreadout.configuration.optimalfilters.filters as filter_functions


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

DEFAULT_SAVE_NAME = "filter_solution.p"


class Solution(object):
    """
    Solution class for the filter generation.

    Args:
        config: yaml config object
            The configuration object for the calculation loaded by
            mkidcore.config.load().
        file_names: list of strings
            The file names for the resonator time streams.
        save_name: string (optional)
            The name to use for saving the file. The prefix will be used for
            saving its output products.
    """
    def __init__(self, config, file_names, save_name=DEFAULT_SAVE_NAME):
        # input attributes
        self._cfg = config
        self.fallback_template = utils.load_fallback_template(self.cfg.filters.pulses)
        self.file_names = file_names
        self.save_name = save_name
        # computation attributes
        self.res_ids = np.array([utils.res_id_from_file_name(file_name) for file_name in file_names])
        self.time_streams = np.array([TimeStream(self.cfg.filters, file_name,
                                                 fallback_template=self.fallback_template, index=index)
                                      for index, file_name in enumerate(file_names)])
        # output products
        self.filters = {}
        self.flags = {}

    @property
    def cfg(self):
        """The configuration object."""
        return self._cfg

    @cfg.setter
    def cfg(self, config):
        self._cfg = config
        # change each stream config and delete data that become invalid with the new settings
        for stream in self.time_streams:
            cfg = stream.cfg
            # overload pulses, noise, template & filter if the pulse finding configuration changed
            if any([getattr(self.cfg.filters.pulses, key) != item for key, item in cfg.pulses.items()]):
                stream.clear_results()
            # overload noise, template & filter the results if the noise configuration changed
            if any([getattr(self.cfg.filters.noise, key) != item for key, item in cfg.noise.items()]):
                stream.clear_noise()
                stream.clear_template()
                resotator.clear_filter()
            # overload template & filter if the template configuration changed
            if any([getattr(self.cfg.filters.template, key) != item for key, item in cfg.template.items()]):
                stream.clear_template()
                resotator.clear_filter()
            # overload filter if the filter configuration changed
            if any([getattr(self.cfg.filters.filter, key) != item for key, item in cfg.filter.items()]):
                resotator.clear_filter()
            # overload stream configurations
            stream.cfg = self.cfg.filters
        log.info("Configuration file updated")

    @classmethod
    def load(cls, file_path):
        """Load in the solution object from a file."""
        with open(file_path, 'rb') as f:
            solution = pickle.load(f)
        solution.save_name = os.path.basename(file_path)
        log.info("Filter solution loaded from {}".format(file_path))
        return solution

    def save(self, file_name=None):
        """Save the solution object to a file."""
        if file_name is None:
            file_name = self.save_name
        file_path = os.path.join(self.cfg.paths.out, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Filter solution saved to {}".format(file_path))

    def save_filters(self, file_name=None):
        """Save the filters to a file readable by the firmware."""
        if file_name is None:
            file_name = os.path.splitext(self.save_name)[0] + "_coefficients.txt"
        file_path = os.path.join(self.cfg.paths.out, file_name)
        np.savetxt(file_path, self.filters[self.cfg.filters.filter.filter_type])
        log.info("Filter coefficients saved to {}".format(file_path))

    def process(self, ncpu=1, progress=True):
        """Process all of the files and compute the filters."""
        if ncpu > 1:
            pool = mp.Pool(min(self.cfg.ncpu, mp.cpu_count()), initializer=initialize_worker)
            results = utils.map_async_progress(pool, process_time_stream, self.time_streams, progress=progress)
            pool.close()
            try:
                # TODO: Python 2.7 bug: hangs on pool.join() with KeyboardInterrupt. The workaround is to use a really
                #  long timeout that hopefully never gets triggered. The 'better' code is:
                #  > pool.join()
                #  > time_streams = results.get()
                time_streams = results.get(1e5)
                self._add_streams(time_streams)
            except KeyboardInterrupt as error:
                log.error("Keyboard Interrupt encountered: retrieving computed filters before exiting")
                pool.terminate()
                pool.join()
                # TODO: not sure how long it will take to get() Resonator objects (don't use timeout?)
                time_streams = results.get(timeout=0.001)
                self._add_streams(time_streams)
                raise error  # propagate error to the main program
        else:
            pbar = utils.setup_progress(self.time_streams) if progress and utils.HAS_PB else None
            for index, stream in enumerate(self.time_streams):
                result = process_time_stream(stream)
                self._add_streams([result])
                if progress and utils.HAS_PB:
                    pbar.update(index)
        self._collect_data()
        self.clear_time_stream_data()

    def clear_time_stream_data(self):
        """Clear all unnecessary data from the Resonator sub-objects."""
        for stream in self.time_streams:
            stream.clear_filter()

    def plot_summary(self):
        """Plot a summary of the filter computation."""
        pass

    def _add_streams(self, time_streams):
        for stream in time_streams:
            if stream is not None:
                self.time_streams[stream.index] = stream

    def _collect_data(self):
        filter_array = np.empty((self.res_ids.size, self.cfg.filters.filter.nfilter))
        for index, stream in enumerate(self.time_streams):
            filter_array[index, :] = stream.result["filter"]
        self.filters.update({self.cfg.filters.filter.filter_type: filter_array})
        self.flags.update({self.cfg.filters.filter.filter_type: [ts.result["flag"] for ts in self.time_streams]})


class TimeStream(object):
    """
    Class for holding and manipulating a resonator's phase time-stream.

    Args:
        config: yaml config object
            The filter configuration object for the calculation loaded by
            mkidcore.config.load().
        file_name: string
            The file name containing the phase time-stream.
        fallback_template: numpy.ndarray (optional)
            A 1D numpy array of size config.pulses.ntemplate with the correct
            config.pulses.offset that will be used for the time stream template
            if it cannot be computed from the phase time-stream. If not
            supplied, the template is loaded in according to the config.
        index: integer (optional)
            An integer used to index the time stream objects. It is not used
            directly by this class.
    """
    def __init__(self, config, file_name, fallback_template=None, index=None):
        self.cfg = config
        self.file_name = file_name
        if fallback_template is None:
            self.fallback_template = utils.load_fallback_template(self.cfg.pulses)
        else:
            utils.check_template(self.cfg.pulses, fallback_template)
            self.fallback_template = fallback_template
        self.index = index

        self._phase = None

        self._init_result()

    def __getstate__(self):
        self.clear_file_properties()
        return self.__dict__

    @property
    def phase(self):
        """The phase time-stream of the resonator."""
        if self._phase is None:
            npz = np.load(self.file_name)
            self._phase = npz[npz.keys()[0]]
            # unwrap the phase
            if self.cfg.unwrap:
                self._phase = unwrap_phase(self._phase)  # much faster than np.unwrap

            # self._phase = np.zeros(int(60e6))  # TODO: remove
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

    def clear_file_properties(self):
        """Free up memory by removing properties that can be reloaded from files."""
        self.phase = None

    def clear_results(self):
        """Delete computed results from the time stream."""
        self._init_result()
        self.phase = None  # technically computed since it is unwrapped
        log.debug("Resonator {}: results reset.")

    def clear_noise(self):
        """Delete computed noise from the time stream."""
        self.result["psd"] = None
        log.debug("Resonator {}: noise reset.".format(self.index))
        # if the filter was flagged reset the flag bitmask
        if self.result["flag"] & flags["bad_noise"]:
            self.result["flag"] ^= flags["bad_noise"]
            log.debug("Resonator {}: noise problem flag reset.".format(self.index))
        if self.result["flag"] & flags["noise_computed"]:
            self.result["flag"] ^= flags["noise_computed"]
            log.debug("Resonator {}: noise status flag reset.".format(self.index))

    def clear_template(self):
        """Delete computed template from the time stream."""
        self.result["template"] = None
        log.debug("Resonator {}: template reset.".format(self.index))
        # if the filter was flagged reset the flag bitmask
        if self.result["flag"] & flags["bad_template"]:
            self.result["flag"] ^= flags["bad_template"]
            log.debug("Resonator {}: template problem flag reset.".format(self.index))
        if self.result["flag"] & flags["template_computed"]:
            self.result["flag"] ^= flags["template_computed"]
            log.debug("Resonator {}: template status flag reset.".format(self.index))

    def clear_filter(self):
        """Delete computed filter from the time stream."""
        self.result["filter"] = None
        log.debug("Resonator {}: filter reset.".format(self.index))
        # if the filter was flagged reset the flag bitmask
        if self.result["flag"] & flags["bad_filter"]:
            self.result["flag"] ^= flags["bad_filter"]
            log.debug("Resonator {}: filter problem flag reset.".format(self.index))
        if self.result["flag"] & flags["filter_computed"]:
            self.result["flag"] ^= flags["filter_computed"]
            log.debug("Resonator {}: filter status flag reset.".format(self.index))

    def make_pulses(self, save=True, use_filter=False):
        """
        Find the pulse index locations in the time stream.

        Args:
            save: boolean (optional)
                Save the result in the result attribute when done. If save is
                False, none of the flags or results will change.
            use_filter: boolean, numpy.ndarray (optional)
                Use a pre-computed filter to find the pulse indices. If True,
                the filter from the result attribute is used. If False, the
                fallback template is used to make a filter. Otherwise,
                use_filter is assumed to be a pre-computed filter and is used.
                Any precomputed filter should have the same offset and size as
                specified in the configuration to get the right peak indices.

        Returns:
            pulses: numpy.ndarray
                An array of indices corresponding to the pulse peak locations.
            mask: numpy.ndarray
                A boolean mask that filters out bad pulses from the pulse
                indices.
        """
        if self.result['flag'] & flags['pulses_computed'] and save:
            return
        cfg = self.cfg.pulses

        # filter the time stream (ignore DC component)
        if use_filter is True:
            filter_ = self.result["filter"]
        elif use_filter is not False:
            filter_ = use_filter
        else:
            filter_ = filter_functions.matched(self.fallback_template, nfilter=cfg.ntemplate, dc=True)
        filtered_phase = sp.signal.convolve(self.phase, -filter_, mode='same')  # "-" so that pulses are positive

        # find pulse indices
        sigma = mad_std(filtered_phase)
        characteristic_time = -np.trapz(self.fallback_template)  # ~decay time in units of dt for a perfect exponential
        pulses, _ = sp.signal.find_peaks(filtered_phase, height=cfg.threshold * sigma, distance=characteristic_time)
        # TODO: skimage.feature.peak_local_max may be faster?

        # correct for correlation offset and save results
        pulses += cfg.offset - filter_.size // 2
        mask = np.ones_like(pulses, dtype=bool)

        # mask piled up pulses
        diff = np.diff(np.insert(np.append(pulses, filtered_phase.size + 1), 0, 0))  # assume pulses are at the ends
        bad_previous = (diff < cfg.separation)[:-1]  # far from previous previous pulse (remove last)
        bad_next = (diff < cfg.ntemplate - cfg.offset)[1:]  # far from next pulse  (remove first)
        mask[bad_next | bad_previous] = False

        # TODO: mask wrapped pulses?

        # save and return the pulse indices and mask
        if save:
            self.result["pulses"] = pulses
            self.result["mask"] = mask

            # set flags
            self.result['flag'] |= flags['pulses_computed']
            if self.result["mask"].sum() < cfg.min_pulses:  # not enough good pulses to make a reliable template
                self.result['template'] = self.fallback_template
                self.result['flag'] |= flags['bad_pulses'] | flags['bad_template'] | flags['template_computed']
            return self.result['pulses'], self.result['mask']
        else:
            return pulses, mask

    def make_noise(self):
        """Make the noise spectrum for the time stream."""
        if self.result['flag'] & flags['noise_computed']:
            return
        self._flag_checks(pulses=True)
        cfg = self.cfg.noise
        pulse_cfg = self.cfg.pulses

        # add pulses to the ends so that the bounds are treated correctly
        pulses = np.insert(np.append(self.result["pulses"], self.phase.size + 1), 0, 0)

        # loop space between peaks and compute noise
        n = 0
        self.result['psd'] = np.zeros(int(cfg.nwindow / 2. + 1))
        for peak1, peak2 in zip(pulses[:-1], pulses[1:]):
            if n > cfg.max_noise:
                break  # no more noise is needed
            if peak2 - peak1 < cfg.isolation + pulse_cfg.offset + cfg.nwindow:
                continue  # not enough space between peaks
            data = self.phase[peak1 + cfg.isolation: peak2 - pulse_cfg.offset]
            self.result['psd'] += sp.signal.welch(data, fs=1. / self.cfg.dt, nperseg=cfg.nwindow, detrend="constant",
                                                  return_onesided=True, scaling="density")[1]
            n += 1

        # set flags and results
        if n == 0:
            self.result['flag'] |= flags['bad_noise']
            self.result['psd'][:] = 1.  # set to white noise
        else:
            self.result['psd'] /= n  # finish the average
        self.result['flag'] |= flags['noise_computed']

    def make_template(self):
        """Make the template for the photon pulse."""
        if self.result['flag'] & flags['template_computed']:
            return
        self._flag_checks(pulses=True, noise=True)
        cfg = self.cfg.template

        # compute the template
        phase = self.phase - np.median(self.phase)
        template = self._average_pulses(phase, self.result['pulses'], self.result['mask'])

        # shift and normalize template
        template = self._shift_and_normalize(template)

        # filter again to get the best possible pulse locations
        if self._good_template(template):
            filter_ = filter_functions.dc_orthogonal(template, self.result['psd'], cutoff=cfg.cutoff)
            pulses, mask = self.make_pulses(save=False, use_filter=filter_)
            template = self._average_pulses(phase, pulses, mask)
            template = self._shift_and_normalize(template)

        # TODO: fit template?

        # set flags and results
        if self._good_template(template):
            self.result['template'] = template
        else:
            self.result['flag'] |= flags['bad_template']
            self.result['template'] = self.fallback_template
        self.result['flag'] |= flags['template_computed']

    def make_filter(self):
        """Make the filter for the time stream."""
        if self.result['flag'] & flags['filter_computed']:
            return
        self._flag_checks(pulses=True, noise=True, template=True)
        cfg = self.cfg.filter

        # compute filter (some options may be ignored by the filter function)
        filter_ = getattr(filter_functions, cfg.filter_type)(
            self.result["template"],
            self.result["psd"],
            nfilter=cfg.nfilter,
            cutoff=self.cfg.template.cutoff
        )

        self.result['filter'] = filter_
        self.result['flag'] |= flags['filter_computed']

    def _init_result(self):
        self.result = {"pulses": None, "mask": None, "template": None, "filter": None, "psd": None,
                       "flag": flags["not_started"]}

    def _flag_checks(self, pulses=False, noise=False, template=False):
        if pulses:
            assert self.result['flag'] & flags['pulses_computed'], "run self.make_pulses() first."
        if noise:
            assert self.result['flag'] & flags['noise_computed'], "run self.make_noise() first."
        if template:
            assert self.result['flag'] & flags['template_computed'], "run self.make_template first."

    def _average_pulses(self, phase, indices, mask):
        # get parameters
        offset = self.cfg.pulses.offset
        ntemplate = self.cfg.pulses.ntemplate
        percent = self.cfg.template.percent

        # make a pulse array
        index_array = (indices[mask] + np.arange(-offset, ntemplate - offset)[:, np.newaxis]).T
        pulses = phase[index_array]

        # weight the pulses by pulse height and remove those that are outside the middle percent of the data
        pulse_heights = np.abs(np.min(pulses, axis=1))
        percentiles = np.percentile(pulse_heights, [(100. - percent) / 2., (100. + percent) / 2.])
        logic = (pulse_heights != 0) & (pulse_heights <= percentiles[1]) & (percentiles[0] <= pulse_heights)
        weights = np.zeros_like(pulse_heights)
        weights[logic] = 1. / pulse_heights[logic]

        # compute the template
        template = np.sum(pulses * weights[:, np.newaxis], axis=0)
        return template

    def _shift_and_normalize(self, template):
        template = template.copy()
        if template.min() != 0:  # all weights could be zero
            template /= np.abs(template.min())  # correct over all template height

        # shift template (max may not be exactly at offset due to filtering and imperfect default template)
        start = 10 + np.argmin(template) - self.cfg.pulses.offset
        stop = start + self.cfg.pulses.ntemplate
        template = np.pad(template, 10, mode='wrap')[start:stop]  # use wrap to not change the frequency content
        return template

    def _good_template(self, template):
        tau = -np.trapz(template)
        return  self.cfg.template.min_tau < tau < self.cfg.template.max_tau


def initialize_worker():
    """Initialize multiprocessing.pool worker to ignore keyboard interrupts."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore keyboard interrupt in worker process


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


def run(config, progress=False, force=False, save_name=DEFAULT_SAVE_NAME):
    """
    Run the main logic for the filter generation.

    Args:
        config: yaml config object
            The configuration object for the calculation loaded by
            mkidcore.config.load().
        progress: boolean (optional)
            If progressbar is installed and progress=True, a progress bar will
            be displayed showing the progress of the computation.
        force: boolean (optional)
            If force is True, a new solution object will be made. If False and
            'save_name' is a real file, the solution from 'save_name' will be
            loaded in and the computation will be continued from where it left
            off.
        save_name: string (optional)
            If provided, the solution object will be saved with this name.
            Otherwise, a default name will be used. See 'force' for details
            on when the file 'save_name' already exists.
    """
    # set up the Solution object
    if force or not os.path.isfile(save_name):
        log.info("Creating new solution object")
        # get file name list
        file_names = utils.get_file_list(config.paths.data)
        # file_names = ["snap_112_resID10000_3212323-2323232.npz" for _ in range(2000)]  # TODO: remove
        # set up solution file
        sol = Solution(config, file_names, save_name=save_name)
    else:
        log.info("Loading solution object from {}".format(save_name))
        sol = Solution.load(save_name)
        sol.cfg = config

    # get the number of cores to use
    try:
        ncpu = max(1, int(min(config.ncpu, mp.cpu_count())))
    except KeyError:
        ncpu = 1
    log.info("Using {} cores".format(ncpu))

    # make the filters
    try:
        if force or config.filters.filter.filter_type not in sol.filters.keys():
            sol.process(ncpu=ncpu, progress=progress)
            sol.save()
        else:
            log.info("Filter type '{}' has already been computed".format(config.filters.filter.filter_type))
    except KeyboardInterrupt:
        log.error("Keyboard Interrupt encountered: saving the partial solution before exiting")
        sol.save()
        return

    # save the filters
    sol.save_filters()

    # plot summary
    if config.filters.summary_plot:
        sol.plot_summary()


if __name__ == "__main__":
    # make sure the Solution is unpickleable if created from __main__
    from mkidreadout.configuration.optimalfilters.make_filters import Solution

    # parse the command line arguments
    parser = argparse.ArgumentParser(description='Filter Computation Utility')
    parser.add_argument('cfg_file', type=str, help='The configuration file to use for the computation.')
    parser.add_argument('-p', '--progress', action='store_true', dest='progress', help='Enable the progress bar.')
    parser.add_argument('-f', '--force', action='store_true', dest='force',
                        help='Force the recomputation of all of the computation steps.')
    parser.add_argument('-n', '--name', type=str, dest='name', default=DEFAULT_SAVE_NAME,
                        help='The name of the saved solution. The default is used if a name is not supplied.')
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(level="INFO")

    # load the configuration file
    configuration = mkidcore.config.load(args.cfg_file)

    # run the code
    run(configuration, progress=args.progress, force=args.force, save_name=args.name)
