import os
import signal
import pickle
import logging
import argparse
import numpy as np
import scipy as sp
import multiprocessing as mp
from astropy.stats import mad_std

import mkidcore.config
import mkidcore.objects  # must be imported for beam map to load from yaml
from mkidcore.pixelflags import filters as flag_dict

import mkidreadout.configuration.optimalfilters.utils as utils

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
            The file names for the resonator phase snaps.
        save_name: string (optional)
            The name to use for saving the file. The prefix will be used for
            saving its output products.
    """
    def __init__(self, config, file_names, save_name=DEFAULT_SAVE_NAME):
        # input attributes
        self._cfg = config
        self.fallback_template = utils.load_fallback_template(self.cfg.filters)
        self.file_names = file_names
        self.save_name = save_name
        # computation attributes
        self.res_ids = np.array([utils.res_id_from_file_name(file_name) for file_name in file_names])
        self.resonators = np.array([Resonator(self.cfg.filters, file_name,
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
        # change each resonator config and delete data that become invalid with the new settings
        for resonator in self.resonators:
            cfg = resonator.cfg
            # overload pulses, noise, template & filter if the pulse finding configuration changed
            if any([getattr(self.cfg.filters.pulses, key) != item for key, item in cfg.pulses.items()]):
                resonator.clear_results()
            # overload template & filter if the template configuration changed
            if any([getattr(self.cfg.filters.template, key) != item for key, item in cfg.template.items()]):
                resonator.clear_template()
                resotator.clear_filter()
            # overload noise, template & filter the results if the noise configuration changed
            if any([getattr(self.cfg.filters.noise, key) != item for key, item in cfg.noise.items()]):
                resonator.clear_noise()
                resonator.clear_template()
                resotator.clear_filter()
            # overload filter if the filter configuration changed
            if any([getattr(self.cfg.filters.filter, key) != item for key, item in cfg.filter.items()]):
                resotator.clear_filter()
            # overload resonator configurations
            resonator.cfg = self.cfg.filters
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
            results = utils.map_async_progress(pool, process_resonator, self.resonators, progress=progress)
            pool.close()
            try:
                # TODO: Python 2.7 bug: hangs on pool.join() with KeyboardInterrupt. The workaround is to use a really
                #  long timeout that hopefully never gets triggered. The 'better' code is:
                #  > pool.join()
                #  > resonators = results.get()
                resonators = results.get(1e5)
                self._add_resonators(resonators)
            except KeyboardInterrupt as error:
                log.error("Keyboard Interrupt encountered: retrieving computed filters before exiting")
                pool.terminate()
                pool.join()
                # TODO: not sure how long it will take to transfer real Resonator objects
                #  (timeout may need to be rethought)
                resonators = results.get(timeout=0.001)
                self._add_resonators(resonators)
                raise error  # propagate error to the main program
        else:
            pbar = utils.setup_progress(self.resonators) if progress and utils.HAS_PB else None
            for index, resonator in enumerate(self.resonators):
                result = process_resonator(resonator)
                self._add_resonators([result])
                if progress and utils.HAS_PB:
                    pbar.update(index)
        self._collect_data()
        self.clear_resonator_data()

    def clear_resonator_data(self):
        """Clear all unnecessary data from the Resonator sub-objects."""
        for resonator in self.resonators:
            resonator.clear_filter()

    def plot_summary(self):
        """Plot a summary of the filter computation."""
        pass

    def _add_resonators(self, resonators):
        for resonator in resonators:
            if resonator is not None:
                self.resonators[resonator.index] = resonator

    def _collect_data(self):
        filter_array = np.empty((self.res_ids.size, self.cfg.filters.filter.nfilter))
        for index, resonator in enumerate(self.resonators):
            filter_array[index, :] = resonator.result["filter"]
        self.filters.update({self.cfg.filters.filter.filter_type: filter_array})
        self.flags.update({self.cfg.filters.filter.filter_type: [r.result["flag"] for r in self.resonators]})


class Resonator(object):
    """
    Class for holding and manipulating a resonator's phase time-stream.

    Args:
        config: yaml config object
            The filter configuration object for the calculation loaded by
            mkidcore.config.load().
        file_name: string
            The file name containing the phase time-stream.
        fallback_template: numpy.ndarray (optional)
            A 1D numpy array of size config.template.ntemplate with the correct
            config.template.offset that will be used for the resonator template
            if it cannot be computed from the phase time-stream. If not
            supplied, the template is loaded in according to the config.
        index: integer (optional)
            An integer used to index the resonator objects. It is not used
            directly by this class.
    """
    def __init__(self, config, file_name, fallback_template=None, index=None):
        self.cfg = config
        self.file_name = file_name
        if fallback_template is None:
            self.fallback_template = utils.load_fallback_template(self.cfg)
        else:
            utils.check_template(config, fallback_template)
            self.fallback_template = fallback_template
        self.index = index

        self._time_stream = None

        self._init_results()

    def __getstate__(self):
        self.clear_file_properties()
        return self.__dict__

    @property
    def time_stream(self):
        """The phase time-stream of the resonator."""
        if self._time_stream is None:
            npz = np.load(self.file_name)
            self._time_stream = npz[npz.keys()[0]]
            # unwrap the time stream
            if self.cfg.unwrap:
                self._time_stream = np.unwrap(self._time_stream)
            # self._time_stream = np.zeros(int(60e6))  # TODO: remove
        return self._time_stream

    @time_stream.setter
    def time_stream(self, value):
        self._time_stream = value

    def clear_file_properties(self):
        """Free up memory by removing properties that can be reloaded from files."""
        self.time_stream = None

    def clear_results(self):
        """Delete computed results from the resonator."""
        self._init_results()
        self.time_stream = None  # technically computed since it is unwrapped
        log.debug("Resonator {}: results reset.")

    def clear_noise(self):
        """Delete computed noise from the resonator."""
        self.result["psd"] = None
        log.debug("Resonator {}: noise reset.".format(self.index))
        # if the filter was flagged reset the flag bitmask
        if self.result["flag"] & flag_dict["bad_noise"]:
            self.result["flag"] ^= flag_dict["bad_noise"]
            log.debug("Resonator {}: noise problem flag reset.".format(self.index))
        if self.result["flag"] & flag_dict["noise_computed"]:
            self.result["flag"] ^= flag_dict["noise_computed"]
            log.debug("Resonator {}: noise status flag reset.".format(self.index))

    def clear_template(self):
        """Delete computed template from the resonator."""
        self.result["template"] = None
        log.debug("Resonator {}: template reset.".format(self.index))
        # if the filter was flagged reset the flag bitmask
        if self.result["flag"] & flag_dict["bad_template"]:
            self.result["flag"] ^= flag_dict["bad_template"]
            log.debug("Resonator {}: template problem flag reset.".format(self.index))
        if self.result["flag"] & flag_dict["template_computed"]:
            self.result["flag"] ^= flag_dict["template_computed"]
            log.debug("Resonator {}: template status flag reset.".format(self.index))

    def clear_filter(self):
        """Delete computed filter from the resonator."""
        self.result["filter"] = None
        log.debug("Resonator {}: filter reset.".format(self.index))
        # if the filter was flagged reset the flag bitmask
        if self.result["flag"] & flag_dict["bad_filter"]:
            self.result["flag"] ^= flag_dict["bad_filter"]
            log.debug("Resonator {}: filter problem flag reset.".format(self.index))
        if self.result["flag"] & flag_dict["filter_computed"]:
            self.result["flag"] ^= flag_dict["filter_computed"]
            log.debug("Resonator {}: filter status flag reset.".format(self.index))

    def make_pulses(self):
        """Find the pulse index locations in the time stream."""
        cfg = self.cfg.pulses

        # filter the time stream
        fallback_filter = self.fallback_template - np.mean(self.fallback_template)  # Ignore DC component for the filter
        filtered_stream = sp.signal.convolve(self.time_stream, fallback_filter, mode='same')

        # find pulse indices
        sigma = mad_std(filtered_stream)
        characteristic_time = -np.trapz(self.fallback_template)  # ~decay time in units of dt for a perfect exponential
        indices, _ = sp.signal.find_peaks(-filtered_stream, height=cfg.threshold * sigma, distance=characteristic_time)
        self.result["pulses"] = indices.copy()
        self.result["mask"] = np.ones_like(self.result["pulses"], dtype=bool)

        # mask piled up pulses
        indices = np.insert(np.append(indices, filtered_stream.size), 0, 0)  # assume pulses are at the ends
        diff = np.diff(indices)
        bad_previous = (diff < cfg.separation)[:-1]  # far from previous previous pulse (remove last)
        bad_next = (diff < cfg.ntemplate - cfg.offset)[1:]  # far from next pulse  (remove first)
        self.result["mask"][bad_next | bad_previous] = False

        # TODO: mask wrapped pulses?

        # set flags
        self.result['flag'] |= flag_dict['pulses_computed']
        if self.result["mask"].sum() < cfg.min_pulses:  # not enough good pulses to make a reliable template
            self.result['template'] = self.fallback_template
            self.result['flag'] |= flag_dict['bad_pulses'] | flag_dict['bad_template'] | flag_dict['template_completed']

    def make_noise(self):
        """Make the noise spectrum for the resonator."""
        if self.result['flag'] & flag_dict['noise_computed']:
            return
        self._flag_checks(pulses=True)
        cfg = self.cfg.noise

        self.result['psd'] = np.zeros(cfg.nwindow)
        self.result['flag'] |= flag_dict['noise_computed']

    def make_template(self):
        """Make the template for the photon pulse."""
        if self.result['flag'] & flag_dict['template_computed']:
            return
        self._flag_checks(pulses=True, noise=True)
        cfg = self.cfg.template

        # make a pulse array
        index_array = (self.result["pulses"][self.result["mask"]] +
                       np.arange(-cfg.offset, cfg.ntemplate - cfg.offset)[:, np.newaxis])
        pulses = self.time_stream[index_array]

        # compute a rough template
        weights = 1 / np.abs(pulses[:, cfg.offset]) / pulses.shape[0]
        template = np.sum(pulses * weights[:, np.newaxis], axis=0)
        template /= np.abs(template.min())  # correct for floating point errors

        # TODO: make filter and recompute? (don't use make_filter code)
        # TODO: fit template?

        # set flags and results
        tau = -np.trapz(template)
        if tau < cfg.min_tau or tau > cfg.max_tau:
            self.result['flag'] |= flag_dict['bad_template']
            self.result['template'] = self.fallback_template
        else:
            self.result['template'] = template
        self.result['flag'] |= flag_dict['template_computed']

    def make_filter(self):
        """Make the filter for the resonator."""
        if self.result['flag'] & flag_dict['filter_computed']:
            return
        self._flag_checks(pulses=True, noise=True, template=True)
        cfg = self.cfg.filter

        self.result['filter'] = np.zeros(cfg.nfilter)
        self.result['flag'] |= flag_dict['filter_computed']

    def _init_results(self):
        self.result = {"pulses": None, "mask": None, "template": None, "filter": None, "psd": None,
                       "flag": flag_dict["not_started"]}

    def _flag_checks(self, pulses=False, noise=False, template=False):
        if pulses:
            assert self.result['flag'] & flag_dict['pulses_computed'], "run self.make_pulses() first."
        if noise:
            assert self.result['flag'] & flag_dict['noise_computed'], "run self.make_noise() first."
        if template:
            assert self.result['flag'] & flag_dict['template_computed'], "run self.make_template first."


def initialize_worker():
    """Initialize multiprocessing.pool worker to ignore keyboard interrupts."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore keyboard interrupt in worker process


def process_resonator(resonator):
    """Process the resonator object and compute it's filter."""
    resonator.make_pulses()
    resonator.make_noise()
    resonator.make_template()
    resonator.make_filter()
    # print(resonator.index)
    from time import sleep
    sleep(.01)
    return resonator


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
