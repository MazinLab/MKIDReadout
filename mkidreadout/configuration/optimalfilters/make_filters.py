import os
import pickle
import logging
import argparse
import numpy as np
import mkidcore.config
import mkidcore.objects  # must be imported for beam map to load from yaml
import multiprocessing as mp
from functools import partial

import mkidreadout.configuration.optimalfilters.utils as utils

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Solution(object):
    def __init__(self, config, file_names, default_template, save_name):
        # input attributes
        self._cfg = config
        self.file_names = file_names
        self.default_template = default_template
        self.save_name = save_name
        # computation attributes
        self.res_ids = np.array([utils.res_id_from_file_name(file_name) for file_name in file_names])
        self.resonators = np.array([Resonator(self.cfg.filter, file_name, self.default_template, index)
                                    for index, file_name in enumerate(file_names)])
        # output products
        self.filters = {}
        self.flags = {}

    @property
    def cfg(self):
        return self._cfg

    @cfg.seterr
    def cfg(self, config):
        self._cfg = config
        for resonator in self.resonators:
            # overload resonator configurations
            resonator.cfg = self.cfg.filter
            # overload resonator filter if the configuration filter types don't match
            if self.cfg.filter.filter_type != resonator.cfg.filter_type:
                resonator.results["filter"] = None
        log.info("Configuration file updated")

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            solution = pickle.load(f)
        solution.save_name = os.path.basename(file_path)
        log.info("Filter solution loaded from {}".format(file_path))
        return solution

    def save(self, file_name=None):
        if file_name is None:
            file_name = self.save_name
        file_path = os.path.join(self.cfg.paths.out, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Filter solution saved to {}".format(file_path))

    def save_filters(self, file_name=None):
        if file_name is None:
            file_name = os.path.splitext(self.save_name)[0] + "_coefficients.txt"
        file_path = os.path.join(self.cfg.paths.out, file_name)
        np.savetxt(file_path, self.filters[self.cfg.filter.filter_type])
        log.info("Filter coefficients saved to {}".format(file_path))

    def process(self, ncpu=1, progress=True, force=False):
        if ncpu > 1:
            pool = mp.Pool(min(self.cfg.ncpu, mp.cpu_count()))
            results = utils.map_async_progress(pool, partial(process_resonator, force=force),
                                               self.resonators, progress=progress)
            error = None
            try:
                pool.join()
            except KeyboardInterrupt as error:
                log.error("Keyboard Interrupt encountered: retrieving computed filters before exiting")
                results.get(timeout=0.001)
            finally:
                self._add_results(results)
                pool.close()
                if error is not None:
                    raise error
        else:
            pbar = utils.setup_progress(self.resonators) if progress and utils.HAS_PB else None
            for index, resonator in enumerate(self.resonators):
                result = process_resonator(resonator)
                self._add_results([result])
                if progress and utils.HAS_PB:
                    pbar.update(index)
        self._add_filters()

    def plot_summary(self):
        pass

    def _add_results(self, results):
        for resonator in results:
            self.resonators[resonator.index] = resonator

    def _add_filters(self):
        self.filters.update({self.cfg.filter.filter_type: np.zeros((self.res_ids.size, self.cfg.filter.nfilter))})  # TODO: load in from resonators
        self.flags.update({self.cfg.filter.filter_type: [resonator.flag for resonator in self.resonators]})


class Resonator(object):
    def __init__(self, config, file_name, default_template, index):
        self.index = index
        self.file_name = file_name
        self.cfg = config
        self.default_template = default_template
        self._time_stream = None

        self.result = {"template": None, "filter": None, "psd": None, "flag": 0}

    def __getstate__(self):
        self.clean()
        return self.__dict__

    @property
    def time_stream(self):
        if self._time_stream is None:
            npz = np.load(self.file_name)
            self._time_stream = npz[npz.keys()[0]]
        return self._time_stream

    def clean(self):
        self._time_stream = None

    def compute_traces(self):
        """Turn the time series into an array of traces (N triggers x config.filter.ntemplate)."""
        pass

    def make_template(self, force=False):
        """Make the template for the photon pulse."""
        self.compute_traces()

    def compute_autocorrelation(self):
        """Use the time between pulses to compute the noise autocorrelation function."""

    def make_filter(self, force=False):
        """Make the filter for the resonator."""
        self.compute_autocorrelation()


def process_resonator(resonator, force=False):
    if force:  # reset flag if recomputing
        resonator.flag = 0
    resonator.make_template(force=force)
    resonator.make_filter(force=force)
    return resonator


def run(config, progress=False, force=False, save_name="filter_solution.p"):
    if force or not os.path.isfile(save_name):
        log.info("Creating new solution object")
        # get/make default template
        default_template = utils.load_default_template(config)

        # get file name list
        file_names = utils.get_file_list(config.paths.data)

        # set up solution file
        sol = Solution(config, default_template, file_names, save_name)
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
        sol.process(ncpu=ncpu, progress=progress, force=force)
    except KeyboardInterrupt as error:
        log.error("Keyboard Interrupt encountered: saving the partial solution before exiting")
        sol.save()
        raise error

    # save the filters
    sol.save_filters()

    # plot summary
    if config.filter.summary_plot:
        sol.plot_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter Computation Utility')
    parser.add_argument('cfg_file', type=str, help='The configuration file to use for the computation.')
    parser.add_argument('-p', '--progress', action='store_true', dest='progress', help='Enable the progress bar.')
    parser.add_argument('-f', '--force', action='store_true', dest='force',
                        help='Force the recomputation of all of the computation steps.')
    parser.add_argument('-n', '--name', type=str, dest='name',
                        help='The name of the saved solution. The default is used if a name is not supplied.')
    args = parser.parse_args()

    configuration = mkidcore.config.load(args.cfg_file)
    run(configuration, progress=args.progress, force=args.force, save_name=args.name)
