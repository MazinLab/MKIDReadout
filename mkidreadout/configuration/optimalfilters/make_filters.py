from __future__ import division
import os
import signal
import pickle
import logging
import argparse
import numpy as np
import multiprocessing as mp

import mkidcore.config
import mkidcore.objects  # must be imported for beam map to load from yaml

import mkidreadout.configuration.optimalfilters.utils as utils
from mkidreadout.configuration.optimalfilters.time_stream import TimeStream

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
        self.time_streams = np.array([TimeStream(file_name, config=self.cfg.filters,
                                                 fallback_template=self.fallback_template, name=index)
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
        # overload stream configurations
        for stream in self.time_streams:
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
            results = utils.map_async_progress(pool, utils.process_time_stream, self.time_streams, progress=progress)
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
                result = utils.process_time_stream(stream)
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


def initialize_worker():
    """Initialize multiprocessing.pool worker to ignore keyboard interrupts."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore keyboard interrupt in worker process


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
    parser.add_argument('-l', '--log', type=str, dest='level', default="INFO",
                        help='The logging level to display.')
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(level=args.level)

    # load the configuration file
    configuration = mkidcore.config.load(args.cfg_file)

    # run the code
    run(configuration, progress=args.progress, force=args.force, save_name=args.name)
