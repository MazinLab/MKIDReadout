import argparse
import tempfile
import numpy as np
import mkidcore.config
import mkidcore.objects  # must be imported for beam map to load from yaml
import multiprocessing as mp
from functools import partial

from utils import load_default_template, get_file_list, map_progress, res_id_from_file_name


class Solution(object):
    def __init__(self, file_name=None, results=None, config=None):
        if results is None and config is None:
            self.load(file_name)
        elif results is not None and config is not None:
            self.results = results
            self.cfg = config

    def load(self, file_name):
        pass

    def save(self, file_name):
        pass

    def save_filters(self, file_name):
        pass

    def plot_summary(self):
        pass


class Resonator(object):
    def __init__(self, file_name, config, default_template):
        self.file_name = file_name
        self.cfg = config
        self.default_template = default_template
        self.output = {"filter": np.empty(self.cfg.nfilter), "template": np.empty(self.cfg.nfilter),
                       "psd": np.empty(self.cfg.nwindow), "flag": 0, "res_id": res_id_from_file_name(file_name)}

    def compute_traces(self):
        """Turn the time series into an array of traces (N triggers x config.filter.npulse)."""
        pass

    def make_template(self):
        """Make the template for the photon pulse."""
        self.compute_traces()

    def compute_autocorrelation(self):
        """Use the time between pulses to compute the noise autocorrelation function."""

    def make_filter(self):
        """Make the filter for the resonator."""
        self.compute_autocorrelation()


def process_file(config, default_template, file_name):
    resonator = Resonator(file_name, config, default_template)
    resonator.make_template()
    resonator.make_filter()
    return resonator.output


def run(config, progress=False):
    # get/make default template
    default_template = load_default_template(config)

    # get file name list
    file_names = get_file_list(config.paths.data)

    # multiprocessing w/progress bar
    pool = mp.Pool(min(config.ncpu, mp.cpu_count()))
    results = map_progress(pool, partial(process_file, config, default_template), file_names, progress=progress)
    pool.close()

    # save the result
    sol = Solution(results=results, config=config)
    with tempfile.NamedTemporaryFile(prefix="filter_solution_", suffix=".npz", dir=config.paths.out, delete=False) as f:
        sol.save(f)
    with tempfile.NamedTemporaryFile(prefix="filter_", suffix=".npz", dir=config.paths.out, delete=False) as f:
        sol.save_filters(f)

    # plot summary
    if config.filter.summary_plot:
        sol.plot_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter Computation Utility')
    parser.add_argument('cfg_file', type=str, help='The configuration file')
    parser.add_argument('--progress', action='store_true', dest='progress', help='Enable the progress bar')
    args = parser.parse_args()

    configuration = mkidcore.config.load(args.cfg_file)
    run(configuration, progress=args.progress)
