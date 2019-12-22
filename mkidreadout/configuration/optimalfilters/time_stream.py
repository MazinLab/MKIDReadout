from __future__ import division
import logging
import numpy as np
import scipy as sp
import pkg_resources as pkg
from astropy.stats import mad_std
from skimage.restoration import unwrap_phase

import mkidcore.config
import mkidcore.objects  # must be imported for beam map to load from yaml
from mkidcore.pixelflags import filters as flags
from mkidreadout.configuration.optimalfilters import utils
from mkidreadout.configuration.optimalfilters import filters as filter_functions

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class TimeStream(object):
    """
    Class for holding and manipulating a resonator's phase time-stream.

    Args:
        file_name: string
            The file name containing the phase time-stream.
        config: yaml config object (optional)
            The filter configuration object for the calculation loaded by
            mkidcore.config.load(). If not supplied, the default configuration
            will be used.
        fallback_template: numpy.ndarray (optional)
            A 1D numpy array of size config.pulses.ntemplate with the correct
            config.pulses.offset that will be used for the time stream template
            if it cannot be computed from the phase time-stream. If
            supplied, the template is not updated if the config changes.
        name: any (optional)
            An object used to identify the time stream. It is not used directly
            by this class except for when printing log messages.
    """
    yaml_tag = u'!ts'

    def __init__(self, file_name, config=None, fallback_template=None, name=None):
        if config is None:
            self._cfg = None
        else:
            self._cfg = config
        self.file_name = file_name
        if fallback_template is not None:
            utils.check_template(self.cfg.pulses, fallback_template)
            self._fallback_template = fallback_template
            self._reload_fallback = False
        else:
            self._fallback_template = None
            self._reload_fallback = True

        self.name = name

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

    @property
    def cfg(self):
        """
        The filter computation configuration. Resetting it will clear parts of
        the filter result that are not consistent with the new settings.
        """
        if self._cfg is None:
            self._cfg = mkidcore.config.load(pkg.resource_filename(__name__, 'filter.yml')).filters
        return self._cfg

    @cfg.setter
    def cfg(self, value):
        # overload pulses, noise, template & filter if the pulse finding configuration changed
        if any([getattr(self.cfg.pulses, key) != item for key, item in value.pulses.items()]):
            self.clear_results()
            if self._reload_fallback:  # if the fallback wasn't supplied on the init, reset it
                self._fallback_template = None
        # overload noise, template & filter the results if the noise configuration changed
        if any([getattr(self.cfg.noise, key) != item for key, item in value.noise.items()]):
            self.clear_noise()
            self.clear_template()
            self.clear_filter()
        # overload template & filter if the template configuration changed
        if any([getattr(self.cfg.template, key) != item for key, item in value.template.items()]):
            self.clear_template()
            self.clear_filter()
        # overload filter if the filter configuration changed
        if any([getattr(self.cfg.filter, key) != item for key, item in value.filter.items()]):
            self.clear_filter()
        # set the config
        self._cfg = value

    @property
    def fallback_template(self):
        if self._fallback_template is None:
            self._fallback_template = utils.load_fallback_template(self.cfg.pulses)
        return self._fallback_template

    @property
    def characteristic_time(self):
        """Time constant of the response in units of dt."""
        if self.result['template'] is not None:
            return -np.trapz(self.result['template'])
        else:
            return -np.trapz(self.fallback_template)

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_mapping(cls.yaml_tag, dict(file=node.file, name=node.name))

    @classmethod
    def from_yaml(cls, constructor, node):
        d = dict(constructor.construct_pairs(node))
        if 'wavelength' in d.keys():
            ts = cls(d['file'], name=d['wavelength'])
        elif 'name' in d.keys():
            ts = cls(d['file'], name=d['name'])
        else:
            ts = cls(d['file'])
        return ts

    def clear_file_properties(self):
        """Free up memory by removing properties that can be reloaded from files."""
        self.phase = None

    def clear_results(self):
        """Delete computed results from the time stream."""
        self._init_result()
        self.phase = None  # technically computed since it is unwrapped
        log.debug("Time stream {}: results reset.")

    def clear_noise(self):
        """Delete computed noise from the time stream."""
        self.result["psd"] = None
        log.debug("Time stream {}: noise reset.".format(self.name))
        # if the filter was flagged reset the flag bitmask
        if self.result["flag"] & flags["bad_noise"]:
            self.result["flag"] ^= flags["bad_noise"]
            log.debug("Time stream {}: noise problem flag reset.".format(self.name))
        if self.result["flag"] & flags["noise_computed"]:
            self.result["flag"] ^= flags["noise_computed"]
            log.debug("Time stream {}: noise status flag reset.".format(self.name))

    def clear_template(self):
        """Delete computed template from the time stream."""
        self.result["template"] = None
        log.debug("Time stream {}: template reset.".format(self.name))
        # if the filter was flagged reset the flag bitmask
        if self.result["flag"] & flags["bad_template"]:
            self.result["flag"] ^= flags["bad_template"]
            log.debug("Time stream {}: template problem flag reset.".format(self.name))
        if self.result["flag"] & flags["template_computed"]:
            self.result["flag"] ^= flags["template_computed"]
            log.debug("Time stream {}: template status flag reset.".format(self.name))

    def clear_filter(self):
        """Delete computed filter from the time stream."""
        self.result["filter"] = None
        log.debug("Time stream {}: filter reset.".format(self.name))
        # if the filter was flagged reset the flag bitmask
        if self.result["flag"] & flags["bad_filter"]:
            self.result["flag"] ^= flags["bad_filter"]
            log.debug("Resonator {}: filter problem flag reset.".format(self.name))
        if self.result["flag"] & flags["filter_computed"]:
            self.result["flag"] ^= flags["filter_computed"]
            log.debug("Time stream {}: filter status flag reset.".format(self.name))

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

        # make a filter to use on the time stream (ignore DC component)
        if use_filter is True:
            filter_ = self.result["filter"]
        elif use_filter is not False:
            filter_ = use_filter
        else:
            filter_ = filter_functions.matched(self.fallback_template, nfilter=cfg.ntemplate, dc=True)

        # find pulse indices
        _, pulses = self.compute_responses(cfg.threshold, filter_=filter_)

        # correct for correlation offset and save results
        pulses += cfg.offset - filter_.size // 2
        mask = np.ones_like(pulses, dtype=bool)

        # mask piled up pulses
        diff = np.diff(np.insert(np.append(pulses, self.phase.size + 1), 0, 0))  # assume pulses are at the ends
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
        return pulses, mask

    def make_noise(self, save=True):
        """
        Make the noise spectrum for the time stream.

        Args:
            save: boolean (optional)
                Save the result in the result attribute when done. If save is
                False, none of the flags or results will change.

        Returns:
            psd: numpy.ndarray
                The computed power spectral density.
        """
        if self.result['flag'] & flags['noise_computed'] and save:
            return
        self._flag_checks(pulses=True)
        cfg = self.cfg.noise
        pulse_cfg = self.cfg.pulses

        # add pulses to the ends so that the bounds are treated correctly
        pulses = np.insert(np.append(self.result["pulses"], self.phase.size + 1), 0, 0)

        # loop space between peaks and compute noise
        n = 0
        psd = np.zeros(int(cfg.nwindow / 2. + 1))
        for peak1, peak2 in zip(pulses[:-1], pulses[1:]):
            if n > cfg.max_noise:
                break  # no more noise is needed
            if peak2 - peak1 < cfg.isolation + pulse_cfg.offset + cfg.nwindow:
                continue  # not enough space between peaks
            data = self.phase[peak1 + cfg.isolation: peak2 - pulse_cfg.offset]
            psd += sp.signal.welch(data, fs=1. / self.cfg.dt, nperseg=cfg.nwindow, detrend="constant",
                                   return_onesided=True, scaling="density")[1]
            n += 1

        # finish the average, assume white noise if there was no data
        if n == 0:
            psd[:] = 1.
        else:
            psd /= n

        if save:
            # set flags and results
            self.result['psd'] = psd
            if n == 0:
                self.result['flag'] |= flags['bad_noise']
            self.result['flag'] |= flags['noise_computed']
        return psd

    def make_template(self, save=True, refilter=True, pulses=None, mask=None, phase=None):
        """
        Make the template for the photon pulse.

        Args:
            save: boolean (optional)
                Save the result in the result attribute when done. If save is
                False, none of the flags or results will change.
            refilter: boolean (optional)
                Recompute the pulse indices and remake the template with a
                filter computed from the data.
            pulses: numpy.ndarray
                A list of pulse indices corresponding to phase to use in
                computing the template. The default is to use the value in
                the result.
            mask: numpy.ndarray
                A boolean array that picks out good indices from pulses.
            phase: numpy.ndarray
                The phase data to use to compute the template. The default
                is to use the median subtracted phase time-stream.

        Returns:
            template: numpy.ndarray
                A template for the pulse shape.
        """
        if self.result['flag'] & flags['template_computed'] and save:
            return
        self._flag_checks(pulses=True, noise=True)
        cfg = self.cfg.template
        if pulses is None:
            pulses = self.result['pulses']
        if mask is None:
            mask = self.result['mask']
        if phase is None:
            phase = self.phase - np.median(self.phase)

        # compute the template
        template = self._average_pulses(phase, pulses, mask)

        # shift and normalize template
        template = self._shift_and_normalize(template)

        # TODO: fit template?

        # filter again to get the best possible pulse locations
        if self._good_template(template) and refilter:
            filter_ = filter_functions.dc_orthogonal(template, self.result['psd'], cutoff=cfg.cutoff)
            pulses, mask = self.make_pulses(save=False, use_filter=filter_)
            template = self.make_template(save=False, refilter=False, pulses=pulses, mask=mask, phase=phase)

        # save and return the template
        if save:
            if self._good_template(template):
                self.result['template'] = template
            else:
                self.result['flag'] |= flags['bad_template']
                self.result['template'] = self.fallback_template
            self.result['flag'] |= flags['template_computed']
        return template

    def make_filter(self, save=True, filter_type=None):
        """
        Make the filter for the time stream.

        Args:
            save: boolean (optional)
                Save the result in the result attribute when done. If save is
                False, none of the flags or results will change.
            filter_type: string (optional)
                Create a filter of this type. The default is to use the value
                in the configuration file. Valid filters are function names in
                filters.py.

        Returns:
            filter_: numpy.ndarray
                The computed filter.
        """
        if self.result['flag'] & flags['filter_computed'] and save:
            return
        self._flag_checks(pulses=True, noise=True, template=True)
        cfg = self.cfg.filter
        if filter_type is None:
            filter_type = cfg.filter_type

        # compute filter (some options may be ignored by the filter function)
        filter_ = getattr(filter_functions, filter_type)(
            self.result["template"],
            self.result["psd"],
            nfilter=cfg.nfilter,
            cutoff=self.cfg.template.cutoff
        )

        # save and return the filter
        if save:
            self.result['filter'] = filter_
            self.result['flag'] |= flags['filter_computed']
        return filter_

    def apply_filter(self, filter_=None, positive=False):
        if filter_ is None:
            filter_ = self.result["filter"]
        if positive:
            filter_ = -filter_  # "-" so that pulses are positive
        filtered_phase = sp.signal.convolve(self.phase, filter_, mode='same')
        return filtered_phase

    def compute_responses(self, threshold, filter_=None):
        """
        Computes the pulse responses in the time stream.

        Args:
            threshold: float
                Only pulses above threshold sigma in the filtered time stream
                are included.
            filter_: numpy.ndarray
                A filter to use to calculate the responses. If None, the filter
                from the result attribute is used.

        Returns:
            responses: numpy.ndarray
                The response values of each pulse found in the time stream.
            pulses: numpy.ndarray
                The indices of each pulse found in the time stream.
        """
        if threshold is None:
            threshold = self.cfg.pulses.threshold
        filtered_phase = self.apply_filter(filter_=filter_, positive=True)
        sigma = mad_std(filtered_phase)
        pulses, _ = sp.signal.find_peaks(filtered_phase, height=threshold * sigma,
                                         distance=self.characteristic_time)
        responses = -filtered_phase[pulses]
        return responses, pulses

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
        logic = (pulse_heights <= percentiles[1]) & (percentiles[0] <= pulse_heights)
        weights = np.zeros_like(pulse_heights)
        weights[logic] = pulse_heights[logic]

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
        return self.cfg.template.min_tau < tau < self.cfg.template.max_tau


mkidcore.config.yaml.register_class(TimeStream)
