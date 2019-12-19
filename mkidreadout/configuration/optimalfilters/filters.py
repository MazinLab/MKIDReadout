from __future__ import division
import numpy as np

import mkidreadout.configuration.optimalfilters.utils as utils


def matched(*args, **kwargs):
    """
    Create a filter matched to a template.

    Args:
        config: yaml config object
            The configuration object for the calculation loaded by
            mkidcore.config.load().
        template:  numpy.ndarray
            The template with which to construct the filter.
        nfilter: integer (optional)
            The number of taps to use in the filter. The default is to use the
            value specified in config.
        dc: boolean (optional)
            If True, the mean of the template is subtracted to make the
            template insensitive to a DC baseline. The default is False.
    """
    # collect inputs
    config, template = args[0], args[1]
    nfilter = kwargs.get("nfilter", config.nfilter)
    dc = kwargs.get("dc", False)

    # compute filter
    filter_ = template[:nfilter][::-1].copy()
    if dc:
        filter_ -= filter_.mean()
    filter_ /= -np.matmul(template[:nfilter], filter_[::-1])  # "-" to give negative pulse heights after filtering

    return filter_


def wiener(*args, **kwargs):
    """
    Create a filter that minimizes the squared error between the template and
    the data.

    Args:
        config: yaml config object
            The configuration object for the calculation loaded by
            mkidcore.config.load().
        template:  numpy.ndarray
            The template with which to construct the filter.
        psd:  numpy.ndarray
            The power spectral density of the noise.
        nfilter: integer (optional)
            The number of taps to use in the filter. The default is to use the
            value specified in config.
        cutoff: float (optional)
            Set the filter response to zero above this frequency (in units of
            1 / dt). If False, no cutoff is applied. The default is to use the
            value specified in config.
        fft: boolean (optional)
            If True, the filter will be computed in the Fourier domain, which
            could be faster for very long filters but will also introduce
            assumptions about periodicity of the signal. The default is False,
            and the filter is computed in the time domain.
    """
    # collect inputs
    config, template, psd = args[0], args[1], args[2]
    nfilter = kwargs.get("nfilter", config.nfilter)
    cutoff = kwargs.get("cutoff", config.cutoff)
    fft = kwargs.get("fft", False)

    if fft:  # compute the filter in the frequency domain (introduces periodicity assumption)
        template_fft = np.fft.rfft(template)
        filter_ = np.fft.irfft(np.conj(template_fft) / psd, len(template))  # must be same size else ValueError
        filter_ = np.roll(filter_, -1)  # roll to put the zero time index on the far right

    else:  # compute filter in the time domain
        # only use the first third of the covariance matrix since computing from the PSD assumes periodicity
        if template.size // 3 < nfilter:
            raise ValueError("ntemplate must be at least 3x the size of nfilter")
        covariance = utils.covariance_from_psd(psd, size=template.size // 3)
        filter_ = np.linalg.solve(covariance, template[:template.size // 3])[::-1]

    # remove high frequency filter content
    if cutoff:
        filter_ = utils.filter_cutoff(filter_, cutoff)

    # clip to the right size
    filter_ = filter_[-nfilter:]

    # normalize
    filter_ /= -np.matmul(template[:nfilter], filter_[::-1])  # "-" to give negative pulse heights after filtering

    return filter_


def dc_orthogonal(*args, **kwargs):
    """
    Create a filter that minimizes the squared error between the template and
    the data, while also being insensitive to a drifting baseline.

    Args:
        config: yaml config object
            The configuration object for the calculation loaded by
            mkidcore.config.load().
        template:  numpy.ndarray
            The template with which to construct the filter.
        psd:  numpy.ndarray
            The power spectral density of the noise.
        nfilter: integer (optional)
            The number of taps to use in the filter. The default is to use the
            value specified in config.
        cutoff: float (optional)
            Set the filter response to zero above this frequency (in units of
            1 / dt). If False, no cutoff is applied. The default is to use the
            value specified in config.
        fft: boolean (optional)
            If True, the filter will be computed in the Fourier domain, which
            could be faster for very long filters but will also introduce
            assumptions about periodicity of the signal. The default is False,
            and the filter is computed in the time domain.
    """
    # collect inputs
    config, template, psd = args[0], args[1], args[2]
    nfilter = kwargs.get("nfilter", config.nfilter)
    cutoff = kwargs.get("cutoff", config.cutoff)
    fft = kwargs.get("fft", False)

    if fft:  # compute the filter in the frequency domain (introduces periodicity assumption)
        filter_ = wiener(config, template, psd, fft=True, nfilter=nfilter, cutoff=cutoff)
        filter_ -= filter_.mean()  # mean subtract to remove the f=0 component of its fft
        filter_ /= -np.matmul(template[:nfilter], filter_[::-1])  # "-" to give negative pulse heights after filtering

    else:  # compute filter in the time domain
        # only use the first third of the covariance matrix since computing from the PSD assumes periodicity
        if template.size // 3 < nfilter:
            raise ValueError("ntemplate must be at least 3x the size of nfilter")
        covariance = utils.covariance_from_psd(psd, size=template.size // 3)
        template = template[:template.size // 3]
        vbar = np.vstack((template, np.ones_like(template))).T  # DC orthogonality vector

        # compute the filter from the covariance matrix
        filter_2d = np.linalg.solve(covariance, vbar)

        # remove high frequency filter content
        if cutoff:
            filter_2d = utils.filter_cutoff(filter_2d, cutoff)

        # clip to the right size
        filter_2d = filter_2d[:nfilter, :]

        # normalize and flip to work with convolution
        norm = np.matmul(vbar[:nfilter, :].T, filter_2d)
        filter_ = -np.linalg.solve(norm.T, filter_2d.T)[0, ::-1]  # "-" to give negative pulse heights after filtering

    return filter_


def exp_orthogonal(*args, **kwargs):
    raise NotImplementedError


def dc_exp_orthogonal(*args, **kwargs):
    raise NotImplementedError
