from __future__ import division
import numpy as np

import mkidreadout.configuration.optimalfilters.utils as utils


__all__ = ["matched", "wiener", "dc_orthogonal", "exp_orthogonal", "dc_exp_orthogonal"]


def matched(*args, **kwargs):
    """
    Create a filter matched to a template.

    Args:
        template:  numpy.ndarray
            The template with which to construct the filter.
        nfilter: integer (optional)
            The number of taps to use in the filter. The default is to use
            template.size.
        dc: boolean (optional)
            If True, the mean of the template is subtracted to make the
            template insensitive to a DC baseline. The default is True.
    """
    # collect inputs
    template = args[0]
    nfilter = kwargs.get("nfilter", template.size)
    dc = kwargs.get("dc", True)

    # compute filter
    filter_ = template[:nfilter][::-1].copy()
    if dc:
        filter_ -= filter_.mean()
    filter_ /= -np.matmul(template[:nfilter], filter_[::-1])  # "-" to give negative pulse heights after filtering

    return filter_


def wiener(*args, **kwargs):
    """
    Create a filter that minimizes the chi squared statistic when aligned
    with a photon pulse.

    Args:
        template:  numpy.ndarray
            The template with which to construct the filter.
        psd:  numpy.ndarray
            The power spectral density of the noise.
        nfilter: integer (optional)
            The number of taps to use in the filter. The default is to use
            2 * psd.size // 3.
        cutoff: float (optional)
            Set the filter response to zero above this frequency (in units of
            1 / dt). If False, no cutoff is applied. The default is False.
        fft: boolean (optional)
            If True, the filter will be computed in the Fourier domain, which
            could be faster for very long filters but will also introduce
            assumptions about periodicity of the signal. In this case, the
            template and the psd must be the same size. The default is False,
            and the filter is computed in the time domain.
    """
    # collect inputs
    template, psd = args[0], args[1]
    ntemplate = min(2 * psd.size // 3, len(template))
    nfilter = kwargs.get("nfilter", ntemplate)
    cutoff = kwargs.get("cutoff", False)
    fft = kwargs.get("fft", False)

    if fft:  # compute the filter in the frequency domain (introduces periodicity assumption)
        template_fft = np.fft.rfft(template)
        filter_ = np.fft.irfft(np.conj(template_fft) / psd, len(template))  # must be same size else ValueError
        filter_ = np.roll(filter_, -1)  # roll to put the zero time index on the far right

    else:  # compute filter in the time domain
        # only use the first third of the covariance matrix since computing from the PSD assumes periodicity
        if ntemplate < nfilter:
            raise ValueError("nfilter must be less then the template length and smaller than 2 / 3 the psd size")
        covariance = utils.covariance_from_psd(psd, size=ntemplate)
        template = template[:ntemplate]
        filter_ = np.linalg.solve(covariance, template)[::-1]

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
    Create a filter that minimizes the chi squared statistic when aligned
    with a photon pulse, while also being insensitive to a drifting baseline.

    Args:
        template:  numpy.ndarray
            The template with which to construct the filter.
        psd:  numpy.ndarray
            The power spectral density of the noise.
        nfilter: integer (optional)
            The number of taps to use in the filter. The default is to use
            2 * psd.size // 3.
        cutoff: float (optional)
            Set the filter response to zero above this frequency (in units of
            1 / dt). If False, no cutoff is applied. The default is False.
        fft: boolean (optional)
            If True, the filter will be computed in the Fourier domain, which
            could be faster for very long filters but will also introduce
            assumptions about periodicity of the signal. In this case, the
            template and the psd must be the same size. The default is False,
            and the filter is computed in the time domain.
    """
    # collect inputs
    template, psd = args[0], args[1]
    ntemplate = min(2 * psd.size // 3, len(template))
    nfilter = kwargs.get("nfilter", ntemplate)
    cutoff = kwargs.get("cutoff", False)
    fft = kwargs.get("fft", False)

    if fft:  # compute the filter in the frequency domain (introduces periodicity assumption)
        filter_ = wiener(config, template, psd, fft=True, nfilter=nfilter, cutoff=cutoff)
        filter_ -= filter_.mean()  # mean subtract to remove the f=0 component of its fft
        filter_ /= -np.matmul(template[:nfilter], filter_[::-1])  # "-" to give negative pulse heights after filtering

    else:  # compute filter in the time domain
        # only use the first third of the covariance matrix since computing from the PSD assumes periodicity
        if ntemplate < nfilter:
            raise ValueError("nfilter must be less then the template length and smaller than 2 / 3 the psd size")
        covariance = utils.covariance_from_psd(psd, size=ntemplate)
        template = template[:ntemplate]
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
