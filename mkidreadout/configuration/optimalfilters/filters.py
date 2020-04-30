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
    Returns:
        filter_: numpy.ndarray
            The computed matched filter.
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
        nwindow: integer
            The window size used to compute the PSD.
        nfilter: integer (optional)
            The number of taps to use in the filter. The default is to use
            the template size.
        cutoff: float (optional)
            Set the filter response to zero above this frequency (in units of
            1 / dt). If False, no cutoff is applied. The default is False.
        fft: boolean (optional)
            If True, the filter will be computed in the Fourier domain, which
            could be faster for very long filters but will also introduce
            assumptions about periodicity of the signal. In this case, the
            psd must be the same size as the filter Fourier transform
            (nfilter // 2 + 1 points). The default is False, and the filter is
            computed in the time domain.
    Returns:
        filter_: numpy.ndarray
            The computed wiener filter.
    """
    # collect inputs
    template, psd, nwindow = args[0], args[1], args[2]
    nfilter = kwargs.get("nfilter", len(template))
    cutoff = kwargs.get("cutoff", False)
    fft = kwargs.get("fft", False)

    # need at least this long of a PSD
    if nwindow < nfilter:
        raise ValueError("The psd must be at least as long as the length of the FFT of the filter")

    # pad the template if it's shorter than the filter (linear ramp to avoid discontinuities)
    if nfilter > len(template):
        template = np.pad(template, (0, nfilter - len(template)), mode='linear_ramp')
    # truncate the template if it's longer than the filter
    elif nfilter < len(template):
        template = template[:nfilter]

    if fft:  # compute the filter in the frequency domain (introduces periodicity assumption, requires nwindow=nfilter)
        if nwindow != nfilter:
            raise ValueError("The psd must be exactly the length of the FFT of the filter to use the 'fft' method.")
        template_fft = np.fft.rfft(template)
        filter_ = np.fft.irfft(np.conj(template_fft) / psd, nwindow)  # must be same size else ValueError
        filter_ = np.roll(filter_, -1)  # roll to put the zero time index on the far right

    else:  # compute filter in the time domain (nfilter should be << nwindow for this method to be better than fft)
        covariance = utils.covariance_from_psd(psd, size=nfilter, window=nwindow)
        filter_ = np.linalg.solve(covariance, template)[::-1]

    # remove high frequency filter content
    if cutoff:
        filter_ = utils.filter_cutoff(filter_, cutoff)

    # normalize
    filter_ /= -np.matmul(template, filter_[::-1])  # "-" to give negative pulse heights after filtering

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
        nwindow: integer
            The window size used to compute the PSD.
        nfilter: integer (optional)
            The number of taps to use in the filter. The default is to use
            the template size.
        cutoff: float (optional)
            Set the filter response to zero above this frequency (in units of
            1 / dt). If False, no cutoff is applied. The default is False.
        fft: boolean (optional)
            If True, the filter will be computed in the Fourier domain, which
            could be faster for very long filters but will also introduce
            assumptions about periodicity of the signal. In this case, the
            psd must be the same size as the filter Fourier transform
            (nfilter // 2 + 1 points). The default is False, and the filter is
            computed in the time domain.
    Returns:
        filter_: numpy.ndarray
            The computed dc orthogonal  filter.
    """
    # collect inputs
    template, psd, nwindow = args[0], args[1], args[2]
    nfilter = kwargs.get("nfilter", len(template))
    cutoff = kwargs.get("cutoff", False)
    fft = kwargs.get("fft", False)

    if fft:  # compute the filter in the frequency domain (introduces periodicity assumption)
        filter_ = wiener(template, psd, nwindow, fft=True, nfilter=nfilter, cutoff=cutoff)
        filter_ -= filter_.mean()  # mean subtract to remove the f=0 component of its fft
        filter_ /= -np.matmul(template[:nfilter], filter_[::-1])  # "-" to give negative pulse heights after filtering

    else:  # compute filter in the time domain
        if nfilter > len(template):  # pad the template if it's shorter than the filter
            template = np.pad(template, (0, nfilter - len(template)), mode='linear_ramp')
        elif nfilter < len(template):  # truncate the template if it's longer than the filter
            template = template[:nfilter]
        covariance = utils.covariance_from_psd(psd, size=nfilter, window=nwindow)
        vbar = np.vstack((template, np.ones_like(template))).T  # DC orthogonality vector

        # compute the filter from the covariance matrix
        filter_2d = np.linalg.solve(covariance, vbar)

        # remove high frequency filter content
        if cutoff:
            filter_2d = utils.filter_cutoff(filter_2d, cutoff)

        # normalize and flip to work with convolution
        norm = np.matmul(vbar.T, filter_2d)
        filter_ = -np.linalg.solve(norm.T, filter_2d.T)[0, ::-1]  # "-" to give negative pulse heights after filtering

    return filter_


def exp_orthogonal(*args, **kwargs):
    raise NotImplementedError


def dc_exp_orthogonal(*args, **kwargs):
    raise NotImplementedError
