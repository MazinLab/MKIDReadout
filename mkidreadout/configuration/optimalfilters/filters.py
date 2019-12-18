import numpy as np

import mkidreadout.configuration.optimalfilters.utils as utils


def matched(config, template, **kwargs):
    nfilter = kwargs.get("nfilter", config.nfilter)
    pass


def wiener(config, template, fft=False, **kwargs):
    psd = kwargs["psd"]
    nfilter = kwargs.get("nfilter", config.nfilter)

    if fft:
        # compute filter from the PSD
        template_fft = np.fft.rfft(template)
        filter_fft = np.conj(template_fft) / psd
        # roll to put the zero time index on the far right
        filter_ = np.roll(np.fft.irfft(filter_fft, len(template)), -1)[-nfilter:]
        filter_ /= (template[:nfilter] * filter_[::-1]).sum()
    else:
        # compute filter from covariance matrix
        template = template[:nfilter]
        covariance = utils.covariance_from_psd(psd, size=nfilter)
        filter_ = np.linalg.solve(covariance, template)[::-1]
        filter_ /= (template * filter_[::-1]).sum()
    return filter_


def baseline_insensitive(config, template, fft=False, **kwargs):
    psd = kwargs["psd"]
    nfilter = kwargs.get("nfilter", config.nfilter)
    # TODO: add template filtering

    if fft:
        filter_ = wiener(config, template, psd=psd, fft=True)
        filter_ -= filter_.mean()  # mean subtract to remove the f=0 component of its fft
        filter_ /= (template[:nfilter] * filter_[::-1]).sum()  # re-normalize
    else:
        # get trimmed template and corresponding covariance matrix
        template = template[:nfilter]
        covariance = utils.covariance_from_psd(psd, size=nfilter)
        vbar = np.vstack((template, np.ones_like(template))).T  # DC orthogonality vector
        # compute the filter from the covariance matrix
        filter_2d = np.linalg.solve(covariance, vbar)
        norm = np.matmul(vbar.T, filter_2d)
        filter_ = np.matmul(np.linalg.solve(norm.T, filter_2d.T).T, np.array([1, 0]))[::-1]
        filter_ /= (template * filter_[::-1]).sum()
    return filter_  # reverse so that it works with a correlation not a convolution


def exponential_insensitive(config, template, **kwargs):
    pass


def baseline_exponential_insensitive(config, template, **kwargs):
    pass
