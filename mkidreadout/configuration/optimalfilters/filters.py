import numpy as np

import mkidreadout.configuration.optimalfilters.utils as utils


def matched(config, template, **kwargs):
    nfilter = kwargs.get("nfilter", config.nfilter)
    pass


def wiener(config, template, **kwargs):
    psd = kwargs["psd"]
    nfilter = kwargs.get("nfilter", config.nfilter)

    # compute filter from the PSD
    template_fft = np.fft.rfft(template)
    filter_fft = np.conj(template_fft) / psd
    filter_ = np.fft.irfft(filter_fft, len(template))[-nfilter:]

    # normalize filter
    norm = (template[:nfilter] * filter_[::-1]).sum()
    filter_ /= norm

    return filter_


def baseline_insensitive2(config, template, **kwargs):
    psd = kwargs["psd"]
    nfilter = kwargs.get("nfilter", config.nfilter)

    # compute filter from the PSD
    template_fft = np.fft.rfft(template)
    filter_fft = np.zeros_like(template_fft)
    filter_fft[1:] = np.conj(template_fft[1:]) / psd[1:]  # set f=0 fft bin to zero remove the DC component
    filter_ = np.fft.irfft(filter_fft, len(template))[-nfilter:]

    # normalize filter
    norm = (template[:nfilter] * filter_[::-1]).sum()
    filter_ /= norm

    return filter_


def baseline_insensitive(config, template, **kwargs):
    psd = kwargs["psd"]
    nfilter = kwargs.get("nfilter", config.nfilter)

    # get trimmed template and corresponding covariance matrix
    template = template[:nfilter]
    covariance = utils.covariance_from_psd(psd, size=nfilter)
    vbar = np.vstack((template, np.ones_like(template))).T

    # compute the filter from the covariance matrix
    filter_2d = np.linalg.solve(covariance, vbar)
    norm = np.matmul(vbar.T, filter_2d)
    filter_ = np.matmul(np.linalg.solve(norm.T, filter_2d.T).T, np.array([1, 0]))

    return filter_


def exponential_insensitive(config, template, **kwargs):
    pass


def baseline_exponential_insensitive(config, template, **kwargs):
    pass
