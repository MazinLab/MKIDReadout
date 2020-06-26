from __future__ import division
import types
import numpy as np
import lmfit as lm
from scipy.signal import argrelmax

EPS = np.finfo(np.float64).eps

__all__ = ["exponential", "double_exponential", "triple_exponential"]


def _compute_t0(data, t=None, rise_time=None):
    peak = np.argmin(data)
    t0 = t[peak] if t is not None else float(peak)
    if rise_time is not None:
        t0 -= rise_time
    return t0


def _compute_rise_time(data, t=None):
    peak = np.argmin(data)
    try:
        start = argrelmax(data[:peak])[0][-1]  # nearest extrema
    except IndexError:
        start = 0  # no relative max before the peak
    rise_time = (t[peak] - t[start]) / 2. if t is not None else max(1., (peak - start) / 2.)
    return rise_time


def _compute_fall_time(data, t=None):
    return -np.trapz(data, x=t)


def exponential(t, a, t0, fall_time):
    p = np.zeros_like(t, dtype=np.float)
    arg1 = -(t[t >= t0] - t0) / max(EPS, fall_time)
    p[t >= t0] = a * np.exp(arg1)
    return p


def _exponential_guess(model, data, t=None, **kwargs):
    """Estimate initial model parameter values from data."""
    t0 = kwargs.get("t0", _compute_t0(data, t=t))
    fall_time = kwargs.get("fall_time", _compute_fall_time(data, t=t))

    params = model.make_params(a=-1., t0=t0, fall_time=fall_time)
    params["a"].set(max=0.)
    params["t0"].set(min=0. if t is None else np.min(t), max=float(len(data)) if t is None else np.max(t))
    params["fall_time"].set(min=0.)
    return params


def double_exponential(t, a, t0, rise_time, fall_time):
    p = np.zeros_like(t, dtype=np.float)
    arg0 = -(t[t >= t0] - t0) / max(EPS, rise_time)
    arg1 = -(t[t >= t0] - t0) / max(EPS, fall_time)
    p[t >= t0] = a * (1 - np.exp(arg0)) * np.exp(arg1)
    return p


def _double_exponential_guess(model, data, t=None, **kwargs):
    """Estimate initial model parameter values from data."""
    rise_time = kwargs.get("rise_time", _compute_rise_time(data, t=t))
    t0 = kwargs.get("t0", _compute_t0(data, t=t, rise_time=rise_time))
    fall_time = kwargs.get("fall_time", _compute_fall_time(data, t=t))

    params = model.make_params(a=-1., t0=t0, rise_time=rise_time, fall_time=fall_time)
    params["a"].set(max=0.)
    params["t0"].set(min=0. if t is None else np.min(t), max=float(len(data)) if t is None else np.max(t))
    params["rise_time"].set(min=0.)
    params["fall_time"].set(min=0.)
    return params


def triple_exponential(t, a, b, t0, rise_time, fall_time1, fall_time2):
    p = np.zeros_like(t, dtype=np.float)
    arg0 = -(t[t >= t0] - t0) / max(EPS, rise_time)
    arg1 = -(t[t >= t0] - t0) / max(EPS, fall_time1)
    arg2 = -(t[t >= t0] - t0) / max(EPS, fall_time2)
    p[t >= t0] = a * (1 - np.exp(arg0)) * (np.exp(arg1) + b * np.exp(arg2))
    return p


def _triple_exponential_guess(model, data, t=None, **kwargs):
    """Estimate initial model parameter values from data."""
    rise_time = kwargs.get("rise_time", _compute_rise_time(data, t=t))
    t0 = kwargs.get("t0", _compute_t0(data, t=t, rise_time=rise_time))
    fall_time = kwargs.get("fall_time", _compute_fall_time(data, t=t))

    params = model.make_params(a=-1., b=1., t0=t0, rise_time=rise_time, fall_time1=fall_time / 2., fall_time2=fall_time)
    params["a"].set(max=0.)
    params["b"].set(min=0.)
    params["t0"].set(min=0. if t is None else np.min(t), max=float(len(data)) if t is None else np.max(t))
    params["rise_time"].set(min=0.)
    params["fall_time1"].set(min=0.)
    params["fall_time2"].set(min=0.)
    return params


# create lmfit Model and ModelResult class that has guesses for each of the models in this file and pickles properly
# (written for lmfit version 0.9.13)
class Model(lm.Model):
    def __init__(self, func, *args, **kwargs):
        if isinstance(func, str):
            func = globals()[func]
        super(Model, self).__init__(func, *args, **kwargs)

    def guess(self, data, **kws):
        if self.func is exponential:
            return _exponential_guess_guess(self, data, **kws)
        elif self.func is double_exponential:
            return _double_exponential_guess(self, data, **kws)
        elif self.func is triple_exponential:
            return _triple_exponential_guess(self, data, **kws)
        else:
            raise NotImplementedError

    def fit(self, *args, **kwargs):
        return ModelResult(super(Model, self).fit(*args, **kwargs))


class ModelResult(lm.model.ModelResult):
    def __init__(self, result):
        super(ModelResult, self).__init__(result.model, result.params)
        self.__dict__ = result.__dict__.copy()

    def __setstate__(self, state):
        result = self.loads(state, funcdefs={name: globals()[name] for name in __all__})
        self.__dict__ = result.__dict__.copy()

    def __getstate__(self):
        return self.dumps()
