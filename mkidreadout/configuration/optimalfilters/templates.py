from __future__ import division
import types
import numpy as np
import lmfit as lm
from scipy.signal import argrelmax

EPS = np.finfo(np.float64).eps

__all__ = ["exponential", "double_exponential", "triple_exponential"]


def _compute_t0(data, t=None):
    peak = np.argmin(data)
    t0 = t[peak] if t is not None else float(peak)
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


def _exponential(t, a, t0, fall_time):
    p = np.zeros_like(t, dtype=np.float)
    arg1 = -(t[t >= t0] - t0) / max(EPS, fall_time)
    p[t >= t0] = a * np.exp(arg1)
    return p


def _exponential_guess(self, data, t=None, **kwargs):
    """Estimate initial model parameter values from data."""
    t0 = kwargs.get("t0", _compute_t0(data, t=t))
    fall_time = kwargs.get("fall_time", _compute_fall_time(data, t=t))

    params = self.make_params(a=-1., t0=t0, fall_time=fall_time)
    params["a"].set(max=0.)
    params["t0"].set(min=0. if t is None else np.min(t), max=float(len(data)) if t is None else np.max(t))
    params["fall_time"].set(min=0.)
    return params


def _double_exponential(t, a, t0, rise_time, fall_time):
    p = np.zeros_like(t, dtype=np.float)
    arg0 = -(t[t >= t0] - t0) / max(EPS, rise_time)
    arg1 = -(t[t >= t0] - t0) / max(EPS, fall_time)
    p[t >= t0] = a * (1 - np.exp(arg0)) * np.exp(arg1)
    return p


def _double_exponential_guess(self, data, t=None, **kwargs):
    """Estimate initial model parameter values from data."""
    t0 = kwargs.get("t0", _compute_t0(data, t=t))
    rise_time = kwargs.get("rise_time", _compute_rise_time(data, t=t))
    fall_time = kwargs.get("fall_time", _compute_fall_time(data, t=t))

    params = self.make_params(a=-1., t0=t0, rise_time=rise_time, fall_time=fall_time)
    params["a"].set(max=0.)
    params["t0"].set(min=0. if t is None else np.min(t), max=float(len(data)) if t is None else np.max(t))
    params["rise_time"].set(min=0.)
    params["fall_time"].set(min=0.)
    return params


def _triple_exponential(t, a, t0, rise_time, fall_time1, fall_time2):
    p = np.zeros_like(t, dtype=np.float)
    arg0 = -(t[t >= t0] - t0) / max(EPS, rise_time)
    arg1 = -(t[t >= t0] - t0) / max(EPS, fall_time1)
    arg2 = -(t[t >= t0] - t0) / max(EPS, fall_time2)
    p[t >= t0] = a * (1 - np.exp(arg0)) * (np.exp(arg1) + np.exp(arg2))
    return p


def _triple_exponential_guess(self, data, t=None, **kwargs):
    """Estimate initial model parameter values from data."""
    t0 = kwargs.get("t0", _compute_t0(data, t=t))
    rise_time = kwargs.get("rise_time", _compute_rise_time(data, t=t))
    fall_time = kwargs.get("fall_time", _compute_fall_time(data, t=t))

    params = self.make_params(a=-1., t0=t0, rise_time=rise_time, fall_time1=fall_time / 2., fall_time2=2. * fall_time)
    params["a"].set(max=0.)
    params["t0"].set(min=0. if t is None else np.min(t), max=float(len(data)) if t is None else np.max(t))
    params["rise_time"].set(min=0.)
    params["fall_time1"].set(min=0.)
    params["fall_time2"].set(min=0.)
    return params


# create lmfit models usable by make_filters.py
exponential = lm.Model(_exponential)
exponential.guess = types.MethodType(_exponential_guess, exponential)

double_exponential = lm.Model(_double_exponential)
double_exponential.guess = types.MethodType(_double_exponential_guess, double_exponential)

triple_exponential = lm.Model(_triple_exponential)
triple_exponential.guess = types.MethodType(_triple_exponential_guess, triple_exponential)
