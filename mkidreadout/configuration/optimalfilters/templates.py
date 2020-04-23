from __future__ import with_statement, division
import types
import flint
import numpy as np
import lmfit as lm
import mpmath as mp
from scipy.signal import argrelmax

EPS = np.finfo(np.float64).eps

__all__ = ["exponential", "double_exponential", "triple_exponential", "exponential_lowpass"]


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
    p = np.empty_like(t, dtype=np.float)
    arg1 = -(t[t >= t0] - t0) / max(EPS, fall_time)
    p[t >= t0] = a * np.exp(arg1)
    p[t < t0] = 0
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
    p = np.empty_like(t, dtype=np.float)
    arg0 = -(t[t >= t0] - t0) / max(EPS, rise_time)
    arg1 = -(t[t >= t0] - t0) / max(EPS, fall_time)
    p[t >= t0] = a * (1 - np.exp(arg0)) * np.exp(arg1)
    p[t < t0] = 0
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


def triple_exponential(t, a, t0, rise_time, fall_time1, fall_time2):
    p = np.empty_like(t, dtype=np.float)
    arg0 = -(t[t >= t0] - t0) / max(EPS, rise_time)
    arg1 = -(t[t >= t0] - t0) / max(EPS, fall_time1)
    arg2 = -(t[t >= t0] - t0) / max(EPS, fall_time2)
    p[t >= t0] = a * (1 - np.exp(arg0)) * (np.exp(arg1) + np.exp(arg2))
    p[t < t0] = 0
    return p


def _triple_exponential_guess(model, data, t=None, **kwargs):
    """Estimate initial model parameter values from data."""
    rise_time = kwargs.get("rise_time", _compute_rise_time(data, t=t))
    t0 = kwargs.get("t0", _compute_t0(data, t=t, rise_time=rise_time))
    fall_time = kwargs.get("fall_time", _compute_fall_time(data, t=t))

    params = model.make_params(a=-1., t0=t0, rise_time=rise_time, fall_time1=fall_time / 2., fall_time2=fall_time)
    params["a"].set(max=0.)
    params["t0"].set(min=0. if t is None else np.min(t), max=float(len(data)) if t is None else np.max(t))
    params["rise_time"].set(min=0.)
    params["fall_time1"].set(min=0.)
    params["fall_time2"].set(min=0.)
    return params


@partial(np.vectorize, otypes=[complex])  # scipy.special.hyp2f1 has lots of numerical problems
def hyp2f1(a, b, c, z):
    try:
        result = flint.good(lambda: flint.acb(z).hypgeom_2f1(a, b, c), dps=25, parts=False)
    except ValueError:
        with mp.workdps(15):
            result = mp.mp.hyp2f1(a, b, c, z)  # slower but can better handle infs and zeros
    return result


def _linear1(t, a, t0, rise_time, fall_time, xqp0):
    """
    Linear approximation for the pulse shape for when xqp0 << 1 and when
    fall_time is near rise_time. (t>t0)
    """
    fall_time = max(EPS, fall_time)
    dtr = rise_time - fall_time
    arg0 = -(t - t0) / fall_time

    p = a * (
            (t - t0) * np.exp(arg0) / fall_time +  # zeroth order
            dtr / fall_time * (  # first order in dtr / fall_time
                (t - t0) * (t - t0 - 2 * fall_time) * np.exp(arg0) / (2 * fall_time**2)
            ) +
            xqp0 * (  # first order in xqp0
                2 * np.exp(arg0) - np.exp(2 * arg0) - (t - t0 + fall_time) * np.exp(arg0) / fall_time
            )
    )
    return p


def _linear2(t, a, t0, rise_time, fall_time, xqp0):
    """
    Linear approximation for the pulse shape for when xqp0 << 1 and when
    fall_time is near 2 * rise_time. (t>t0)
    """
    fall_time = max(EPS, fall_time)
    dtr = rise_time - fall_time / 2
    arg0 = -(t - t0) / fall_time

    p = a * (
        2 * (np.exp(arg0) - np.exp(2 * arg0)) +  # zeroth order
        dtr / fall_time * (  # first order in dtr / fall_time
            4 * (np.exp(arg0) - np.exp(2 * arg0)) - 8 * (t - t0) * np.exp(2 * arg0) / fall_time
        ) +
        xqp0 * (  # first order in xqp0
            2 * (np.exp(2 * arg0) - np.exp(arg0) + (t - t0) * np.exp(2 * arg0) / fall_time)
        )
    )
    return p


def _linear3(t, a, t0, rise_time, fall_time, xqp0):
    """
    Linear approximation for the pulse shape for when xqp0 << 1 and when
    fall_time is not equal to rise_time or 2 * rise_time. (t>t0)
    """
    arg0 = -(t - t0) / max(EPS, fall_time)
    arg1 = -(t - t0) / max(EPS, rise_time)

    p = a * fall_time / max(EPS, fall_time - rise_time, key=abs) * (
            np.exp(arg0) - np.exp(arg1) -  # zeroth order
            xqp0 / max(EPS, fall_time - 2 * rise_time, key=abs) * (  # first order
                np.exp(arg0) * (fall_time - 2 * rise_time) + np.exp(arg1) * rise_time +
                np.exp(2 * arg0) * (rise_time - fall_time)
                )
            )
    return p


def _linear(t, a, t0, rise_time, fall_time, xqp0):
    """
    Linear approximation for the pulse shape for when xqp0 << 1. (t>t0)
    """
    dt_ratio1 = np.abs(fall_time - rise_time) / max(EPS, fall_time)
    dt_ratio2 = np.abs(fall_time - 2 * rise_time) / max(EPS, fall_time)
    if dt_ratio1 < 0.1:
        if dt_ratio1 < 0.05:
            p = _linear1(t, a, t0, rise_time, fall_time, xqp0)
        else:
            p = ((0.1 - dt_ratio1) / (0.1 - 0.05) * _linear1(t, a, t0, rise_time, fall_time, xqp0) +
                 (dt_ratio1 - 0.05) / (0.1 - 0.05) * _linear3(t, a, t0, rise_time, fall_time, xqp0))
    elif dt_ratio2 < 0.1:
        if dt_ratio2 < 0.05:
            p = _linear2(t, a, t0, rise_time, fall_time, xqp0)
        else:
            p = ((0.1 - dt_ratio2) / (0.1 - 0.05) * _linear2(t, a, t0, rise_time, fall_time, xqp0) +
                 (dt_ratio2 - 0.05) / (0.1 - 0.05) * _linear3(t, a, t0, rise_time, fall_time, xqp0))
    else:
        p = _linear3(t, a, t0, rise_time, fall_time, xqp0)
    return p


def _nonlinear(t, a, t0, rise_time, fall_time, xqp0):
    """
    Exact expression for the pulse shape that has an artificial singularity at
    xqp0 = 0. (t>t0)
    """
    with np.errstate(over='ignore'):
        if fall_time == 0:
            return np.zeros_like(t)
        arg0 = -(t - t0) / max(EPS, rise_time)
        arg1 = fall_time / max(EPS, rise_time)
        arg2 = (xqp0 + 1) / max(EPS, xqp0)
        arg3 = (t - t0) / max(EPS, fall_time)

        p = a / xqp0 * (
            np.exp(arg0) * hyp2f1(1, arg1, arg1 + 1, arg2) -
            hyp2f1(1, arg1, arg1 + 1, arg2 * np.exp(arg3))
        ).real
    return p


def exponential_lowpass(t, a, t0, rise_time, fall_time, xqp0):
    p = np.empty_like(t, dtype=np.float)
    # xqp0 is initial fractional quasi-particle density
    # low pass filter Green's function + quasi-particle decay from Fyhrie Proc. SPIE 10708. 2018.
    if xqp0 <= 0.05:  # use linearized version for low quasi-particle density
        p[t >= t0] = _linear(t[t >= t0], a, t0, rise_time, fall_time, xqp0)
    elif 0.05 < xqp0 < 0.1:  # transition to nonlinear version
        p[t >= t0] = ((0.1 - xqp0) / (0.1 - 0.05) * _linear(t[t >= t0], a, t0, rise_time, fall_time, xqp0) +
                      (xqp0 - 0.05) / (0.1 - 0.05) * _nonlinear(t[t >= t0], a, t0, rise_time, fall_time, xqp0))
    else:  # full nonlinear version (may have numerical problems)
        p[t >= t0] = _nonlinear(t[t >= t0], a, t0, rise_time, fall_time, xqp0)
    p[t < t0] = 0
    if np.isnan(p).any() or np.isinf(p).any():
        raise ValueError("bad function evaluation at {}, {}, {}, {}, {}".format(a, t0, rise_time, fall_time, xqp0))
    return p


def _exponential_lowpass_guess(model, data, t=None, **kwargs):
    rise_time = kwargs.get("rise_time", _compute_rise_time(data, t=t))
    t0 = kwargs.get("t0", _compute_t0(data, t=t, rise_time=rise_time))
    fall_time = kwargs.get("fall_time", _compute_fall_time(data, t=t))
    xqp0 = kwargs.get("xqp0", 0.1)

    params = model.make_params(a=-1., t0=t0, rise_time=rise_time, fall_time=fall_time, xqp0=xqp0)
    params["a"].set(max=0.)
    params["t0"].set(min=0. if t is None else np.min(t), max=float(len(data)) if t is None else np.max(t))
    params["rise_time"].set(min=0., max=100.)
    params["fall_time"].set(min=0., max=100.)
    params.add("xqp_sqrt", value=np.sqrt(xqp0))
    params["xqp0"].set(expr="xqp_sqrt**2")
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
        elif self.func is exponential_lowpass:
            return _exponential_lowpass_guess(self, data, **kwargs)
        else:
            raise NotImplementedError

    def fit(self, *args, **kwargs):
        return ModelResult(super(Model, self).fit(*args, **kwargs))


class ModelResult(lm.model.ModelResult):
    def __init__(self, result):
        super(ModelResult, self).__init__(result.model, result.params)
        self.__dict__ = result.__dict__.copy()

    def __setstate__(self, state):
        # allows pickle to work with custom guess methods
        result = self.loads(state, funcdefs={name: globals()[name] for name in __all__})
        self.__dict__ = result.__dict__.copy()

    def __getstate__(self):
        return self.dumps()


if __name__ == '__main__':
    # test widget for the exponential_lowpass function
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    figure, axes = plt.subplots(figsize=(8, 8))

    tt = np.linspace(0, 100, 1000)
    pp = [1, 25, 2, 30, 1]
    y_data = exponential_lowpass(tt, *pp)
    line, = axes.plot(tt, ydata, lw=2)
    axes.set_ylim(bottom=min(-1, y_data.min()), top=max(1, y_data.max()))
    axes.set_xlim(left=tt.min(), right=tt.max())

    axes.set_ylabel("pulse amplitude")
    axes.set_xlabel("time")

    ax_color = 'lightgoldenrodyellow'
    dy = 0.05
    slider_axes = [
        figure.add_axes([0.1, 0.05, 0.8, 0.03], facecolor=ax_color),
        figure.add_axes([0.1, 0.05 + dy * 1, 0.8, 0.03], facecolor=ax_color),
        figure.add_axes([0.1, 0.05 + dy * 2, 0.8, 0.03], facecolor=ax_color),
        figure.add_axes([0.1, 0.05 + dy * 3, 0.8, 0.03], facecolor=ax_color),
        figure.add_axes([0.1, 0.05 + dy * 4, 0.8, 0.03], facecolor=ax_color)
        ]

    sliders = [
        Slider(slider_axes[0], 'A', 0, 10, valinit=pp[0], valstep=0.001),
        Slider(slider_axes[1], 't0', 0, 100, valinit=pp[1], valstep=0.001),
        Slider(slider_axes[2], 'tr', 0, 100, valinit=pp[2], valstep=0.001),
        Slider(slider_axes[3], 'tqp', 0, 100, valinit=pp[3], valstep=0.001),
        Slider(slider_axes[4], 'xqp0', 0, 10, valinit=pp[4], valstep=0.001)
        ]

    def update(val):
        for index, s in enumerate(sliders):
            pp[index] = s.val
        yy = exponential_lowpass(t, *pp)
        line.set_ydata(yy)
        axes.set_ylim(bottom=min(-1, yy.min()), top=max(1, yy.max()))
        figure.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure.tight_layout(rect=[0, dy * (len(sliders) + 2), 1, 1])
    plt.show()
