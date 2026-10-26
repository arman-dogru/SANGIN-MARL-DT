"""Shared exact empirical delivered-command distributions for read-only reports."""
import numpy as np


def empirical_steps(values, *, complementary=False):
    """Right-continuous F(t)=P(T<=t), or P(T>t), with exact tie counts."""
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return np.array([]),np.array([])
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Latency samples must be finite and nonnegative")
    x,counts = np.unique(np.sort(values),return_counts=True)
    y = np.cumsum(counts)/len(values)
    if x[0] > 0:
        x,y = np.r_[0.,x],np.r_[0.,y]
    return x,1-y if complementary else y


def plot_empirical_latency(ax, values, *, complementary=False, log_tail=False,
                           unit_scale=1., **style):
    """The repository's step-CDF convention, also supporting a log CCDF.

    A terminal zero CCDF has no logarithm. Draw its positive left limit up to
    the maximum sample and leave the zero endpoint unplotted. Numeric exports
    retain the exact zero; no synthetic probability or latency is introduced.
    """
    x,y = empirical_steps(values,complementary=complementary)
    if not len(x):
        return None
    if log_tail:
        if not complementary:
            raise ValueError("Log-tail presentation requires a CCDF")
        if len(y) > 1 and y[-1] == 0:
            y = y.copy();y[-1] = y[-2]
        positive = y > 0
        x,y = x[positive],y[positive]
        ax.set_yscale("log")
    return ax.step(x*unit_scale,y,where="post",**style)
