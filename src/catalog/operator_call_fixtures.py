from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
import math
import tempfile
from typing import Any


CallArguments = tuple[list[Any], dict[str, Any]]


def interpolation(_resources: ExitStack) -> CallArguments:
    return [[0.5], [0.0, 1.0], [0.0, 2.0]], {}


def scalar_root(_resources: ExitStack) -> CallArguments:
    return [lambda x: x * x - 2.0], {"bracket": (0.0, 2.0), "method": "bisect"}


def exponential_decay(_resources: ExitStack) -> CallArguments:
    return [lambda t, y: -y, (0.0, 1.0), [1.0]], {"rtol": 1e-8, "atol": 1e-10}


def identity_matrix(_resources: ExitStack) -> CallArguments:
    return [[[1, 0], [0, 1]]], {}


def symbolic_derivative(_resources: ExitStack) -> CallArguments:
    import sympy as sp

    x = sp.Symbol("x")
    return [x**2, x], {}


def geodesic(_resources: ExitStack) -> CallArguments:
    return [], {
        "metric": "Schwarzschild",
        "metric_params": (0.0,),
        "position": [10.0, math.pi / 2, 0.0],
        "momentum": [0.0, 0.0, 3.5],
        "steps": 2,
        "delta": 0.01,
    }


def minkowski_metric(_resources: ExitStack) -> CallArguments:
    import sympy as sp

    return [sp.diag(-1, 1, 1, 1).tolist(), sp.symbols("t x y z")], {"config": "ll"}


def flat_christoffel(_resources: ExitStack) -> CallArguments:
    import sympy as sp

    return [sp.Array.zeros(4, 4, 4), sp.symbols("t x y z")], {"config": "ull"}


def flat_ricci(_resources: ExitStack) -> CallArguments:
    import sympy as sp

    return [sp.Array.zeros(4, 4), sp.symbols("t x y z")], {"config": "ll"}


def time_series(_resources: ExitStack) -> CallArguments:
    return [[0.0] * 16], {"sample_rate": 16}


def binary_waveform(_resources: ExitStack) -> CallArguments:
    return [], {
        "approximant": "TaylorT4",
        "mass1": 30,
        "mass2": 30,
        "delta_t": 1 / 1024,
        "f_lower": 30,
    }


def matched_filter(resources: ExitStack) -> CallArguments:
    import numpy as np
    from pycbc.types import TimeSeries

    data = TimeSeries(np.sin(2 * np.pi * 64 * np.arange(1024) / 1024), delta_t=1 / 1024)
    return [data, data], {"low_frequency_cutoff": 20}


def gaussian_sampler(resources: ExitStack) -> CallArguments:
    import bilby
    import numpy as np

    def constant(x, mu):
        return np.full_like(x, mu, dtype=float)

    likelihood = bilby.core.likelihood.GaussianLikelihood(
        x=np.arange(3.0), y=np.zeros(3), func=constant, sigma=1.0
    )
    outdir = resources.enter_context(tempfile.TemporaryDirectory(prefix="scidata-bilby-"))
    return [likelihood], {
        "priors": {"mu": bilby.core.prior.Uniform(-1, 1, name="mu")},
        "sampler": "dynesty",
        "nlive": 32,
        "dlogz": 10.0,
        "maxcall": 128,
        "sample": "unif",
        "bound": "single",
        "seed": 42,
        "npool": 1,
        "save": False,
        "plot": False,
        "resume": False,
        "check_point": False,
        "print_progress": False,
        "outdir": outdir,
    }


API_CALL_FIXTURES: dict[str, Callable[[ExitStack], CallArguments]] = {
    "bilby.run_sampler": gaussian_sampler,
    "einsteinpy.geodesic.Nulllike": geodesic,
    "einsteinpy.geodesic.Timelike": geodesic,
    "einsteinpy.symbolic.ChristoffelSymbols": flat_christoffel,
    "einsteinpy.symbolic.MetricTensor": minkowski_metric,
    "einsteinpy.symbolic.RicciTensor": flat_ricci,
    "gwpy.timeseries.TimeSeries": time_series,
    "numpy.interp": interpolation,
    "pycbc.filter.matched_filter": matched_filter,
    "pycbc.waveform.get_td_waveform": binary_waveform,
    "scipy.integrate.solve_ivp": exponential_decay,
    "scipy.optimize.root_scalar": scalar_root,
    "sympy.Matrix": identity_matrix,
    "sympy.diff": symbolic_derivative,
}
