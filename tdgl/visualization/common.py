import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class Quantity(Enum):
    ORDER_PARAMETER = "Order parameter"
    PHASE = "Phase"
    SUPERCURRENT = "Supercurrent density"
    NORMAL_CURRENT = "Normal current density"
    VORTICITY = "Vorticity"
    SCALAR_POTENTIAL = "Scalar potential"
    APPLIED_VECTOR_POTENTIAL = "Applied vector potential"
    INDUCED_VECTOR_POTENTIAL = "Induced vector potential"
    EPSILON = "Epsilon"

    @classmethod
    def get_keys(cls) -> Sequence[str]:
        return list(item.name for item in Quantity)

    @classmethod
    def from_key(cls, key: str) -> "Quantity":
        return Quantity[key.upper()]


colormaps = {
    Quantity.ORDER_PARAMETER: "viridis",
    Quantity.PHASE: "twilight_shifted",
    Quantity.SUPERCURRENT: "inferno",
    Quantity.NORMAL_CURRENT: "inferno",
    Quantity.SCALAR_POTENTIAL: "magma",
    Quantity.APPLIED_VECTOR_POTENTIAL: "cividis",
    Quantity.INDUCED_VECTOR_POTENTIAL: "cividis",
    Quantity.EPSILON: "viridis",
    Quantity.VORTICITY: "coolwarm",
}


@dataclass
class PlotDefault:
    cmap: str
    clabel: str
    xlabel: str = "$x/\\xi$"
    ylabel: str = "$y/\\xi$"


PLOT_DEFAULTS = {
    Quantity.ORDER_PARAMETER: PlotDefault(cmap="viridis", clabel="$|\\psi|$"),
    Quantity.PHASE: PlotDefault(
        cmap="twilight_shifted",
        clabel="$\\arg(\\psi)/\\pi$",
    ),
    Quantity.SUPERCURRENT: PlotDefault(cmap="inferno", clabel="$|\\vec{{J}}_s|/J_0$"),
    Quantity.NORMAL_CURRENT: PlotDefault(cmap="inferno", clabel="$|\\vec{{J}}_n|/J_0$"),
    Quantity.SCALAR_POTENTIAL: PlotDefault(cmap="magma", clabel="$\\mu/v_0$"),
    Quantity.APPLIED_VECTOR_POTENTIAL: PlotDefault(
        cmap="cividis", clabel="$a_\\mathrm{{applied}}/(\\xi B_{{c2}})$"
    ),
    Quantity.INDUCED_VECTOR_POTENTIAL: PlotDefault(
        cmap="cividis", clabel="$a_\\mathrm{{induced}}/(\\xi B_{{c2}})$"
    ),
    Quantity.EPSILON: PlotDefault(cmap="viridis", clabel="$\\epsilon$"),
    Quantity.VORTICITY: PlotDefault(
        cmap="coolwarm", clabel="$(\\vec{{\\nabla}}\\times\\vec{{J}})\\cdot\\hat{{z}}$"
    ),
}

DEFAULT_QUANTITIES = (
    "order_parameter",
    "phase",
    "supercurrent",
    "normal_current",
)


def auto_grid(
    num_plots: int,
    max_cols: int = 3,
    delaxes: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Creates a grid of at least ``num_plots`` subplots
    with at most ``max_cols`` columns.

    Additional keyword arguments are passed to ``plt.subplots()``.

    Args:
        num_plots: Total number of plots that will be populated.
        max_cols: Maximum number of columns in the grid.
        delaxes: Whether to remove unused axes.

    Returns:
        matplotlib figure and axes
    """
    ncols = min(max_cols, num_plots)
    nrows = int(np.ceil(num_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    if not isinstance(axes, (list, np.ndarray)):
        axes = np.array([axes])
    axes = np.asarray(axes)
    if delaxes:
        flat_axes = list(axes.flat)
        for ax in flat_axes[num_plots:]:
            fig.delaxes(ax)
    return fig, axes


@contextmanager
def non_gui_backend():
    """A contextmanager that temporarily uses a non-GUI backend for matplotlib."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Matplotlib is currently using agg"
        )
        try:
            old_backend = mpl.get_backend()
            mpl.use("Agg")
            yield
        finally:
            mpl.use(old_backend)


def auto_range_iqr(
    data_array: np.ndarray,
    cutoff_percentile: Union[float, Tuple[float, float]] = 1,
) -> Tuple[float, float]:
    """Get the min and max range of the provided array that excludes outliers
    following the IQR rule.

    This function computes the inter-quartile-range (IQR), defined by Q3-Q1,
    i.e. the percentiles for 75 and 25 percent of the distribution. The region
    without outliers is defined by [Q1-1.5*IQR, Q3+1.5*IQR].
    Taken from `qcodes <https://github.com/QCoDeS/Qcodes/blob/
    6c8f7202f6b6fca4884bfc0f6e1e9a6564628d75/qcodes/utils/plotting.py#L28-L76>`_.

    Args:
        data_array: Array of arbitrary dimension containing the
            statistical data.
        cutoff_percentile: Percentile of data that may maximally be
            clipped on both sides of the distribution. If given a
            tuple (a, b) the percentile limits will be a and 100-b.

    Returns:
        vmin, vmax
    """
    if isinstance(cutoff_percentile, tuple):
        bottom, top = cutoff_percentile
    else:
        bottom = cutoff_percentile
        top = 100 - bottom
    z = data_array.flatten()
    zmax = np.nanmax(z)
    zmin = np.nanmin(z)
    zrange = zmax - zmin
    pmin, q3, q1, pmax = np.nanpercentile(z, [bottom, 75, 25, top])
    iqr = q3 - q1
    # handle corner case of all data zero, such that IQR is zero
    # to counter numerical artifacts do not test IQR == 0, but IQR on its
    # natural scale (zrange) to be smaller than some very small number.
    # also test for zrange to be 0.0 to avoid division by 0.
    if zrange == 0.0 or iqr / zrange < 1e-8:
        vmin = zmin
        vmax = zmax
    else:
        vmin = max(q1 - 1.5 * iqr, zmin)
        vmax = min(q3 + 1.5 * iqr, zmax)
        vmin = min(vmin, pmin)
        vmax = max(vmax, pmax)
    return vmin, vmax
