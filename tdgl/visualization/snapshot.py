import numbers
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ..solution.data import get_data_range
from ..solution.solution import Solution
from .common import DEFAULT_QUANTITIES, PLOT_DEFAULTS, Quantity, auto_grid
from .io import get_plot_data, get_state_string


def generate_snapshots(
    input_path: str,
    times: Union[float, Sequence[float]],
    quantities: Union[str, Sequence[str]] = DEFAULT_QUANTITIES,
    shading: Literal["flat", "gouraud"] = "gouraud",
    autoscale: bool = False,
    max_cols: int = 4,
    dimensionless: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    axis_labels: bool = False,
    axes_off: bool = False,
    title_off: bool = False,
    full_title: bool = True,
    figure_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Tuple[plt.Figure, plt.Axes]]:
    """Generates snapshots of a TDGL simulation.

    Args:
        input_path: The path to an H5 file containing
            the :class:`tdgl.Solution` you would like to animate.
        times: The dimensionless time(s) for which to generate a snapshot.
        quantities: The names of the quantities to animate.
        shading: Shading method, "flat" or "gouraud". See matplotlib.pyplot.tripcolor.
        autoscale: Autoscale colorbar limits to exclude outlier points.
        max_cols: The maxiumum number of columns in the subplot grid.
        dimensionless: Use dimensionless units for axes
        xlim: x-axis limits
        ylim: y-axis limits
        axes_off: Turn off the axes for each subplot.
        title_off: Turn off the figure suptitle.
        full_title: Include the full "state" for each frame in the figure suptitle.
        figure_kwargs: Keyword arguments passed to ``plt.subplots()`` when creating
            the figure.

    Returns:
        The matplotlib figure and axes for each time in ``times``
    """
    if isinstance(times, numbers.Real):
        times = [times]
    if quantities is None:
        quantities = Quantity.get_keys()
    if isinstance(quantities, str):
        quantities = [quantities]
    quantities = [Quantity.from_key(name.upper()) for name in quantities]
    num_plots = len(quantities)
    figure_kwargs = figure_kwargs or dict()
    figure_kwargs.setdefault("constrained_layout", True)
    default_figsize = (
        3.25 * min(max_cols, num_plots),
        2.5 * max(1, num_plots // max_cols),
    )
    figure_kwargs.setdefault("figsize", default_figsize)
    figure_kwargs.setdefault("sharex", True)
    figure_kwargs.setdefault("sharey", True)

    solution = Solution.from_hdf5(input_path)
    device = solution.device
    mesh = device.mesh
    mesh = device.mesh
    if dimensionless:
        scale = 1
        units_str = "\\xi"
    else:
        scale = device.layer.coherence_length
        units_str = f"{device.ureg(device.length_units).units:~L}"
    x, y = scale * mesh.sites.T

    figures = []

    with h5py.File(input_path, "r") as h5file:
        _, max_step = get_data_range(h5file)
        for time in times:
            step = solution.closest_solve_step(time)

            fig, axes = auto_grid(num_plots, max_cols=max_cols, **figure_kwargs)
            state = get_state_string(h5file, step, max_step)
            if not full_title:
                state = state.split(",")[0]
            if not title_off:
                fig.suptitle(state)

            for quantity, ax in zip(quantities, axes.flat):
                ax: plt.Axes
                opts = PLOT_DEFAULTS[quantity]
                values, _, _ = get_plot_data(h5file, mesh, quantity, step)
                if autoscale:
                    mask = np.abs(values - np.mean(values)) <= 6 * np.std(values)
                else:
                    mask = np.ones_like(values, dtype=bool)
                if opts.vmin is None:
                    vmin = np.min(values[mask])
                else:
                    vmin = opts.vmin
                if opts.vmax is None:
                    vmax = np.max(values[mask])
                else:
                    vmax = opts.vmax
                if opts.symmetric:
                    vmax = max(abs(vmin), abs(vmax))
                    vmin = -vmax
                collection = ax.tripcolor(
                    x,
                    y,
                    values,
                    triangles=mesh.elements,
                    shading=shading,
                    cmap=opts.cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                cbar = fig.colorbar(collection, ax=ax)
                cbar.set_label(opts.clabel)
                ax.set_aspect("equal")
                ax.set_title(quantity.value)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                if axes_off:
                    ax.axis("off")
                if axis_labels:
                    ax.set_xlabel(f"$x$ [${units_str}$]")
                    ax.set_ylabel(f"$y$ [${units_str}$]")

            figures.append((fig, axes))

    return figures
