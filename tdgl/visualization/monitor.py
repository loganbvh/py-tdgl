import os
import sys
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ..device.device import Device
from .common import DEFAULT_QUANTITIES, PLOT_DEFAULTS, Quantity, auto_grid
from .io import get_plot_data


def monitor_solution(
    h5path: str,
    quantities: Union[str, Sequence[str]] = DEFAULT_QUANTITIES,
    update_interval: float = 1.0,
    shading: Literal["flat", "gouraud"] = "gouraud",
    max_cols: int = 4,
    dimensionless: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    autoscale: bool = True,
    figure_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Plots the results of a simulation while it is running.

    Args:
        h5path: Path to the temporary HDF5 file generated when running
            the solver with options.monitor=True
        quantities: The names of the quantities to animate
        update_interval: The update interval in seconds
        shading: Shading method, "flat" or "gouraud". See matplotlib.pyplot.tripcolor.
        dimensionless: Use dimensionless units for axes
        xlim: x-axis limits
        ylim: y-axis limits
        autoscale: Autoscale colorbar limits at each frame
    """

    # Try to use the appropriate GUI backend
    plt.switch_backend("agg")
    for candidate in ["macosx", "qt5agg", "qt4agg", "gtk3agg", "tkagg", "wxagg"]:
        try:
            plt.switch_backend(candidate)
            break
        except ImportError:
            continue

    if quantities is None:
        quantities = Quantity.get_keys()
    if isinstance(quantities, str):
        quantities = [quantities]
    quantities = [Quantity.from_key(name.upper()) for name in quantities]
    num_plots = len(quantities)
    figure_kwargs = figure_kwargs or dict()
    figure_kwargs.setdefault("constrained_layout", True)
    default_figsize = (
        3.5 * min(max_cols, num_plots),
        3.0 * max(1, num_plots // max_cols),
    )
    figure_kwargs.setdefault("figsize", default_figsize)
    figure_kwargs.setdefault("sharex", True)
    figure_kwargs.setdefault("sharey", True)

    with h5py.File(h5path, "r", swmr=True, libver="latest") as h5file:
        device = Device.from_hdf5(h5file["solution/device"])
        mesh = device.mesh
        if dimensionless:
            scale = 1
            units_str = "\\xi"
        else:
            scale = device.layer.coherence_length
            units_str = f"{device.ureg(device.length_units).units:~L}"
        x, y = scale * mesh.sites.T

        # Temp data to use in plots
        temp_value = np.ones(len(mesh.sites), dtype=float)
        temp_value[0] = 0
        temp_value[1] = 0.5

        plt.ion()
        fig, axes = auto_grid(num_plots, max_cols=max_cols, **figure_kwargs)
        fig.canvas.mpl_connect("close_event", lambda event: sys.exit(0))

        collections = []
        for quantity, ax in zip(quantities, axes.flat):
            ax: plt.Axes
            opts = PLOT_DEFAULTS[quantity]
            collection = ax.tripcolor(
                x,
                y,
                temp_value,
                triangles=mesh.elements,
                shading=shading,
                cmap=opts.cmap,
                vmin=opts.vmin,
                vmax=opts.vmax,
            )
            cbar = fig.colorbar(collection, ax=ax)
            cbar.set_label(opts.clabel)
            ax.set_aspect("equal")
            ax.set_title(quantity.value)
            ax.set_xlabel(f"$x$ [${units_str}$]")
            ax.set_ylabel(f"$y$ [${units_str}$]")
            collections.append(collection)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        fig.suptitle("Step: 0")
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(1e-3)

        vmins = [+np.inf for _ in quantities]
        vmaxs = [-np.inf for _ in quantities]

        prev_step = 0

        def update():
            grp = h5file["data/-1"]
            step = np.array(grp["step"])[0]
            nonlocal prev_step
            if step == prev_step:
                fig.canvas.start_event_loop(update_interval)
                return
            prev_step = step
            time = np.array(grp["time"])[0]
            dt = np.array(grp["dt"])[0]
            fig.suptitle(f"Step: {step}, time: {time:.2e}, dt: {dt:.2e}")
            for i, (quantity, collection) in enumerate(zip(quantities, collections)):
                opts = PLOT_DEFAULTS[quantity]
                values, _, _ = get_plot_data(h5file, mesh, quantity, -1)
                mask = np.abs(values - np.mean(values)) <= 6 * np.std(values)
                if opts.vmin is None:
                    if autoscale:
                        vmins[i] = np.min(values[mask])
                    else:
                        vmins[i] = min(vmins[i], np.min(values[mask]))
                else:
                    vmins[i] = opts.vmin
                if opts.vmax is None:
                    if autoscale:
                        vmaxs[i] = np.max(values[mask])
                    else:
                        vmaxs[i] = max(vmaxs[i], np.max(values[mask]))
                else:
                    vmaxs[i] = opts.vmax
                if opts.symmetric:
                    vmax = max(abs(vmins[i]), abs(vmaxs[i]))
                    vmaxs[i] = vmax
                    vmins[i] = -vmax
                if shading == "flat":
                    # https://stackoverflow.com/questions/40492511/set-array-in-tripcolor-bug
                    values = values[mesh.elements].mean(axis=1)
                collection.set_array(values)
                collection.set_clim(vmins[i], vmaxs[i])
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(update_interval)

        try:
            while True:
                if os.path.exists(h5path):
                    update()
                else:
                    plt.close(fig)
        except KeyboardInterrupt:
            plt.close(fig)
