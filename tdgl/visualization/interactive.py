import logging
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import h5py
import numpy as np
from matplotlib import pyplot as plt

from ..device.device import Device
from ..solution.data import get_data_range
from .common import DEFAULT_QUANTITIES, PLOT_DEFAULTS, Quantity, auto_grid
from .io import get_plot_data, get_state_string


class InteractivePlot:
    def __init__(
        self,
        input_file: str,
        shading: Literal["flat", "gouraud"] = "gouraud",
        dimensionless: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        axis_labels: bool = False,
        logger: logging.Logger = None,
    ):
        self.input_file = input_file
        self.shading = shading
        self.dimensionless = dimensionless
        self.xlim = xlim
        self.ylim = ylim
        self.axis_labels = axis_labels
        self.frame = 0
        self.quantity = Quantity.ORDER_PARAMETER
        self.logger = logger or logging.getLogger()

    def show(self):
        with h5py.File(self.input_file, "r") as h5file:
            device = Device.from_hdf5(h5file["solution/device"])
            mesh = device.mesh
            if self.dimensionless:
                scale = 1
                units_str = "\\xi"
            else:
                scale = device.layer.coherence_length
                units_str = f"{device.ureg(device.length_units).units:~L}"
            x, y = scale * mesh.sites.T
            # Get the ranges for the frame
            min_frame, max_frame = get_data_range(h5file)

            def on_keypress(event):
                if event.key == "right":
                    self.frame = np.minimum(self.frame + 1, max_frame)

                elif event.key == "left":
                    self.frame = np.maximum(self.frame - 1, min_frame)

                if event.key == "shift+right":
                    self.frame = np.minimum(self.frame + 10, max_frame)

                elif event.key == "shift+left":
                    self.frame = np.maximum(self.frame - 10, min_frame)

                elif event.key == "up":
                    self.frame = np.minimum(self.frame + 100, max_frame)

                elif event.key == "down":
                    self.frame = np.maximum(self.frame - 100, min_frame)

                elif event.key == "shift+up":
                    self.frame = np.minimum(self.frame + 1000, max_frame)

                elif event.key == "shift+down":
                    self.frame = np.maximum(self.frame - 1000, min_frame)

                elif event.key == "home":
                    self.frame = min_frame

                elif event.key == "end":
                    self.frame = max_frame

                elif event.key == "1":
                    self.quantity = Quantity.ORDER_PARAMETER

                elif event.key == "2":
                    self.quantity = Quantity.PHASE

                elif event.key == "3":
                    self.quantity = Quantity.SUPERCURRENT

                elif event.key == "4":
                    self.quantity = Quantity.NORMAL_CURRENT

                elif event.key == "5":
                    self.quantity = Quantity.SCALAR_POTENTIAL

                elif event.key == "6":
                    self.quantity = Quantity.APPLIED_VECTOR_POTENTIAL

                elif event.key == "7":
                    self.quantity = Quantity.INDUCED_VECTOR_POTENTIAL

                elif event.key == "8":
                    self.quantity = Quantity.EPSILON

                elif event.key == "9":
                    self.quantity = Quantity.VORTICITY

                draw()

            def draw():
                values, direction, limits = get_plot_data(
                    h5file, mesh, self.quantity, self.frame
                )
                state = get_state_string(h5file, self.frame, max_frame)

                fig.suptitle(f"{self.quantity.value}\n{state}")
                if self.shading == "flat":
                    # https://stackoverflow.com/questions/40492511/set-array-in-tripcolor-bug
                    values = values[mesh.elements].mean(axis=1)
                triplot.set_array(values)
                triplot.set_clim(*limits)
                triplot.set_cmap(PLOT_DEFAULTS[self.quantity].cmap)
                cbar.set_label(PLOT_DEFAULTS[self.quantity].clabel)
                fig.canvas.draw()

            # Temp data to use in plots
            temp_value = np.ones_like(x)
            temp_value[0] = 0
            temp_value[1] = 0.5

            fig, ax = plt.subplots()
            fig.subplots_adjust(top=0.8)
            fig.canvas.mpl_connect("key_press_event", on_keypress)
            triplot = ax.tripcolor(
                x, y, temp_value, triangles=mesh.elements, shading=self.shading
            )
            cbar = fig.colorbar(triplot)
            ax.set_aspect("equal")
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            if self.axis_labels:
                ax.set_xlabel(f"$x$ [${units_str}$]")
                ax.set_ylabel(f"$y$ [${units_str}$]")
            draw()
            plt.show()


class MultiInteractivePlot:
    def __init__(
        self,
        input_file: str,
        shading: Literal["flat", "gouraud"] = "gouraud",
        dimensionless: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        axis_labels: bool = False,
        quantities: Sequence[str] = DEFAULT_QUANTITIES,
        max_cols: int = 4,
        logger: logging.Logger = None,
        figure_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.input_file = input_file
        self.shading = shading
        self.dimensionless = dimensionless
        self.xlim = xlim
        self.ylim = ylim
        self.axis_labels = axis_labels
        self.frame = 0
        if quantities is None:
            quantities = Quantity.get_keys()
        self.quantities = [Quantity.from_key(name) for name in quantities]
        self.num_plots = len(quantities)
        self.max_cols = max_cols
        self.logger = logger if logger is not None else logging.getLogger()
        self.figure_kwargs = figure_kwargs or dict()
        self.figure_kwargs.setdefault("constrained_layout", True)
        default_figsize = (
            3.25 * min(self.max_cols, self.num_plots),
            3 * max(1, self.num_plots // self.max_cols),
        )
        self.figure_kwargs.setdefault("figsize", default_figsize)
        self.figure_kwargs.setdefault("sharex", True)
        self.figure_kwargs.setdefault("sharey", True)

    def show(self):
        with h5py.File(self.input_file, "r") as h5file:
            device = Device.from_hdf5(h5file["solution/device"])
            mesh = device.mesh
            if self.dimensionless:
                scale = 1
                units_str = "\\xi"
            else:
                scale = device.layer.coherence_length
                units_str = f"{device.ureg(device.length_units).units:~L}"
            x, y = scale * mesh.sites.T

            min_frame, max_frame = get_data_range(h5file)

            def on_keypress(event):
                if event.key == "right":
                    self.frame = np.minimum(self.frame + 1, max_frame)

                elif event.key == "left":
                    self.frame = np.maximum(self.frame - 1, min_frame)

                if event.key == "shift+right":
                    self.frame = np.minimum(self.frame + 10, max_frame)

                elif event.key == "shift+left":
                    self.frame = np.maximum(self.frame - 10, min_frame)

                elif event.key == "up":
                    self.frame = np.minimum(self.frame + 100, max_frame)

                elif event.key == "down":
                    self.frame = np.maximum(self.frame - 100, min_frame)

                elif event.key == "shift+up":
                    self.frame = np.minimum(self.frame + 1000, max_frame)

                elif event.key == "shift+down":
                    self.frame = np.maximum(self.frame - 1000, min_frame)

                elif event.key == "home":
                    self.frame = min_frame

                elif event.key == "end":
                    self.frame = max_frame

                draw()

            vmins = [+np.inf for _ in self.quantities]
            vmaxs = [-np.inf for _ in self.quantities]

            def draw():
                state = get_state_string(h5file, self.frame, max_frame)
                fig.suptitle(state)
                for i, (quantity, collection) in enumerate(
                    zip(self.quantities, collections)
                ):
                    values, direction, limits = get_plot_data(
                        h5file, mesh, quantity, self.frame
                    )
                    if self.shading == "flat":
                        # https://stackoverflow.com/questions/40492511/set-array-in-tripcolor-bug
                        values = values[mesh.elements].mean(axis=1)
                    collection.set_array(values)
                    vmins[i] = min(vmins[i], limits[0])
                    vmaxs[i] = max(vmaxs[i], limits[1])
                    collection.set_clim(vmins[i], vmaxs[i])
                fig.canvas.draw()

            # Temp data to use in plots
            temp_value = np.ones_like(x)
            temp_value[0] = 0
            temp_value[1] = 0.5

            fig, axes = auto_grid(
                self.num_plots, max_cols=self.max_cols, **self.figure_kwargs
            )
            fig.canvas.mpl_connect("key_press_event", on_keypress)

            collections = []
            for quantity, ax in zip(self.quantities, axes.flat):
                ax: plt.Axes
                opts = PLOT_DEFAULTS[quantity]
                collection = ax.tripcolor(
                    x,
                    y,
                    temp_value,
                    triangles=mesh.elements,
                    shading=self.shading,
                    cmap=opts.cmap,
                )
                cbar = fig.colorbar(collection, ax=ax)
                cbar.set_label(opts.clabel)
                ax.set_aspect("equal")
                ax.set_title(quantity.value)
                collections.append(collection)
                ax.set_xlim(self.xlim)
                ax.set_ylim(self.ylim)
                if self.axis_labels:
                    ax.set_xlabel(f"$x$ [${units_str}$]")
                    ax.set_ylabel(f"$y$ [${units_str}$]")

            draw()
            plt.show()
