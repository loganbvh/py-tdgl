import logging
from typing import Any, Dict, Sequence, Union

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from ..finite_volume.mesh import Mesh
from ..solution.data import get_data_range
from .common import DEFAULT_QUANTITIES, PLOT_DEFAULTS, Quantity, auto_grid
from .io import get_plot_data, get_state_string


class InteractivePlot:
    def __init__(
        self,
        input_file: str,
        logger: logging.Logger = None,
    ):

        self.input_file = input_file
        self.frame = 0
        self.quantity = Quantity.ORDER_PARAMETER
        self.logger = logger or logging.getLogger()

    def show(self):
        with h5py.File(self.input_file, "r", libver="latest") as h5file:
            if "mesh" in h5file:
                mesh = Mesh.from_hdf5(h5file["mesh"])
            else:
                mesh = Mesh.from_hdf5(h5file["solution/device/mesh"])

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
                value, direction, limits = get_plot_data(
                    h5file, mesh, self.quantity, self.frame
                )
                state = get_state_string(h5file, self.frame, max_frame)

                fig.suptitle(f"{self.quantity.value}\n{state}")
                triplot.set_array(value)
                triplot.set_clim(*limits)
                triplot.set_cmap(PLOT_DEFAULTS[self.quantity].cmap)
                cbar.set_label(PLOT_DEFAULTS[self.quantity].clabel)
                fig.canvas.draw()

            x, y = mesh.sites.T
            # Temp data to use in plots
            temp_value = np.ones_like(x)
            temp_value[0] = 0
            temp_value[1] = 0.5

            fig, ax = plt.subplots()
            fig.subplots_adjust(top=0.8)
            fig.canvas.mpl_connect("key_press_event", on_keypress)
            triplot = ax.tripcolor(
                x, y, temp_value, triangles=mesh.elements, shading="gouraud"
            )
            cbar = fig.colorbar(triplot)
            ax.set_aspect("equal")
            draw()
            plt.show()


class MultiInteractivePlot:
    def __init__(
        self,
        input_file: str,
        quantities: Sequence[str] = DEFAULT_QUANTITIES,
        max_cols: int = 4,
        logger: logging.Logger = None,
        figure_kwargs: Union[Dict[str, Any], None] = None,
    ):
        self.input_file = input_file
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

    def show(self):
        with h5py.File(self.input_file, "r", libver="latest") as h5file:
            if "mesh" in h5file:
                mesh = Mesh.from_hdf5(h5file["mesh"])
            else:
                mesh = Mesh.from_hdf5(h5file["solution/device/mesh"])

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
                    value, direction, limits = get_plot_data(
                        h5file, mesh, quantity, self.frame
                    )
                    collection.set_array(value)
                    vmins[i] = min(vmins[i], limits[0])
                    vmaxs[i] = max(vmaxs[i], limits[1])
                    collection.set_clim(vmins[i], vmaxs[i])
                fig.canvas.draw()

            x, y = mesh.sites.T
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
                opts = PLOT_DEFAULTS[quantity]
                collection = ax.tripcolor(
                    x,
                    y,
                    temp_value,
                    triangles=mesh.elements,
                    shading="gouraud",
                    cmap=opts.cmap,
                )
                cbar = fig.colorbar(
                    collection, ax=ax, format=FuncFormatter("{:.2f}".format)
                )
                cbar.set_label(opts.clabel)
                ax.set_aspect("equal")
                ax.set_title(quantity.value)
                collections.append(collection)

            draw()
            plt.show()
