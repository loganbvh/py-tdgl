import datetime
import logging
from os import getcwd, path
from typing import Any, Dict, Optional, Sequence

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from ..enums import Observable
from ..mesh.mesh import Mesh
from .defaults import PLOT_DEFAULTS
from .helpers import auto_grid, get_data_range, get_plot_data, get_state_string

_default_observables = (
    "complex_field",
    "phase",
    "supercurrent",
    "vorticity",
)


class InteractivePlot:
    def __init__(
        self,
        input_file: str,
        enable_save: Optional[bool] = False,
        logger: logging.Logger = None,
    ):

        self.input_file = path.join(getcwd(), input_file)
        self.frame = 0
        self.observable = Observable.COMPLEX_FIELD
        self.hide_quiver = False
        self.enable_save = enable_save
        self.logger = logger if logger is not None else logging.getLogger()

    def show(self):
        # Open the file
        with h5py.File(self.input_file, "r", libver="latest") as h5file:

            # Get the mesh
            if "mesh" in h5file:
                mesh = Mesh.load_from_hdf5(h5file["mesh"])
            else:
                mesh = Mesh.load_from_hdf5(h5file["solution/device/mesh"])

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
                    self.observable = Observable.COMPLEX_FIELD

                elif event.key == "2":
                    self.observable = Observable.PHASE

                elif event.key == "3":
                    self.observable = Observable.SUPERCURRENT

                elif event.key == "4":
                    self.observable = Observable.NORMAL_CURRENT

                elif event.key == "5":
                    self.observable = Observable.SCALAR_POTENTIAL

                elif event.key == "6":
                    self.observable = Observable.TOTAL_VECTOR_POTENTIAL

                elif event.key == "7":
                    self.observable = Observable.APPLIED_VECTOR_POTENTIAL

                elif event.key == "8":
                    self.observable = Observable.INDUCED_VECTOR_POTENTIAL

                elif event.key == "9":
                    self.observable = Observable.ALPHA

                elif event.key == "0":
                    self.observable = Observable.VORTICITY

                elif event.key == "w" and self.enable_save:
                    file_name = f"data-{datetime.datetime.now()}.npz"
                    value, direction, limits = get_plot_data(
                        h5file, mesh, self.observable, self.frame
                    )
                    np.savez(
                        file_name,
                        value=value,
                        limits=limits,
                        x=mesh.x,
                        y=mesh.y,
                        elements=mesh.elements,
                    )
                    self.logger.info(f"Saved data to file {file_name}.")

                redraw()

            def redraw():
                value, direction, limits = get_plot_data(
                    h5file, mesh, self.observable, self.frame
                )
                state = get_state_string(h5file, self.frame, max_frame)

                fig.suptitle(f"{self.observable.value}\n{state}")
                triplot.set_array(value)
                triplot.set_clim(*limits)
                triplot.set_cmap(PLOT_DEFAULTS[self.observable].cmap)
                cbar.set_label(PLOT_DEFAULTS[self.observable].clabel)
                # quiver.set_UVC(direction[:, 0], direction[:, 1])
                fig.canvas.draw()

            # Temp data to use in plots
            temp_value = np.ones_like(mesh.x)
            temp_value[0] = 0
            temp_value[1] = 0.5

            fig, ax = plt.subplots()
            fig.subplots_adjust(top=0.8)
            fig.canvas.mpl_connect("key_press_event", on_keypress)
            triplot = ax.tripcolor(
                mesh.x, mesh.y, temp_value, triangles=mesh.elements, shading="gouraud"
            )
            # quiver = ax.quiver(
            #     mesh.x, mesh.y, temp_value, temp_value, scale=0.05, units="dots"
            # )
            cbar = fig.colorbar(triplot)
            ax.set_aspect("equal")
            redraw()
            plt.show()


class MultiInteractivePlot:
    def __init__(
        self,
        input_file: str,
        observables: Sequence[str] = _default_observables,
        max_cols: int = 4,
        logger: logging.Logger = None,
        figure_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.input_file = path.join(getcwd(), input_file)
        self.frame = 0
        if observables is None:
            observables = Observable.get_keys()
        self.observables = [Observable.from_key(name) for name in observables]
        self.num_plots = len(observables)
        self.max_cols = max_cols
        self.hide_quiver = False
        self.logger = logger if logger is not None else logging.getLogger()
        self.figure_kwargs = figure_kwargs or dict()
        self.figure_kwargs.setdefault("constrained_layout", True)
        default_figsize = (
            3.25 * min(self.max_cols, self.num_plots),
            3 * max(1, self.num_plots // self.max_cols),
        )
        self.figure_kwargs.setdefault("figsize", default_figsize)

    def show(self):
        # Open the file
        with h5py.File(self.input_file, "r", libver="latest") as h5file:

            # Get the mesh
            if "mesh" in h5file:
                mesh = Mesh.load_from_hdf5(h5file["mesh"])
            else:
                mesh = Mesh.load_from_hdf5(h5file["solution/device/mesh"])

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

                redraw()

            def redraw():
                state = get_state_string(h5file, self.frame, max_frame)
                fig.suptitle(state)
                for observable, collection in zip(self.observables, collections):
                    value, direction, limits = get_plot_data(
                        h5file, mesh, observable, self.frame
                    )
                    collection.set_array(value)
                    collection.set_clim(*limits)
                # quiver.set_UVC(direction[:, 0], direction[:, 1])
                fig.canvas.draw()

            # Temp data to use in plots
            temp_value = np.ones_like(mesh.x)
            temp_value[0] = 0
            temp_value[1] = 0.5

            fig, axes = auto_grid(
                self.num_plots, max_cols=self.max_cols, **self.figure_kwargs
            )
            fig.canvas.mpl_connect("key_press_event", on_keypress)

            collections = []

            for observable, ax in zip(self.observables, axes.flat):
                opts = PLOT_DEFAULTS[observable]
                collection = ax.tripcolor(
                    mesh.x,
                    mesh.y,
                    temp_value,
                    triangles=mesh.elements,
                    shading="gouraud",
                    cmap=opts.cmap,
                )
                # quiver = ax.quiver(
                #     mesh.x, mesh.y, temp_value, temp_value, scale=0.05, units="dots"
                # )
                cbar = fig.colorbar(
                    collection, ax=ax, format=FuncFormatter("{:.2f}".format)
                )
                cbar.set_label(opts.clabel)
                ax.set_aspect("equal")
                ax.set_title(observable.value)
                collections.append(collection)

            redraw()
            plt.show()
