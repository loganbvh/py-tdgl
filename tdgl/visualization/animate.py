import logging
from logging import Logger
from os import getcwd, path
from typing import Any, Dict, Optional, Sequence

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from ..enums import Observable
from ..finite_volume.mesh import Mesh
from .defaults import PLOT_DEFAULTS
from .helpers import auto_grid, get_data_range, get_plot_data, get_state_string
from .interactive_plot import _default_observables


class Animate:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        observable: Observable,
        fps: int,
        dpi: float,
        skip: int = 0,
        gpu: bool = False,
        logger: Optional[Logger] = None,
        silent: bool = False,
    ):

        self.input_file = path.join(getcwd(), input_file)
        self.output_file = path.join(getcwd(), output_file)
        self.observable = observable
        self.fps = fps
        self.dpi = dpi
        self.skip = skip
        self.gpu = gpu
        self.silent = silent
        self.logger = logger if logger is not None else logging.getLogger()

    def build(self):

        self.logger.info(f"Creating animation for {self.observable.value}.")

        # Set codec to h264_nvenc to enable NVIDIA GPU acceleration support
        codec = "h264_nvenc" if self.gpu else "h264"

        if self.gpu:
            self.logger.info("NVIDIA GPU acceleration is enabled.")

        # Open the file
        with h5py.File(self.input_file, "r", libver="latest") as h5file:

            # Get the mesh
            if "mesh" in h5file:
                mesh = Mesh.load_from_hdf5(h5file["mesh"])
            else:
                mesh = Mesh.load_from_hdf5(h5file["solution/device/mesh"])

            # Get the ranges for the frame
            min_frame, max_frame = get_data_range(h5file)
            min_frame += self.skip

            # Temp data to use in plots
            temp_value = np.ones_like(mesh.x)
            temp_value[0] = 0
            temp_value[1] = 0.5

            fig, ax = plt.subplots()
            fig.subplots_adjust(top=0.8)
            triplot = ax.tripcolor(
                mesh.x,
                mesh.y,
                temp_value,
                triangles=mesh.elements,
                cmap=PLOT_DEFAULTS[self.observable].cmap,
                shading="gouraud",
            )

            cbar = fig.colorbar(triplot)
            cbar.set_label(PLOT_DEFAULTS[self.observable].clabel)
            ax.set_aspect("equal")

            def update(frame):
                value, direction, limits = get_plot_data(
                    h5file, mesh, self.observable, frame
                )
                state = get_state_string(h5file, frame, max_frame)

                ax.set_title(f"{self.observable.value}\n{state}")
                triplot.set_array(value)
                triplot.set_clim(*limits)

                fig.canvas.draw()

            with tqdm(
                total=len(range(min_frame, max_frame)),
                unit="frames",
                disable=self.silent,
            ) as progress:
                ani = FuncAnimation(
                    fig, update, frames=max_frame - min_frame, blit=False
                )
                ani.save(
                    self.output_file,
                    fps=self.fps,
                    dpi=self.dpi,
                    codec=codec,
                    progress_callback=lambda frame, total: progress.update(1),
                )


class MultiAnimate:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        fps: int,
        dpi: float,
        observables: Sequence[str] = _default_observables,
        max_cols: int = 4,
        skip: int = 0,
        gpu: bool = False,
        logger: Optional[Logger] = None,
        silent: bool = False,
        figure_kwargs: Optional[Dict[str, Any]] = None,
    ):

        self.input_file = path.join(getcwd(), input_file)
        self.output_file = path.join(getcwd(), output_file)
        if observables is None:
            observables = Observable.get_keys()
        self.observables = [Observable.from_key(name) for name in observables]
        self.num_plots = len(observables)
        self.max_cols = max_cols
        self.fps = fps
        self.dpi = dpi
        self.skip = skip
        self.gpu = gpu
        self.silent = silent
        self.logger = logger if logger is not None else logging.getLogger()
        self.figure_kwargs = figure_kwargs or dict()
        self.figure_kwargs.setdefault("constrained_layout", True)
        default_figsize = (
            3.25 * min(self.max_cols, self.num_plots),
            3 * max(1, self.num_plots // self.max_cols),
        )
        self.figure_kwargs.setdefault("figsize", default_figsize)

    def build(self):

        self.logger.info(
            f"Creating animation for {[obs.name for obs in self.observables]!r}."
        )

        # Set codec to h264_nvenc to enable NVIDIA GPU acceleration support
        codec = "h264_nvenc" if self.gpu else "h264"

        if self.gpu:
            self.logger.info("NVIDIA GPU acceleration is enabled.")

        # Open the file
        with h5py.File(self.input_file, "r", libver="latest") as h5file:

            # Get the mesh
            if "mesh" in h5file:
                if "mesh" in h5file["mesh"]:
                    mesh = Mesh.load_from_hdf5(h5file["mesh/mesh"])
                else:
                    mesh = Mesh.load_from_hdf5(h5file["mesh"])
            else:
                mesh = Mesh.load_from_hdf5(h5file["solution/device/mesh"])

            # Get the ranges for the frame
            min_frame, max_frame = get_data_range(h5file)
            min_frame += self.skip

            # Temp data to use in plots
            temp_value = np.ones_like(mesh.x)
            temp_value[0] = 0
            temp_value[1] = 0.5

            fig, axes = auto_grid(
                self.num_plots, max_cols=self.max_cols, **self.figure_kwargs
            )
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

            vmins = [+np.inf for _ in self.observables]
            vmaxs = [-np.inf for _ in self.observables]

            def update(frame):
                state = get_state_string(h5file, frame, max_frame)
                fig.suptitle(state)
                for i, (observable, collection) in enumerate(
                    zip(self.observables, collections)
                ):
                    value, direction, limits = get_plot_data(
                        h5file, mesh, observable, frame
                    )
                    vmins[i] = min(vmins[i], limits[0])
                    vmaxs[i] = max(vmaxs[i], limits[1])
                    collection.set_array(value)
                    collection.set_clim(vmins[i], vmaxs[i])
                # quiver.set_UVC(direction[:, 0], direction[:, 1])
                fig.canvas.draw()

            with tqdm(
                total=len(range(min_frame, max_frame)),
                unit="frames",
                disable=self.silent,
            ) as progress:
                ani = FuncAnimation(
                    fig, update, frames=max_frame - min_frame, blit=False
                )
                ani.save(
                    self.output_file,
                    fps=self.fps,
                    dpi=self.dpi,
                    codec=codec,
                    progress_callback=lambda frame, total: progress.update(1),
                )
