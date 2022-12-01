import logging
import os
from logging import Logger
from typing import Any, Dict, Sequence, Union

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from ..finite_volume.mesh import Mesh
from ..solution.data import get_data_range
from ..solution.plot_solution import auto_grid
from .defaults import PLOT_DEFAULTS, Observable
from .interactive_plot import _default_observables
from .io import get_plot_data, get_state_string


def animate(
    input_file: str,
    *,
    output_file: str,
    observable: Union[Observable, str],
    fps: int,
    dpi: float,
    skip: int = 0,
    gpu: bool = False,
    logger: Union[Logger, None] = None,
    silent: bool = False,
):
    input_file = os.path.join(os.getcwd(), input_file)
    output_file = os.path.join(os.getcwd(), output_file)
    logger = logger or logging.getLogger()
    if isinstance(observable, str):
        observable = Observable.from_key(observable.upper())
    logger.info(f"Creating animation for {observable.value}.")
    # Set codec to h264_nvenc to enable NVIDIA GPU acceleration support
    codec = "h264_nvenc" if gpu else "h264"
    if gpu:
        logger.info("NVIDIA GPU acceleration is enabled.")

    with h5py.File(input_file, "r", libver="latest") as h5file:
        with plt.ioff():
            # Get the mesh
            if "mesh" in h5file:
                mesh = Mesh.load_from_hdf5(h5file["mesh"])
            else:
                mesh = Mesh.load_from_hdf5(h5file["solution/device/mesh"])
            # Get the ranges for the frame
            min_frame, max_frame = get_data_range(h5file)
            min_frame += skip
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
                cmap=PLOT_DEFAULTS[observable].cmap,
                shading="gouraud",
            )
            cbar = fig.colorbar(triplot)
            cbar.set_label(PLOT_DEFAULTS[observable].clabel)
            ax.set_aspect("equal")

            def update(frame):
                if not h5file:
                    return
                value, direction, limits = get_plot_data(
                    h5file, mesh, observable, frame
                )
                state = get_state_string(h5file, frame, max_frame)

                ax.set_title(f"{observable.value}\n{state}")
                triplot.set_array(value)
                triplot.set_clim(*limits)

                fig.canvas.draw()

            with tqdm(
                total=len(range(min_frame, max_frame)),
                unit="frames",
                disable=silent,
            ) as progress:
                ani = FuncAnimation(
                    fig, update, frames=max_frame - min_frame, blit=False
                )
                ani.save(
                    output_file,
                    fps=fps,
                    dpi=dpi,
                    codec=codec,
                    progress_callback=lambda frame, total: progress.update(1),
                )


def multi_animate(
    input_file: str,
    *,
    output_file: str,
    fps: int,
    dpi: float,
    observables: Sequence[str] = _default_observables,
    max_cols: int = 4,
    skip: int = 0,
    gpu: bool = False,
    quiver: bool = False,
    full_title: bool = True,
    logger: Union[Logger, None] = None,
    silent: bool = False,
    figure_kwargs: Union[Dict[str, Any], None] = None,
):
    input_file = os.path.join(os.getcwd(), input_file)
    output_file = os.path.join(os.getcwd(), output_file)
    if observables is None:
        observables = Observable.get_keys()
    observables = [Observable.from_key(name.upper()) for name in observables]
    num_plots = len(observables)
    logger = logger or logging.getLogger()
    figure_kwargs = figure_kwargs or dict()
    figure_kwargs.setdefault("constrained_layout", True)
    default_figsize = (
        3.25 * min(max_cols, num_plots),
        3 * max(1, num_plots // max_cols),
    )
    figure_kwargs.setdefault("figsize", default_figsize)

    logger.info(f"Creating animation for {[obs.name for obs in observables]!r}.")

    # Set codec to h264_nvenc to enable NVIDIA GPU acceleration support
    codec = "h264_nvenc" if gpu else "h264"
    if gpu:
        logger.info("NVIDIA GPU acceleration is enabled.")

    with h5py.File(input_file, "r", libver="latest") as h5file:
        with plt.ioff():
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
            min_frame += skip

            # Temp data to use in plots
            temp_value = np.ones_like(mesh.x)
            temp_value[0] = 0
            temp_value[1] = 0.5

            fig, axes = auto_grid(num_plots, max_cols=max_cols, **figure_kwargs)
            collections = []
            for observable, ax in zip(observables, axes.flat):
                opts = PLOT_DEFAULTS[observable]
                collection = ax.tripcolor(
                    mesh.x,
                    mesh.y,
                    temp_value,
                    triangles=mesh.elements,
                    shading="gouraud",
                    cmap=opts.cmap,
                )
                if quiver:
                    quiver = ax.quiver(
                        mesh.x,
                        mesh.y,
                        temp_value,
                        temp_value,
                        scale=0.05,
                        units="dots",
                    )
                cbar = fig.colorbar(
                    collection, ax=ax, format=FuncFormatter("{:.2f}".format)
                )
                cbar.set_label(opts.clabel)
                ax.set_aspect("equal")
                ax.set_title(observable.value)
                collections.append(collection)

            vmins = [+np.inf for _ in observables]
            vmaxs = [-np.inf for _ in observables]

            def update(frame):
                if not h5file:
                    return
                state = get_state_string(h5file, frame, max_frame)
                if not full_title:
                    state = state.split(",")[0]
                fig.suptitle(state)
                for i, (observable, collection) in enumerate(
                    zip(observables, collections)
                ):
                    value, direction, limits = get_plot_data(
                        h5file, mesh, observable, frame
                    )
                    vmins[i] = min(vmins[i], limits[0])
                    vmaxs[i] = max(vmaxs[i], limits[1])
                    collection.set_array(value)
                    collection.set_clim(vmins[i], vmaxs[i])
                if quiver:
                    quiver.set_UVC(direction[:, 0], direction[:, 1])
                fig.canvas.draw()

            with tqdm(
                total=len(range(min_frame, max_frame)),
                unit="frames",
                disable=silent,
            ) as progress:
                ani = FuncAnimation(
                    fig, update, frames=max_frame - min_frame, blit=False
                )
                ani.save(
                    output_file,
                    fps=fps,
                    dpi=dpi,
                    codec=codec,
                    progress_callback=lambda frame, total: progress.update(1),
                )
