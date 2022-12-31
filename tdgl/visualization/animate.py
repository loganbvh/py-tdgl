import logging
import os
from contextlib import nullcontext
from logging import Logger
from typing import Any, Dict, Sequence, Union

import h5py
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from ..finite_volume.mesh import Mesh
from ..solution.data import get_data_range
from .common import PLOT_DEFAULTS, Quantity, auto_grid
from .io import get_plot_data, get_state_string


def create_animation(
    input_file: Union[str, h5py.File],
    *,
    output_file: Union[str, None] = None,
    quantities: Union[str, Sequence[str]],
    fps: int = 30,
    dpi: float = 100,
    max_cols: int = 4,
    min_frame: int = 0,
    max_frame: int = -1,
    quiver: bool = False,
    axes_off: bool = False,
    title_off: bool = False,
    full_title: bool = True,
    logger: Union[Logger, None] = None,
    silent: bool = False,
    figure_kwargs: Union[Dict[str, Any], None] = None,
    writer: Union[str, animation.MovieWriter, None] = None,
) -> animation.FuncAnimation:
    """Generates, and optionally saves, and animation of a TDGL simulation.

    The animation will be in dimensionless units.

    Args:
        input_file: An open h5py file or a path to an H5 file containing
            the :class:`tdgl.Solution` you would like to animate.
        output_file: A path to which to save the animation,
            e.g., as a gif or mp4 video.
        quantities: The names of the quantities to animate.
        fps: Frame rate in frames per second.
        dpi: Resolution in dots per inch.
        max_cols: The maxiumum number of columns in the subplot grid.
        min_frame: The first frame of the animation.
        max_frame: The last frame of the animation.
        quiver: Add quiver arrows to the plots.
        axes_off: Turn off the axes for each subplot.
        title_off: Turn off the figure suptitle.
        full_title: Include the full "state" for each frame in the figure suptitle.
        figure_kwargs: Keyword arguments passed to ``plt.subplots()`` when creating
            the figure.
        writer: A :class:`matplotlib.animation.MovieWriter` instance to use when
            saving the animation.
        logger: A logger instance to use.
        silent: Disable logging.

    Returns:
        The animation as a :class:`matplotlib.animation.FuncAnimation`.
    """
    if isinstance(input_file, str):
        input_file = input_file
    if quantities is None:
        quantities = Quantity.get_keys()
    if isinstance(quantities, str):
        quantities = [quantities]
    quantities = [Quantity.from_key(name.upper()) for name in quantities]
    num_plots = len(quantities)
    logger = logger or logging.getLogger()
    figure_kwargs = figure_kwargs or dict()
    figure_kwargs.setdefault("constrained_layout", True)
    default_figsize = (
        3.25 * min(max_cols, num_plots),
        3 * max(1, num_plots // max_cols),
    )
    figure_kwargs.setdefault("figsize", default_figsize)

    logger.info(f"Creating animation for {[obs.name for obs in quantities]!r}.")

    mpl_context = nullcontext() if output_file is None else plt.ioff()
    if isinstance(input_file, str):
        h5_context = h5py.File(input_file, "r", libver="latest")
    else:
        h5_context = nullcontext(input_file)

    with h5_context as h5file:
        with mpl_context:
            # Get the mesh
            if "mesh" in h5file:
                mesh = Mesh.from_hdf5(h5file["mesh"])
            else:
                mesh = Mesh.from_hdf5(h5file["solution/device/mesh"])

            x, y = mesh.sites.T

            # Get the ranges for the frame
            _min_frame, _max_frame = get_data_range(h5file)
            min_frame = max(min_frame, _min_frame)
            if max_frame == -1:
                max_frame = _max_frame
            else:
                max_frame = min(max_frame, _max_frame)

            # Temp data to use in plots
            temp_value = np.ones(len(mesh.sites), dtype=float)
            temp_value[0] = 0
            temp_value[1] = 0.5

            fig, axes = auto_grid(num_plots, max_cols=max_cols, **figure_kwargs)
            collections = []
            for quantity, ax in zip(quantities, axes.flat):
                opts = PLOT_DEFAULTS[quantity]
                collection = ax.tripcolor(
                    x,
                    y,
                    temp_value,
                    triangles=mesh.elements,
                    shading="gouraud",
                    cmap=opts.cmap,
                )
                if quiver:
                    quiver = ax.quiver(
                        x,
                        y,
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
                ax.set_title(quantity.value)
                if axes_off:
                    ax.axis("off")
                collections.append(collection)

            vmins = [+np.inf for _ in quantities]
            vmaxs = [-np.inf for _ in quantities]

            def update(frame):
                if not h5file:
                    return
                state = get_state_string(h5file, frame, max_frame)
                if not full_title:
                    state = state.split(",")[0]
                if not title_off:
                    fig.suptitle(state)
                for i, (quantity, collection) in enumerate(
                    zip(quantities, collections)
                ):
                    value, direction, limits = get_plot_data(
                        h5file, mesh, quantity, frame
                    )
                    vmins[i] = min(vmins[i], limits[0])
                    vmaxs[i] = max(vmaxs[i], limits[1])
                    collection.set_array(value)
                    collection.set_clim(vmins[i], vmaxs[i])
                if quiver:
                    quiver.set_UVC(direction[:, 0], direction[:, 1])
                fig.canvas.draw()

            anim = animation.FuncAnimation(
                fig,
                update,
                frames=max_frame - min_frame,
                interval=1e3 / fps,
                blit=False,
            )

        if output_file is not None:
            output_file = os.path.join(os.getcwd(), output_file)
            if writer is None:
                kwargs = dict(fps=fps)
            else:
                kwargs = dict(writer=writer)
            fname = os.path.basename(output_file)
            with tqdm(
                total=len(range(min_frame, max_frame)),
                unit="frames",
                disable=silent,
                desc=f"Saving to {fname}",
            ) as pbar:
                anim.save(
                    output_file,
                    dpi=dpi,
                    progress_callback=lambda frame, total: pbar.update(1),
                    **kwargs,
                )

        return anim
