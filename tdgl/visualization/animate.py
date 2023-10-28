import logging
import os
from contextlib import nullcontext
from logging import Logger
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..device.device import Device
from ..solution.data import get_data_range
from .common import DEFAULT_QUANTITIES, PLOT_DEFAULTS, Quantity, auto_grid
from .io import get_plot_data, get_state_string


def create_animation(
    input_file: Union[str, h5py.File],
    *,
    output_file: Optional[str] = None,
    quantities: Union[str, Sequence[str]] = DEFAULT_QUANTITIES,
    shading: Literal["flat", "gouraud"] = "gouraud",
    fps: int = 30,
    dpi: float = 100,
    max_cols: int = 4,
    min_frame: int = 0,
    max_frame: int = -1,
    autoscale: bool = False,
    dimensionless: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    axis_labels: bool = False,
    axes_off: bool = False,
    title_off: bool = False,
    full_title: bool = True,
    logger: Optional[Logger] = None,
    figure_kwargs: Optional[Dict[str, Any]] = None,
    writer: Union[str, animation.MovieWriter, None] = None,
) -> animation.FuncAnimation:
    """Generates, and optionally saves, and animation of a TDGL simulation.

    Args:
        input_file: An open h5py file or a path to an H5 file containing
            the :class:`tdgl.Solution` you would like to animate.
        output_file: A path to which to save the animation,
            e.g., as a gif or mp4 video.
        quantities: The names of the quantities to animate.
        shading: Shading method, "flat" or "gouraud". See matplotlib.pyplot.tripcolor.
        fps: Frame rate in frames per second.
        dpi: Resolution in dots per inch.
        max_cols: The maxiumum number of columns in the subplot grid.
        min_frame: The first frame of the animation.
        max_frame: The last frame of the animation.
        autoscale: Autoscale colorbar limits at each frame.
        dimensionless: Use dimensionless units for axes
        xlim: x-axis limits
        ylim: y-axis limits
        axes_off: Turn off the axes for each subplot.
        title_off: Turn off the figure suptitle.
        full_title: Include the full "state" for each frame in the figure suptitle.
        figure_kwargs: Keyword arguments passed to ``plt.subplots()`` when creating
            the figure.
        writer: A :class:`matplotlib.animation.MovieWriter` instance to use when
            saving the animation.
        logger: A logger instance to use.

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
        2.5 * max(1, num_plots // max_cols),
    )
    figure_kwargs.setdefault("figsize", default_figsize)
    figure_kwargs.setdefault("sharex", True)
    figure_kwargs.setdefault("sharey", True)

    logger.info(f"Creating animation for {[obs.name for obs in quantities]!r}.")

    mpl_context = nullcontext() if output_file is None else plt.ioff()
    if isinstance(input_file, str):
        h5_context = h5py.File(input_file, "r")
    else:
        h5_context = nullcontext(input_file)

    with h5_context as h5file:
        with mpl_context:
            device = Device.from_hdf5(h5file["solution/device"])
            mesh = device.mesh
            if dimensionless:
                scale = 1
                units_str = "\\xi"
            else:
                scale = device.layer.coherence_length
                units_str = f"{device.ureg(device.length_units).units:~L}"
            x, y = scale * mesh.sites.T

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
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                if axes_off:
                    ax.axis("off")
                if axis_labels:
                    ax.set_xlabel(f"$x$ [${units_str}$]")
                    ax.set_ylabel(f"$y$ [${units_str}$]")
                collections.append(collection)

            vmins = [+np.inf for _ in quantities]
            vmaxs = [-np.inf for _ in quantities]

            def update(frame):
                if not h5file:
                    return
                frame += min_frame
                state = get_state_string(h5file, frame, max_frame)
                if not full_title:
                    state = state.split(",")[0]
                if not title_off:
                    fig.suptitle(state)
                for i, (quantity, collection) in enumerate(
                    zip(quantities, collections)
                ):
                    opts = PLOT_DEFAULTS[quantity]
                    values, direction, _ = get_plot_data(h5file, mesh, quantity, frame)
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
                desc=f"Saving to {fname}",
            ) as pbar:
                anim.save(
                    output_file,
                    dpi=dpi,
                    progress_callback=lambda frame, total: pbar.update(1),
                    **kwargs,
                )

        return anim
