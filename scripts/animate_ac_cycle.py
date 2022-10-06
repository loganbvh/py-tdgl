import glob
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from tdgl._core.enums import Observable
from tdgl._core.mesh.mesh import Mesh
from tdgl._core.visualization.defaults import PLOT_DEFAULTS
from tdgl._core.visualization.helpers import (
    get_data_range,
    get_plot_data,
    load_state_data,
)
from tdgl._core.visualization.interactive_plot import _default_observables

logger = logging.getLogger(os.path.basename(__file__).replace(".py", ""))


def get_output_h5_files(dirname: str) -> List[str]:
    h5_files = glob.glob(os.path.expandvars(os.path.join(dirname, "output-*.h5")))
    return sorted(
        h5_files, key=lambda s: int(os.path.basename(s).split("-")[-1].split(".")[0])
    )


def load_mesh(h5_path) -> Mesh:
    with h5py.File(h5_path, "r", libver="latest") as f:
        if "mesh" in f:
            if "mesh" in f["mesh"]:
                return Mesh.load_from_hdf5(f["mesh/mesh"])
            return Mesh.load_from_hdf5(f["mesh"])
        return Mesh.load_from_hdf5(f["solution/device/mesh"])


def get_all_h5_groups(h5_files: Sequence[str]) -> List[np.ndarray]:
    groups = []
    for path in h5_files:
        with h5py.File(path, "r", libver="latest") as f:
            groups.append(np.sort(np.array([int(grp) for grp in f["data"]])))
    return groups


def create_animation_dynamic(
    input_dir: str,
    output_file: str,
    fps: int,
    dpi: float,
    observables: Sequence[str] = _default_observables,
    max_cols: int = 4,
    gpu: bool = False,
    silent: bool = False,
    figure_kwargs: Optional[Dict[str, Any]] = None,
    skip_files: int = 1,
):

    # Set codec to h264_nvenc to enable NVIDIA GPU acceleration support
    codec = "h264_nvenc" if gpu else "h264"

    if gpu:
        logger.info("NVIDIA GPU acceleration is enabled.")

    if observables is None:
        observables = Observable.get_keys()
    observables = [Observable.from_key(name) for name in observables]
    num_plots = len(observables)

    figure_kwargs = figure_kwargs or dict()
    # figure_kwargs.setdefault("constrained_layout", True)
    default_figsize = (3.25 * num_plots, 4)
    figure_kwargs.setdefault("figsize", default_figsize)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    h5_files = get_output_h5_files(input_dir)[skip_files:]
    mesh = load_mesh(h5_files[0])

    with h5py.File(h5_files[0], "r", libver="latest") as f:
        I_min, I_max, num_steps = f.attrs["I_fc"]
        rms_I_fcs = np.linspace(I_min, I_max, int(num_steps))
        cycles = f.attrs["cycles"]
        points_per_cycle = int(f.attrs["points_per_cycle"])
        num_points = int(cycles * points_per_cycle)
        thetas = 2 * np.pi * np.linspace(0, cycles, num_points)

    all_groups = get_all_h5_groups(h5_files)
    assert len(all_groups) == len(h5_files)
    paths = np.concatenate(
        [[path] * len(groups) for path, groups in zip(h5_files, all_groups)],
    ).tolist()
    cycle_indices = np.concatenate(
        [list(range(skip_files, num_points)) * len(groups) for groups in all_groups]
    ).tolist()
    all_groups = np.concatenate(all_groups).tolist()
    frames = list(zip(paths, all_groups, cycle_indices, range(len(paths))))
    # min_frame = 0
    max_frame = len(frames) - 1

    # Temp data to use in plots
    temp_value = np.ones_like(mesh.x)
    temp_value[0] = 0
    temp_value[1] = 0.5

    # fig, axes = auto_grid(num_plots, max_cols=max_cols, **figure_kwargs)
    fig = plt.figure(**figure_kwargs)
    gs = fig.add_gridspec(2, num_plots, height_ratios=[1.25, 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(num_plots)]
    bx = fig.add_subplot(gs[1, :])
    collections = []
    for observable, ax in zip(observables, axes):
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
        cbar = fig.colorbar(collection, ax=ax, format=FuncFormatter("{:.2f}".format))
        cbar.set_label(opts.clabel)
        ax.set_aspect("equal")
        ax.set_title(observable.value)
        collections.append(collection)
    bx.plot(thetas / (2 * np.pi), np.cos(thetas), "r-")
    bx.axhline(0, color="k", lw=0.5)
    bx.set_xlabel("$\\omega t / (2\\pi)$")
    bx.set_ylabel("$I_{{FC}}$ [mA]", color="r")
    bx.tick_params(axis="y", colors="r")
    bx.spines["left"].set_color("r")

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    total_time = 0

    vmins = [+np.inf for _ in observables]
    vmaxs = [-np.inf for _ in observables]

    def update(frame):
        nonlocal total_time
        path, group, cycle_index, index = frame
        with h5py.File(path, "r", libver="latest") as f:
            state = load_state_data(f, group)
            total_time += state["dt"]
            I_fc_index = int(f.attrs["index"])
            I_fc_rms = rms_I_fcs[I_fc_index]
            for i, (observable, collection) in enumerate(zip(observables, collections)):
                value, direction, limits = get_plot_data(f, mesh, observable, group)
                vmins[i] = min(vmins[i], limits[0])
                vmaxs[i] = max(vmaxs[i], limits[1])
                collection.set_array(value)
                collection.set_clim(vmins[i], vmaxs[i])
        state_string = (
            f"Frame {index} of {max_frame},"
            f" $I_{{FC,\\,\\mathrm{{rms}}}}$={I_fc_rms:.3f} mA,"
            f"\ndt: {state['dt']:.2e}, time: {total_time:.2e},"
            f"\ngamma: {state['gamma']:.2e}, u: {state['u']:.2e}"
        )
        # quiver.set_UVC(direction[:, 0], direction[:, 1])
        bx.clear()
        bx.plot(
            thetas / (2 * np.pi),
            np.sqrt(2) * I_fc_rms * np.cos(thetas),
            "r-",
        )
        bx.plot(
            thetas / (2 * np.pi),
            np.sqrt(2) * I_fc_rms * np.cos(thetas)[cycle_index],
            "ro",
        )
        bx.axhline(0, color="k", lw=0.5)
        bx.set_xlabel("$\\omega t / (2\\pi)$")
        bx.set_ylabel("$I_{{FC}}$ [mA]", color="r")
        bx.tick_params(axis="y", colors="r")
        bx.spines["left"].set_color("r")
        fig.suptitle(state_string)
        fig.canvas.draw()

    with tqdm(
        total=len(frames),
        unit="frame",
        disable=silent,
    ) as progress:
        ani = FuncAnimation(fig, update, frames=frames, blit=False)
        ani.save(
            output_file,
            fps=fps,
            dpi=dpi,
            codec=codec,
            progress_callback=lambda frame, total: progress.update(1),
        )


def create_animation_steady_state(
    input_file: str,
    output_file: str,
    fps: int,
    dpi: float,
    gpu: bool = False,
    silent: bool = False,
    figure_kwargs: Optional[Dict[str, Any]] = None,
    skip: int = 1,
):
    observables = _default_observables
    observables = [Observable.from_key(name) for name in observables]
    num_plots = len(observables)
    figure_kwargs = figure_kwargs or dict()
    # figure_kwargs.setdefault("constrained_layout", True)
    default_figsize = (3.25 * num_plots, 4)
    figure_kwargs.setdefault("figsize", default_figsize)

    logger.info(f"Creating animation for {[obs.name for obs in observables]!r}.")

    # Set codec to h264_nvenc to enable NVIDIA GPU acceleration support
    codec = "h264_nvenc" if gpu else "h264"

    if gpu:
        logger.info("NVIDIA GPU acceleration is enabled.")

    mesh = load_mesh(input_file)
    with h5py.File(input_file, "r", libver="latest") as h5file:

        I_min, I_max, num_steps = h5file.attrs["I_fc"]
        index = int(h5file.attrs["index"])
        I_fc_rms = np.linspace(I_min, I_max, int(num_steps))[index]
        cycles = h5file.attrs["cycles"]
        points_per_cycle = int(h5file.attrs["points_per_cycle"])
        thetas = 2 * np.pi * np.linspace(0, cycles, int(cycles * points_per_cycle))
        I_fc = np.sqrt(2) * I_fc_rms * np.cos(thetas)

        # Get the ranges for the frame
        min_frame, max_frame = get_data_range(h5file)
        min_frame += skip

        # Temp data to use in plots
        temp_value = np.ones_like(mesh.x)
        temp_value[0] = 0
        temp_value[1] = 0.5

        fig = plt.figure(**figure_kwargs)
        gs = fig.add_gridspec(2, num_plots, height_ratios=[1.25, 1])
        axes = [fig.add_subplot(gs[0, i]) for i in range(num_plots)]
        bx = fig.add_subplot(gs[1, :])
        collections = []
        for observable, ax in zip(observables, axes):
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
        bx.plot(thetas / (2 * np.pi), I_fc, "r-")
        bx.axhline(0, color="k", lw=0.5)
        bx.set_xlabel("$\\omega t / (2\\pi)$")
        bx.set_ylabel("$I_{{FC}}$ [mA]", color="r")
        bx.tick_params(axis="y", colors="r")
        bx.spines["left"].set_color("r")
        bx2 = bx.twinx()
        bx2.set_ylabel("$-\\Phi_\\mathrm{{PL}}$ [m$\\Phi_0$]")

        flux_vals = []
        for i in range(min_frame, max_frame):
            flux_vals.append(h5file[f"data/{i}"].attrs["pl_fluxoid_in_Phi_0"])

        flux = np.array(flux_vals) * 1e3
        fmax = np.abs(flux).max() * 1.1
        fmin = -fmax
        bx2.set_ylim(fmin, fmax)
        bx2.plot(thetas[-len(flux) :] / (2 * np.pi), -flux, "k-")

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        vmins = [+np.inf for _ in observables]
        vmaxs = [-np.inf for _ in observables]

        def update(frame):
            # state = load_state_data(h5file, frame)
            state_string = (
                f"Frame {frame-skip} of {max_frame-min_frame}"
                f", $I_{{FC}}$={I_fc[frame]:.3f} mA"
            )
            fig.suptitle(state_string)
            for i, (observable, collection) in enumerate(zip(observables, collections)):
                value, direction, limits = get_plot_data(
                    h5file, mesh, observable, frame
                )
                vmins[i] = min(vmins[i], limits[0])
                vmaxs[i] = max(vmaxs[i], limits[1])
                collection.set_array(value)
                collection.set_clim(vmins[i], vmaxs[i])
            # quiver.set_UVC(direction[:, 0], direction[:, 1])
            bx.plot(thetas[frame] / (2 * np.pi), I_fc[frame], "ro")
            bx2.plot(thetas[frame] / (2 * np.pi), -flux[frame - skip], "ks")
            fig.canvas.draw()

        frames = range(min_frame, max_frame)
        with tqdm(
            total=len(frames),
            unit="frames",
            disable=silent,
        ) as progress:
            ani = FuncAnimation(fig, update, frames=frames, blit=False)
            ani.save(
                output_file,
                fps=fps,
                dpi=dpi,
                codec=codec,
                progress_callback=lambda frame, total: progress.update(1),
            )


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file or directory.")
    parser.add_argument("-o", "--output", help="Output file.")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument(
        "-d", "--dpi", type=float, default=200, help="Resolution: dots per inch."
    )
    parser.add_argument(
        "-s", "--skip", type=int, default=1, help="Number of files to skip."
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="Enable GPU encoding.")
    parser.add_argument(
        "--dynamic", action="store_true", help="Run create_animation_dynamic()."
    )

    args = parser.parse_args()

    if args.dynamic:
        create_animation_dynamic(
            args.input,
            args.output,
            args.fps,
            args.dpi,
            gpu=args.gpu,
            skip_files=args.skip,
        )
    else:
        if os.path.isdir(args.input):
            subdirs = []
            for name in os.listdir(args.input):
                try:
                    subdirs.append(int(name))
                except ValueError:
                    pass
            subdirs = list(map(str, sorted(subdirs)))
            for subdir in subdirs:
                path = os.path.join(args.input, subdir)
                create_animation_steady_state(
                    os.path.join(path, "steady-state.h5"),
                    os.path.join(path, args.output.format(n=subdir)),
                    args.fps,
                    args.dpi,
                    gpu=args.gpu,
                    skip=args.skip,
                )
        else:
            create_animation_steady_state(
                args.input,
                args.output,
                args.fps,
                args.dpi,
                gpu=args.gpu,
                skip=args.skip,
            )


if __name__ == "__main__":
    main()
