import glob
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from tdgl._core.enums import Observable
from tdgl._core.mesh.mesh import Mesh
from tdgl._core.visualization.defaults import PLOT_DEFAULTS
from tdgl._core.visualization.helpers import auto_grid, get_plot_data, get_state_string
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


def create_animation(
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
    figure_kwargs.setdefault("constrained_layout", True)
    default_figsize = (
        3.25 * min(max_cols, num_plots),
        3 * max(1, num_plots // max_cols),
    )
    figure_kwargs.setdefault("figsize", default_figsize)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    h5_files = get_output_h5_files(input_dir)[skip_files:]
    mesh = load_mesh(h5_files[0])

    all_groups = get_all_h5_groups(h5_files)
    assert len(all_groups) == len(h5_files)
    paths = np.concatenate(
        [[path] * len(groups) for path, groups in zip(h5_files, all_groups)],
    ).tolist()
    all_groups = np.concatenate(all_groups).tolist()
    frames = list(zip(paths, all_groups))
    min_frame = 0
    max_frame = len(frames) - 1

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
        # quiver = ax.quiver(
        #     mesh.x, mesh.y, temp_value, temp_value, scale=0.05, units="dots"
        # )
        cbar = fig.colorbar(collection, ax=ax, format=FuncFormatter("{:.2f}".format))
        cbar.set_label(opts.clabel)
        ax.set_aspect("equal")
        ax.set_title(observable.value)
        collections.append(collection)

    def update(frame):
        path, group = frame
        with h5py.File(path, "r", libver="latest") as f:
            state = get_state_string(f, group, max_frame)
            for observable, collection in zip(observables, collections):
                value, direction, limits = get_plot_data(f, mesh, observable, group)
                collection.set_array(value)
                if frame == min_frame:
                    collection.set_clim(*limits)
        # quiver.set_UVC(direction[:, 0], direction[:, 1])
        fig.suptitle(state)
        fig.canvas.draw()

    with tqdm(
        total=len(range(min_frame, max_frame)),
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
    parser.add_argument("-i", "--input", help="Input directory.")
    parser.add_argument("-o", "--output", help="Output file.")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument(
        "-d", "--dpi", type=float, default=200, help="Resolution: dots per inch."
    )
    parser.add_argument(
        "-s", "--skip", type=int, default=1, help="Number of files to skip."
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="Enable GPU encoding.")

    args = parser.parse_args()

    create_animation(
        args.input,
        args.output,
        args.fps,
        args.dpi,
        gpu=args.gpu,
        skip_files=args.skip,
    )


if __name__ == "__main__":
    main()
