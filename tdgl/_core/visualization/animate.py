import logging
from logging import Logger
from os import getcwd, path
from typing import Optional

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from ..mesh.mesh import Mesh
from ..enums import Observable
from .helpers import (
    get_data_range,
    get_plot_data,
    get_state_string,
)


class Animate:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        observable: Observable,
        fps: int,
        dpi: float,
        gpu: bool = False,
        logger: Optional[Logger] = None,
        silent: bool = False,
    ):

        self.input_file = path.join(getcwd(), input_file)
        self.output_file = path.join(getcwd(), output_file)
        self.observable = observable
        self.fps = fps
        self.dpi = dpi
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
        with h5py.File(self.input_file, "r") as h5file:

            # Get the mesh
            mesh = Mesh.load_from_hdf5(h5file["mesh"])

            # Get the ranges for the frame
            min_frame, max_frame = get_data_range(h5file)

            # Temp data to use in plots
            temp_value = np.ones_like(mesh.x)
            temp_value[0] = 0
            temp_value[1] = 0.5

            fig, ax = plt.subplots()
            triplot = ax.tripcolor(
                mesh.x, mesh.y, temp_value, triangles=mesh.elements, shading="gouraud"
            )

            fig.colorbar(triplot)
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
