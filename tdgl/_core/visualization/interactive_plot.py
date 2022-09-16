import datetime
import logging
from os import getcwd, path
from typing import Optional

import h5py
import numpy as np
from matplotlib import pyplot as plt

from ..mesh.mesh import Mesh
from ..enums import Observable
from .helpers import (
    get_data_range,
    get_plot_data,
    get_state_string,
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
        with h5py.File(self.input_file, "r") as h5file:

            # Get the mesh
            mesh = Mesh.load_from_hdf5(h5file["mesh"])

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
                    self.observable = Observable.VECTOR_POTENTIAL

                elif event.key == "7":
                    self.observable = Observable.ALPHA

                elif event.key == "8":
                    self.observable = Observable.VORTICITY

                elif event.key == "w" and self.enable_save:
                    file_name = "data-{}.npz".format(datetime.datetime.now())
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
                    self.logger.info("Saved data to file {}.".format(file_name))

                redraw()

            def redraw():
                value, direction, limits = get_plot_data(
                    h5file, mesh, self.observable, self.frame
                )
                state = get_state_string(h5file, self.frame, max_frame)

                ax.set_title("{}\n{}".format(self.observable.value, state))
                triplot.set_array(value)
                triplot.set_clim(*limits)
                quiver.set_UVC(direction[:, 0], direction[:, 1])
                fig.canvas.draw()

            # Temp data to use in plots
            temp_value = np.ones_like(mesh.x)
            temp_value[0] = 0
            temp_value[1] = 0.5

            fig, ax = plt.subplots()
            fig.canvas.mpl_connect("key_press_event", on_keypress)
            triplot = ax.tripcolor(
                mesh.x, mesh.y, temp_value, triangles=mesh.elements, shading="gouraud"
            )
            quiver = ax.quiver(
                mesh.x, mesh.y, temp_value, temp_value, scale=0.1, units="dots"
            )
            fig.colorbar(triplot)
            ax.set_aspect("equal")
            redraw()
            plt.show()
