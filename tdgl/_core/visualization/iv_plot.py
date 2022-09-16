import logging
from os import getcwd, path, listdir
from os.path import isfile
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .helpers import get_mean_voltage


class IvPlot:
    def __init__(
        self,
        input_path: str,
        output_file: Optional[str] = None,
        output_format: str = "pdf",
        marker_size: int = 5,
        logger: logging.Logger = None,
        save: Optional[str] = None,
    ):
        self.input_path = path.join(getcwd(), input_path)
        self.output_file = (
            path.join(getcwd(), output_file) if output_file is not None else None
        )
        self.output_format = output_format
        self.marker_size = marker_size
        self.logger = logger if logger is not None else logging.getLogger()
        self.save = save

    def show(self):

        files = []

        if Path(self.input_path).is_file():
            files.append(self.input_path)
        else:
            files = [
                path.join(self.input_path, f)
                for f in listdir(self.input_path)
                if isfile(path.join(self.input_path, f))
            ]

        for f in tqdm(files):
            current, voltage = get_mean_voltage(f)

            plt.plot(current, voltage, ".", markersize=self.marker_size)
            plt.plot(current, np.zeros_like(current), "--")
            plt.xlabel("Current density at terminals [a.u.]")
            plt.ylabel("Voltage [a.u.]")

            if self.save:
                if len(files) == 1:
                    np.savez(self.save, current=current, voltage=voltage)
                else:
                    path_head, path_tail = path.split(f)
                    path_tail_name, path_tail_ext = path.splitext(path_tail)
                    np.savez(
                        path.join(self.save, "{}.npz".format(path_tail_name)),
                        current=current,
                        voltage=voltage,
                    )

            elif self.output_file:

                if len(files) == 1:
                    plt.savefig(self.output_file)
                else:
                    path_head, path_tail = path.split(f)
                    path_tail_name, path_tail_ext = path.splitext(path_tail)
                    plt.savefig(
                        path.join(
                            self.output_file,
                            "{}.{}".format(path_tail_name, self.output_format),
                        )
                    )

            else:
                plt.show()

            plt.clf()
