import logging
from os import getcwd, listdir, path
from os.path import isfile
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .helpers import get_mean_voltage


class IcDist:
    def __init__(
        self,
        threshold: float,
        input_path: str,
        output_file: Optional[str] = None,
        data_file: Optional[str] = None,
        output_format: str = "pdf",
        logger: logging.Logger = None,
        bins: int = 10,
        normalized: bool = False,
    ):
        self.threshold = threshold
        self.input_path = path.join(getcwd(), input_path)
        self.output_file = (
            path.join(getcwd(), output_file) if output_file is not None else None
        )
        self.output_format = output_format
        self.data_file = data_file
        self.logger = logger if logger is not None else logging.getLogger()
        self.bins = bins
        self.normalized = normalized

    def show(self):

        # Get files
        files = [
            path.join(self.input_path, f)
            for f in listdir(self.input_path)
            if isfile(path.join(self.input_path, f))
        ]

        # Create arrays to store data
        critical_current = np.zeros(len(files))

        for i, f in enumerate(tqdm(files)):

            try:
                current, voltage = get_mean_voltage(f)
            except OSError:
                logging.error(f"Could not parse {f}")
                continue

            # Find the first current with voltage larger than the threshold
            # This current corresponds to the critical current
            critical_current_index = np.argmax(voltage > self.threshold)
            critical_current[i] = current[critical_current_index]

        if self.data_file is not None:
            self.logger.info(
                f"Saving Ic dist data to {self.data_file} in numpy npz format."
            )
            np.savez(self.data_file, critical_current=critical_current)

        plt.hist(critical_current, bins=self.bins, density=self.normalized)
        plt.xlabel("Critical current")

        if self.output_file:
            plt.savefig(self.output_file)
        else:
            plt.show()
