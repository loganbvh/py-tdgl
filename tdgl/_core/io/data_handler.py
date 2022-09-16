import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
from os import path, getcwd

import h5py
import numpy as np

from ..mesh.mesh import Mesh


class DataHandler:
    """
    The data handler is responsible for reading from and writing to disk.

    Args:
        input_file: File to use as input for the simulation.
        output_file: File to use as output for simulation data.
        logger: Logger used to inform about errors.
    """

    def __init__(
        self,
        input_value: Union[Mesh, str],
        output_file: str,
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(input_value, str):
            with h5py.File(path.join(getcwd(), input_value), "r") as f:
                self.mesh = self.__create_mesh(f)
        else:
            self.mesh = input_value

        self.output_file = None
        self.mesh_group = None
        self.time_step_group = None
        self.save_number = 0
        self.logger = logger if logger is not None else logging.getLogger()

        self.output_file, self.output_path = self.__create_output_file(
            output_file, self.logger
        )
        self.mesh_group = self.output_file.create_group("mesh")
        self.time_step_group = self.output_file.create_group("data")
        self.mesh.save_to_hdf5(self.mesh_group)

    @classmethod
    def __create_output_file(
        cls, output: str, logger: logging.Logger
    ) -> Tuple[h5py.File, str]:
        """Create an output file.

        Args:
            output: The output file path.
            logger: Logger output logs.

        Returns:
            A file handle.
        """

        # Make sure the directory exists
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        # Split the output into file name and suffix
        name_parts = output.split(".")
        name = ".".join(name_parts[:-1])
        suffix = name_parts[-1]

        # Number to be added to the end of file name
        # If this is None do not add the number
        serial_number = None

        while True:

            # Create a new file name
            name_suffix = (
                "-{}".format(serial_number) if serial_number is not None else ""
            )
            file_name = "{}{}.{}".format(name, name_suffix, suffix)
            file_path = path.join(getcwd(), file_name)

            try:
                file = h5py.File(file_path, "x")
            except (OSError, FileExistsError):

                # Increment the serial number if the file could not be created
                # and try again
                if serial_number is None:
                    serial_number = 1
                else:
                    serial_number += 1

                continue

            else:

                # Inform the user about the name change
                if serial_number is not None:
                    logger.warning(
                        "Output file already exists. Renaming to {}.".format(file_name)
                    )

            return file, file_path

    @classmethod
    def __get_save_number_stored(cls, h5group: h5py.Group) -> int:
        keys = np.asarray(list(int(key) for key in h5group))
        return np.max(keys)

    @classmethod
    def __create_mesh(cls, input_file: h5py.File) -> Mesh:
        return Mesh.load_from_hdf5(input_file)

    def close(self):
        self.output_file.close()

    def get_last_step(self) -> h5py.Group:
        last_save_number = self.__get_save_number_stored(self.time_step_group)
        return self.time_step_group["{}".format(last_save_number)]

    def get_mesh(self) -> Mesh:
        return self.mesh

    def get_voltage_points(self) -> np.ndarray:
        return self.mesh.voltage_points

    def save_time_step(self, params: Dict[str, float], data: Dict[str, np.ndarray]):
        group = self.time_step_group.create_group("{}".format(self.save_number))
        self.save_number += 1

        # Set an attribute to specify for which values this data was recorded
        for key, value in params.items():
            group.attrs[key] = value

        # Save the data
        for key, value in data.items():
            group[key] = value
