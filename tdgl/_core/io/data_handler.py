import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

import h5py
import numpy as np

from ..mesh.mesh import Mesh


class DataHandler:
    """
    The data handler is responsible for reading from and writing to disk.

    Args:
        input_file: File to use as input for the simulation.
        output_file: File to use as output for simulation data.
        save_mesh: Whether to save the mesh.
        logger: Logger used to inform about errors.
    """

    def __init__(
        self,
        input_value: Union[Mesh, str],
        output_file: str,
        save_mesh: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(input_value, str):
            with h5py.File(os.path.join(os.getcwd(), input_value), "r") as f:
                self.mesh = Mesh.load_from_hdf5(f)
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
        self.time_step_group = self.output_file.create_group("data")
        if save_mesh:
            self.mesh_group = self.output_file.create_group("mesh")
            self.mesh.save_to_hdf5(self.mesh_group)
        else:
            self.mesh_group = None

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
            name_suffix = f"-{serial_number}" if serial_number is not None else ""
            file_name = f"{name}{name_suffix}.{suffix}"
            file_path = os.path.join(os.getcwd(), file_name)

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
                        f"Output file already exists. Renaming to {file_name}."
                    )
            return file, file_path

    @classmethod
    def __get_save_number_stored(cls, h5group: h5py.Group) -> int:
        keys = np.asarray(list(int(key) for key in h5group))
        return np.max(keys)

    def close(self):
        self.output_file.close()

    def get_last_step(self) -> h5py.Group:
        last_save_number = self.__get_save_number_stored(self.time_step_group)
        return self.time_step_group[f"{last_save_number}"]

    def get_mesh(self) -> Mesh:
        return self.mesh

    def get_voltage_points(self) -> np.ndarray:
        return self.mesh.voltage_points

    def save_time_step(self, state: Dict[str, float], data: Dict[str, np.ndarray]):
        group = self.time_step_group.create_group(f"{self.save_number}")
        group.attrs["timestamp"] = datetime.now().isoformat()
        self.save_number += 1
        # Set an attribute to specify for which values this data was recorded
        for key, value in state.items():
            group.attrs[key] = value
        # Save the data
        for key, value in data.items():
            group[key] = value
