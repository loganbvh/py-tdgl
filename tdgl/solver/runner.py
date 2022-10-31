import itertools
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm

from ..finite_volume.mesh import Mesh
from .options import SolverOptions


class DataHandler:
    """The data handler is responsible for reading from and writing to disk.

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
            path = os.path.join(os.getcwd(), input_value)
            with h5py.File(path, "r", libver="latest") as f:
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
                file = h5py.File(file_path, "x", libver="latest")
            except (OSError, FileExistsError):
                if serial_number is None:
                    serial_number = 1
                else:
                    serial_number += 1
                continue
            else:
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


class RunningState:
    """Storage class for saving data that should be saved each time step.
    Used for IV curves or similar.

    Args:
        names: Names of the parameters to be saved.
        buffer: Size of the buffer.
    """

    def __init__(self, names: Sequence[str], buffer: int):
        self.step = 0
        self.buffer = buffer
        self.values = dict((name, np.zeros(buffer)) for name in names)

    def next(self) -> None:
        """Go to the next step."""
        self.step += 1

    def set_step(self, step: int) -> None:
        """Set the current step.

        Args:
            step: Step to go to.
        """
        self.step = step

    def clear(self) -> None:
        """Clear the buffer."""

        self.step = 0
        for key in self.values:
            self.values[key] = np.zeros(self.buffer)

    def append(self, name: str, value: Union[float, int]) -> None:
        """Append data to the buffer.

        Args:
            name: Data to append.
            value: Value of the data.
        """

        self.values[name][self.step] = value

    def export(self) -> Dict[str, np.ndarray]:
        """Export data to save to disk.

        Returns:
            A dict with the data.
        """

        return self.values


class Runner:
    """The runner is responsible for the simulation loop. It handles the state,
    runs the update function and saves the data.

    Args:
        function: The update function that takes the state from the current state
            to the next.
        options: SolverOptions instance.
        initial_values: Initial values passed as parameters to the update function.
        names: Names of the parameters.
        data_handler: The data handler used to save to disk.
        fixed_values: Values that do not change over time, but should be added
            to saved data.
        fixed_names: Fixed data variable names.
        running_names: Names of running state variables.
        logger: A logger to print information about simulation.
        state: The current state variables.
    """

    def __init__(
        self,
        function: Callable,
        options: SolverOptions,
        initial_values: List[Any],
        names: Sequence[str],
        data_handler: DataHandler,
        fixed_values: Optional[List[Any]] = None,
        fixed_names: Optional[Sequence] = None,
        running_names: Optional[Sequence[str]] = None,
        logger: Optional[logging.Logger] = None,
        state: Optional[Dict[str, Any]] = None,
    ):
        # Set the initial data.
        self.time = 0
        self.options = options
        self.dt = self.options.dt_min
        self.function = function
        self.values = initial_values
        self.names = names
        self.fixed_values = fixed_values if fixed_values is not None else []
        self.fixed_names = fixed_names if fixed_names is not None else []
        self.running_names = running_names if running_names is not None else []
        self.running_state = RunningState(
            running_names if running_names is not None else [], self.options.save_every
        )
        self.state = state if state is not None else {}
        self.logger = logger if logger is not None else logging.getLogger()
        self.data_handler = data_handler

    def run(self) -> None:
        """Run the simulation loop."""
        # Set the initial data.
        self.time = 0
        self.state["step"] = 0
        self.state["time"] = self.time
        self.state["dt"] = self.dt
        # Thermalize if enabled.
        if self.options.skip_steps or self.options.skip_time:
            self._run_stage_(
                "Thermalizing",
                start_step=0,
                start_time=self.time,
                end_step=self.options.skip_steps,
                end_time=self.options.skip_time,
                save=False,
            )
            self.running_state.clear()
        self.time = 0
        self.state["step"] = 0
        self.state["time"] = self.time
        self.state["dt"] = self.dt
        # Run simulation.
        self._run_stage_(
            "Simulating",
            start_step=0,
            start_time=self.time,
            end_step=self.options.steps,
            end_time=self.options.total_time,
            save=True,
        )

    def _run_stage_(
        self,
        name: str,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        save: bool = True,
    ) -> None:
        """Run a stage of the simulation.

        Args:
            name: Name of the solver stage.
            start_step: Start step.
            end_step: End step.
            start_time: Start time.
            end_time: End time.
            save: If the data should be saved.
        """
        # Check if the progress bar is disabled.
        prog_disabled = (
            self.options.progress_interval is not None
            and self.options.progress_interval > 0
        )
        now = None
        if end_step is None and end_time is None:
            raise ValueError("Either 'end_step' or 'end_time' must be specified.")
        if end_step is not None and end_time is not None:
            raise ValueError(
                "Either 'end_step' or 'end_time' must be specified (but not both)."
            )
        if end_step is not None:
            initial = start_step
            total = end_step + 1
            bar_format = None
            unit = "it"
            it = range(start_step, end_step + 1)
        else:
            initial = start_time
            total = end_time
            r_bar = (
                "| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]"
            )
            unit = "tau"
            bar_format = "{l_bar}{bar}" + r_bar
            it = itertools.count()

        def save_step(step):
            data = dict(zip(self.names, self.values))
            # Add the fixed values.
            for idx, name in enumerate(self.fixed_names):
                data[name] = self.fixed_values[idx]
            # Add the running state data to the dict.
            if step != 0:
                data.update(self.running_state.export())
            self.data_handler.save_time_step(self.state, data)

        with tqdm(
            initial=initial,
            total=total,
            desc=name,
            disable=prog_disabled,
            unit=unit,
            bar_format=bar_format,
        ) as pbar:
            for i in it:
                saved_this_iteration = False
                # Update the state
                self.state["step"] = i
                self.state["time"] = self.time
                self.state["dt"] = self.dt
                # Print progress if TQDM is disabled.
                if prog_disabled and i % self.options.progress_interval == 0:
                    then, now = now, time.perf_counter()
                    if then is None:
                        speed = 0
                    else:
                        speed = self.options.progress_interval / (now - then)
                    if end_step is None:
                        self.logger.info(
                            f"{name}: Time {self.time}/{end_time}, "
                            f"dt={self.dt:.2e}, {speed:.2f} it/s"
                        )
                    else:
                        self.logger.info(
                            f"{name}: Iteration {i}/{end_step + 1}, "
                            f"dt={self.dt:.2e}, {speed:.2f} it/s"
                        )
                # Save data
                if i % self.options.save_every == 0:
                    if save:
                        save_step(i)
                        saved_this_iteration = True
                    self.running_state.clear()

                # Run time step.
                function_result = self.function(
                    self.state, self.running_state, *self.values, self.dt
                )
                *self.values, new_dt = function_result

                if end_step is None:
                    # tqdm will spit out a warning if you try to update past "total"
                    if self.time + self.dt < end_time:
                        pbar.update(self.dt)
                    else:
                        pbar.update(end_time - self.time)
                else:
                    pbar.update(1)

                if self.options.dt_max is not None:
                    self.dt = max(self.options.dt_min, min(self.options.dt_max, new_dt))

                if self.time >= end_time:
                    if save and not saved_this_iteration:
                        save_step(i)
                    break

                self.running_state.next()
                self.time += self.dt
