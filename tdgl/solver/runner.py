import itertools
import logging
import numbers
import os
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import h5py
import numpy as np
from tqdm import TqdmWarning, tqdm

from ..finite_volume.mesh import Mesh
from .options import SolverOptions


def _get(item):
    if not isinstance(item, np.ndarray):
        item = item.get()
    return item


class DataHandler:
    """A context manager that is responsible for reading from and writing to disk.

    Args:
        output_file: File to use as output for simulation data.
        logger: Logger used to inform about errors.
    """

    def __init__(
        self,
        output_file: Union[str, None],
        logger: Union[logging.Logger, None] = None,
    ):
        self.tempdir = None
        self.mesh_group = None
        self.time_step_group = None
        self.save_number = 0
        self.logger = logger if logger is not None else logging.getLogger()
        self._base_output_file = output_file
        self.output_file: Union[h5py.File, None] = None
        self.output_path: Union[str, None] = None
        self.tmp_file: Union[h5py.File, None] = None
        self.tmp_path: Union[str, None] = None
        self.time_step_group: Union[h5py.Group, None] = None
        self.mesh_group: Union[h5py.Group, None] = None

    def _create_output_file(self, output: str) -> Tuple[h5py.File, str, h5py.File, str]:
        """Create an output file.

        Args:
            output: The output file path.
            logger: Logger output logs.

        Returns:
            A :class:`h5py.File` and the path to the file.
        """
        if output is None:
            self.tempdir = tempfile.TemporaryDirectory()
            directory = self.tempdir.name
            name = "output"
            suffix = "h5"
        else:
            # Make sure the directory exists
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            # Split the output into file name and suffix
            name_parts = output.split(".")
            name = ".".join(name_parts[:-1])
            suffix = name_parts[-1]
            directory = os.getcwd()
        # Number to be added to the end of file name
        # If this is None do not add the number
        serial_number = None
        while True:
            # Create a new file name
            name_suffix = f"-{serial_number}" if serial_number is not None else ""
            file_name = f"{name}{name_suffix}.{suffix}"
            file_path = os.path.join(directory, file_name)
            tmp_file_name = f"{file_name}.tmp"
            tmp_file_path = os.path.join(directory, tmp_file_name)
            try:
                file = h5py.File(file_path, "x")
                tmp_file = h5py.File(tmp_file_path, "x", libver="latest")
            except (OSError, FileExistsError):
                if serial_number is None:
                    serial_number = 1
                else:
                    serial_number += 1
                continue
            else:
                if serial_number is not None:
                    self.logger.warning(
                        f"Output file already exists. Renaming to {file_name}."
                    )
            return file, file_path, tmp_file, tmp_file_path

    def __enter__(self) -> "DataHandler":
        (
            self.output_file,
            self.output_path,
            self.tmp_file,
            self.tmp_path,
        ) = self._create_output_file(self._base_output_file)
        self.time_step_group = self.output_file.create_group("data", track_order=True)

        grp = self.tmp_file.create_group("data/-1")
        grp["step"] = np.array([0])
        grp["time"] = np.array([0.0])
        grp["dt"] = np.array([0.0])
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if exc_value is not None:
            self.logger.warning(
                "Ignoring the following exception in DataHandler.__exit__():"
            )
            self.logger.warning(
                "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            )
        self.close()

    def close(self):
        """Clean up by closing the output file."""
        self.output_file.close()
        if self.tmp_file is not None:
            self.tmp_file.flush()
            self.tmp_file.close()
            os.remove(self.tmp_path)
        if self.tempdir is not None:
            self.tempdir.cleanup()

    def save_mesh(self, mesh: Mesh) -> None:
        """Save the ``Mesh`` to ``self.output_file["mesh"]``.

        Args:
            mesh: The ``Mesh`` to save.
        """
        self.mesh_group = self.output_file.create_group("mesh")
        mesh.to_hdf5(self.mesh_group)

    def save_fixed_values(self, fixed_data: Dict[str, np.ndarray]) -> None:
        """Save the fixed values, i.e., those that aren't updated at each solve step."""
        for key, value in fixed_data.items():
            value = _get(value)
            self.output_file[key] = value
            self.tmp_file[key] = value

    def save_time_step(
        self,
        state: Dict[str, numbers.Real],
        data: Dict[str, np.ndarray],
        running_state: Union[Dict[str, np.ndarray], None],
    ) -> None:
        """Save the state and data that are updated at each solve step."""
        group = self.time_step_group.create_group(f"{self.save_number}")
        group.attrs["timestamp"] = datetime.now().isoformat()
        self.save_number += 1
        tmp_grp = self.tmp_file["data/-1"]

        for key, value in state.items():
            group.attrs[key] = value
        for key, value in data.items():
            value = _get(value)
            group[key] = value
            if key in tmp_grp:
                tmp_grp[key][:] = value
            else:
                tmp_grp[key] = value
            tmp_grp[key].flush()
        for key in ("step", "time", "dt"):
            tmp_grp[key][:] = np.array([state[key]])
            tmp_grp[key].flush()
        if running_state is not None:
            running_grp = group.create_group("running_state")
            for key, value in running_state.items():
                running_grp[key] = np.squeeze(_get(value))


class RunningState:
    """Storage class for saving scalar data that is saved at each time step.

    Args:
        names_and_sizes: A dict of the parameters to be saved and their sizes, i.e.,
            the number of values measured at each time step for each parameter.
        buffer_size: The size of the buffer, i.e., the number of values to record
            before writing to disk.
    """

    def __init__(
        self, names_and_sizes: Dict[str, int], buffer_size: int, array_module=np
    ):
        self.step = 0
        self.array_module = array_module
        self.buffer_size = buffer_size
        self.names_and_sizes = names_and_sizes
        self.values = {
            name: array_module.zeros((size, buffer_size))
            for name, size in self.names_and_sizes.items()
        }

    def clear(self) -> None:
        """Clear the buffer."""
        self.step = 0
        for name, size in self.names_and_sizes.items():
            self.values[name] = self.array_module.zeros((size, self.buffer_size))

    def append(self, name: str, value: Sequence[float]) -> None:
        """Append data to the buffer.

        Args:
            name: Data to append.
            value: Value of the data.
        """
        self.values[name][:, self.step] = value


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
        monitor: Launch a subprocess to plot results during the simulation.
        monitor_update_interval: The monitor update interval in seconds.
        fixed_values: Values that do not change over time, but should be added
            to saved data.
        fixed_names: Fixed data variable names.
        running_names_and_sizes: Names and shapes of running state variables.
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
        monitor: bool = False,
        monitor_update_interval: float = 1.0,
        fixed_values: Union[List[Any], None] = None,
        fixed_names: Union[Sequence, None] = None,
        running_names_and_sizes: Union[Dict[str, int], None] = None,
        logger: Union[logging.Logger, None] = None,
        state: Union[Dict[str, Any], None] = None,
    ):
        self.time = 0
        self.options = options
        self.dt = self.options.dt_init
        self.function = function
        self.values = initial_values
        self.names = names
        self.fixed_values = fixed_values if fixed_values is not None else []
        self.fixed_names = fixed_names if fixed_names is not None else []
        self.running_names_and_sizes = (
            running_names_and_sizes if running_names_and_sizes is not None else {}
        )
        if options.gpu:
            import cupy  # type: ignore

            array_module = cupy
        else:
            array_module = np
        self.running_state = RunningState(
            self.running_names_and_sizes,
            self.options.save_every,
            array_module=array_module,
        )
        self.state = state if state is not None else {}
        self.logger = logger if logger is not None else logging.getLogger()
        self.data_handler = data_handler
        self.monitor = monitor
        self.monitor_update_interval = monitor_update_interval

    def run(self) -> bool:
        """Run the simulation loop.

        Returns:
            A boolean indicating whether any data was generated.
        """
        self.time = 0
        self.state["step"] = 0
        self.state["time"] = self.time
        self.state["dt"] = float(self.dt)
        self.data_handler.save_fixed_values(
            dict(zip(self.fixed_names, self.fixed_values))
        )
        success = True
        # Thermalize if enabled
        if self.options.skip_time:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=TqdmWarning)
                success = self._run_stage(
                    "Thermalizing",
                    start_time=self.time,
                    end_time=self.options.skip_time,
                    save=False,
                )
            self.running_state.clear()
        if not success:
            return False
        self.time = 0
        self.state["step"] = 0
        self.state["time"] = self.time
        self.state["dt"] = self.options.dt_init
        # Run the simulation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=TqdmWarning)
            self._run_stage(
                "Simulating",
                start_time=self.time,
                end_time=self.options.solve_time,
                save=True,
            )
        return True

    def _run_stage(
        self,
        name: str,
        start_time: float,
        end_time: float,
        save: bool = True,
    ) -> bool:
        """Run a stage of the simulation.

        Args:
            name: Name of the solver stage.
            start_time: Start time.
            end_time: End time.
            save: If the data should be saved.

        Returns:
            A boolean indicating whether the stage terminated normally. If False is returned,
            future stages are cancelled.
        """
        prog_disabled = (
            self.options.progress_interval is not None
            and self.options.progress_interval > 0
        )
        now = None
        initial = start_time
        total = end_time
        r_bar = "| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]"
        unit = "tau"
        bar_format = "{l_bar}{bar}" + r_bar
        it = itertools.count()

        def save_step(step):
            data = dict(zip(self.names, self.values))
            if step == 0:
                running_state = None
            else:
                running_state = self.running_state.values
            self.data_handler.save_time_step(self.state, data, running_state)

        cancelled = False
        with tqdm(
            initial=initial,
            total=total,
            desc=name,
            disable=prog_disabled,
            unit=unit,
            bar_format=bar_format,
            dynamic_ncols=True,
        ) as pbar:
            for i in it:
                try:
                    dt = self.dt
                    self.state["step"] = i
                    self.state["time"] = self.time
                    self.state["dt"] = dt
                    # Print progress if TQDM is disabled.
                    if prog_disabled and (i % self.options.progress_interval) == 0:
                        then, now = now, time.perf_counter()
                        if then is None:
                            speed = 0
                        else:
                            speed = self.options.progress_interval / (now - then)
                        self.logger.info(
                            f"{name}: Time {self.time}/{end_time}, "
                            f"dt={dt:.2e}, {speed:.2f} it/s"
                        )
                    if i % self.options.save_every == 0:
                        if save:
                            save_step(i)
                        self.running_state.clear()

                    # Start the monitor subprocess
                    if save and i == 0 and self.data_handler.tmp_file is not None:
                        self.data_handler.tmp_file.swmr_mode = True
                        if self.monitor:
                            cmd = [
                                sys.executable,
                                "-m",
                                "tdgl.visualize",
                                "--input",
                                self.data_handler.output_path,
                                "monitor",
                                "--interval",
                                str(self.monitor_update_interval),
                            ]
                            _ = subprocess.Popen(cmd, start_new_session=True)
                    # Run time step.
                    function_result = self.function(
                        self.state,
                        self.running_state,
                        dt,
                        **dict(zip(self.names, self.values)),
                    )
                    new_dt, *self.values = function_result
                    # tqdm will spit out a warning if you try to update past "total"
                    if self.time + dt < end_time:
                        pbar.update(dt)
                    else:
                        pbar.update(end_time - self.time)
                    if self.time >= end_time:
                        break
                    self.dt = new_dt
                    self.running_state.step += 1
                    self.time += self.dt
                except KeyboardInterrupt:
                    msg = f"{{}} simulation at step {i} of stage {name!r}."
                    if self.options.pause_on_interrupt:
                        response = input(
                            f"Simulation paused at stage {name!r} (step {i})."
                            " Continue simulation? [yN]"
                        )
                        resume = response.lower().startswith("y")
                        if resume:
                            self.logger.info(msg.format("Resuming"))
                        else:
                            self.logger.warning(msg.format("Cancelling"))
                            cancelled = True
                            break
                    else:
                        self.logger.warning(msg.format("Cancelling"))
                        cancelled = True
                        break
            if save and (i % self.options.save_every):
                save_step(i)
            return not cancelled
