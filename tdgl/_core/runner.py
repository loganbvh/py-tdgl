import datetime
import logging
from typing import Callable, Sequence, Optional, Any, List, Dict

from tqdm import tqdm

from .io.data_handler import DataHandler
from .io.running_state import RunningState


class Runner:
    """The runner is responsible for the simulation loop. It handles the state,
    runs the update function and saves the data.

    Args:
        function: The update function that takes the state from the current state
            to the next.
        initial_values: Initial values passed as parameters to the update function.
        names: Names of the parameters.
        steps: The number of steps to run the simulation.
        save_every: How many steps to simulate before saving data.
        data_handler: The data handler used to save to disk.
        dt: The time step.
        skip: The number of time steps to skip to thermalize.
        fixed_values: Values that do not change over time, but should be added
            to saved data.
        fixed_names: Fixed data variable names.
        running_names: Names of running state variables.
        logger: A logger to print information about simulation.
        state: The current state variables.
        miniters: Number of steps between progress update.
    """

    def __init__(
        self,
        function: Callable,
        initial_values: List[Any],
        names: Sequence[str],
        max_steps: int,
        save_every: int,
        data_handler: DataHandler,
        dt: float,
        skip: int = 0,
        min_steps: int = 0,
        fixed_values: Optional[List[Any]] = None,
        fixed_names: Optional[Sequence] = None,
        running_names: Optional[Sequence[str]] = None,
        logger: Optional[logging.Logger] = None,
        state: Optional[Dict[str, Any]] = None,
        miniters: Optional[int] = None,
    ):
        # Set the initial data.
        self.time = 0
        self.dt = dt

        # Set the number of steps to take for simulation and thermalization.
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.skip = skip

        # Set the function to run in the loop.
        self.function = function

        # Set the initial parameter values for the function and the names.
        self.values = initial_values
        self.names = names
        self.fixed_values = fixed_values if fixed_values is not None else []
        self.fixed_names = fixed_names if fixed_names is not None else []
        self.running_names = running_names if running_names is not None else []
        self.running_state = RunningState(
            running_names if running_names is not None else [], save_every
        )
        self.state = state if state is not None else {}

        # Set how often to save.
        self.save_every = save_every

        # Set the logger.
        self.logger = logger if logger is not None else logging.getLogger()

        # Set the data handler.
        self.data_handler = data_handler

        # Set how often to update the progress bar.
        self.miniters = miniters

    def run(self) -> None:
        """Run the simulation loop."""
        # Set the initial data.
        self.state["step"] = 0
        self.state["time"] = self.time
        self.state["dt"] = self.dt
        # Thermalize if enabled.
        if self.skip > 0:
            self._run_stage_(0, self.skip, "Thermalizing", False)
            self.running_state.clear()
        self.state["step"] = 0
        self.state["time"] = self.time
        self.state["dt"] = self.dt
        # Run simulation.
        self._run_stage_(0, self.max_steps, "Simulating", True)

    def _run_stage_(self, start: int, end: int, stage_name: str, save: bool) -> None:
        """Run a stage of the simulation.

        Args:
            start: Start step.
            end: End step.
            stage_name: Name of the stage.
            save: If the data should be saved.
        """
        # Check if the progress bar is disabled.
        prog_disabled = self.miniters is not None
        # Create variable to save the current time.
        now = None
        for i in tqdm(range(start, end + 1), desc=stage_name, disable=prog_disabled):
            saved_this_iteration = False
            # Update the state
            self.state["step"] = i
            self.state["time"] = self.time
            self.state["dt"] = self.dt
            # Print progress if TQDM is disabled.
            if prog_disabled and i % self.miniters == 0:
                then = now
                now = datetime.datetime.now()
                it = (
                    self.miniters / (now - then).total_seconds()
                    if then is not None
                    else 0
                )
                self.logger.info(
                    "{} {}/{} {:.2f} it/s".format(stage_name, i, end + 1, it)
                )
            # Save data
            if i % self.save_every == 0:
                # Save data if it is enabled.
                if save:
                    # Create a dict containing the data.
                    data = dict(
                        (self.names[i], self.values[i]) for i in range(len(self.names))
                    )
                    # Add the fixed values.
                    for idx, name in enumerate(self.fixed_names):
                        data[name] = self.fixed_values[idx]
                    # Add the running state data to the dict.
                    if i != 0:
                        data.update(self.running_state.export())
                    # Save the time step.
                    self.data_handler.save_time_step(self.state, data)
                # Clear the running state.
                self.running_state.clear()
                saved_this_iteration = True

            # Run time step.
            (
                converged,
                *self.values,
            ) = self.function(self.state, self.running_state, *self.values)

            if converged and i > self.min_steps:
                if save and not saved_this_iteration:
                    # Create a dict containing the data.
                    data = dict(
                        (self.names[i], self.values[i]) for i in range(len(self.names))
                    )
                    # Add the fixed values.
                    for idx, name in enumerate(self.fixed_names):
                        data[name] = self.fixed_values[idx]
                    # Add the runnings state data to the dict.
                    if i != 0:
                        data.update(self.running_state.export())
                    # Save the time step.
                    self.data_handler.save_time_step(self.state, data)
                self.logger.warning(f"\nSimulation converged at step {i}.")
                return
            # Update the running state.
            self.running_state.next()
            # Run update time
            self.time += self.dt
