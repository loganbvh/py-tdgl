import itertools
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from tqdm import tqdm

from .data import DataHandler, RunningState


@dataclass
class SolverOptions:
    dt_min: float = 1e-4
    dt_max: Optional[float] = None
    total_time: Optional[float] = None
    min_steps: Optional[int] = None
    max_steps: Optional[int] = None
    adaptive_window: int = 10
    save_every: int = 100
    skip_steps: Optional[int] = None
    skip_time: Optional[float] = None
    progress_interval: int = 0
    rtol: float = 0.0

    def __post_init__(self):
        if self.total_time is not None and self.min_steps is not None:
            raise ValueError(
                "Options 'total_time' and 'min_steps' are mutually exclusive."
            )
        if self.total_time is not None and self.max_steps is not None:
            raise ValueError(
                "Options 'total_time' and 'max_steps' are mutually exclusive."
            )
        if self.skip_steps and self.skip_time:
            raise ValueError(
                "Options 'skip_steps' and 'skip_time' are mutually exclusive."
                " Only one of these options can be nonzero."
            )


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
        self.state["step"] = 0
        self.state["time"] = 0
        self.state["dt"] = self.dt
        # Run simulation.
        self._run_stage_(
            "Simulating",
            start_step=0,
            start_time=0,
            end_step=self.options.min_steps,
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
                converged, *self.values, new_dt = function_result

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

                if converged and i > self.min_steps:
                    if save and not saved_this_iteration:
                        save_step(i)
                    self.logger.warning(f"\nSimulation converged at step {i}.")
                    break

                if self.time >= end_time:
                    if save and not saved_this_iteration:
                        save_step(i)
                    break

                self.running_state.next()
                self.time += self.dt
