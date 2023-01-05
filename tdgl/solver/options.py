import os
from dataclasses import dataclass
from typing import Union


class SolverOptionsError(ValueError):
    pass


@dataclass
class SolverOptions:
    """Options for the TDGL solver.

    Args:
        solve_time: Total simulation time, after any thermalization.
        skip_time: Amount of 'thermalization' time to simulate before recording data.
        dt_init: Initial time step.
        dt_max: Maximum adaptive time step.
        adaptive: Whether to use an adpative time step. Setting ``dt_init = dt_max``
            is equivalent to setting ``adaptive = False``.
        adaptive_window: Number of most recent solve steps to consider when
            computing the time step adaptively.
        max_solve_retries: The maximum number of times to reduce the time step in a
            given solve iteration before giving up.
        adaptive_time_step_multiplier: The factor by which to multiple the time
            step ``dt`` for each adaptive solve retry.
        field_units: The units for magnetic fields.
        current_units: The units for currents.
        output_file: Path to an HDF5 file in which to save the data.
            If the file name already exists, a unique name will be generated.
            If ``output_file`` is ``None``, the solver results will not be saved
            to disk.
        save_every: Save interval in units of solve steps.
        progress_interval: Minimum number of solve steps between progress bar updates.
        include_screening: Whether to include screening in the simulation.
        max_iterations_per_step: The maximum number of screening iterations per solve
            step.
        screening_tolerance: Relative tolerance for the induced vector potential, used
            to evaluate convergence of the screening calculation within a single time
            step.
        screening_step_size: Step size :math:`\\alpha` for Polyak's method.
        screening_step_drag: Drag parameter :math:`\\beta` for Polyak's method.
    """

    solve_time: float
    skip_time: float = 0.0
    dt_init: float = 1e-6
    dt_max: float = 1e-1
    adaptive: bool = True
    adaptive_window: int = 10
    max_solve_retries: int = 10
    adaptive_time_step_multiplier: float = 0.25
    save_every: int = 100
    progress_interval: int = 0
    field_units: str = "mT"
    current_units: str = "uA"
    output_file: Union[os.PathLike, None] = None
    include_screening: bool = False
    max_iterations_per_step: int = 1000
    screening_tolerance: float = 1e-3
    screening_step_size: float = 1.0
    screening_step_drag: float = 0.5

    def validate(self) -> None:
        if self.dt_init > self.dt_max:
            raise SolverOptionsError("dt_init must be less than or equal to dt_max.")
        if not (0 < self.adaptive_time_step_multiplier < 1):
            raise SolverOptionsError(
                "adaptive_time_step_multiplier must be in (0, 1)"
                f" (got {self.adaptive_time_step_multiplier})."
            )
        if not (0 < self.screening_step_drag <= 1):
            raise SolverOptionsError(
                "screening_step_drag must be in (0, 1)"
                f" (got {self.screening_step_drag})."
            )
        if self.screening_step_size <= 0:
            raise SolverOptionsError(
                "screening_step_size must be in > 0"
                f" (got {self.screening_step_size})."
            )
        if self.screening_tolerance <= 0:
            raise SolverOptionsError(
                "screening_tolerance must be in > 0"
                f" (got {self.screening_tolerance})."
            )
