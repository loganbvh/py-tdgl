from dataclasses import dataclass
from typing import Union

import numpy as np


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
        include_screening: Whether to include screening in the simulation.
        adaptive: Whether to use an adpative time step. Setting ``dt_init = dt_max``
            is equivalent to setting ``adaptive = False``.
        adaptive_window: Number of most recent solve steps to consider when
            computing the time step adaptively.
        max_solve_retries: The maximum number of times to reduce the time step in a
            given solve iteration before giving up.
        save_every: Save interval in units of solve steps.
        progress_interval: Minimum number of solve steps between progress bar updates.
        rng_seed: An integer to used as a seed for the pseudorandom number generator.
    """

    solve_time: float
    skip_time: float = 0.0
    dt_init: float = 1e-4
    dt_max: float = 1e-1
    include_screening: bool = False
    adaptive: bool = True
    adaptive_window: int = 10
    max_solve_retries: int = 10
    save_every: int = 100
    progress_interval: int = 0
    rng_seed: Union[int, str, None] = None

    def __post_init__(self) -> None:
        if self.rng_seed is None:
            self.rng_seed = np.random.SeedSequence().entropy
        self.rng_seed = str(self.rng_seed)

    def validate(self) -> None:
        if self.dt_init > self.dt_max:
            raise SolverOptionsError("dt_init must be less than or equal to dt_max.")
