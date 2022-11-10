from dataclasses import dataclass


class SolverOptionsError(ValueError):
    pass


@dataclass
class SolverOptions:
    """Options for the TDGL solver.

    Args:
        dt_init: Initial time step.
        adaptive: Whether to use an adpative time step.
        dt_max: Maximum adaptive time step.
        total_time: Total simulation time.
        skip_time: Amount of 'thermalization' time to simulate before recording data.
        adaptive_window: Number of most recent solve steps to consider when
            computing the time step adaptively.
        save_every: Save interval in units of solve steps.
        progress_interval: Minimum number of solve steps between progress bar updates.
    """

    solve_time: float
    skip_time: float = 0.0
    dt_init: float = 1e-4
    dt_max: float = 1e-1
    adaptive: bool = True
    adaptive_window: int = 10
    max_solve_retries: int = 10
    save_every: int = 100
    progress_interval: int = 0

    def validate(self) -> None:
        pass
