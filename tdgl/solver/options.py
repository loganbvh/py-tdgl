from dataclasses import dataclass
from typing import Optional


@dataclass
class SolverOptions:
    """Options for the TDGL solver.

    Args:
        dt_min: Minimum time step.
        steps: Number of solve steps.
        dt_max: Maximum time step.
        total_time: Total simulation time.
        adaptive_window: Number of most recent solve steps to consider when
            computing the time step adaptively.
        save_every: Save interval in units of solve steps.
        progress_interval: Minimum number of solve steps between progress bar updates.
    """

    dt_min: float = 1e-4
    steps: Optional[int] = None
    dt_max: Optional[float] = None
    total_time: Optional[float] = None
    adaptive_window: int = 1
    save_every: int = 100
    skip_steps: Optional[int] = None
    skip_time: Optional[float] = None
    progress_interval: int = 0

    def __post_init__(self):
        if self.total_time is not None and self.steps is not None:
            raise ValueError("Options 'total_time' and 'steps' are mutually exclusive.")
        if self.dt_max is not None and self.steps is not None:
            raise ValueError("Options 'dt_max' and 'steps' are mutually exclusive.")
        if self.skip_steps and self.skip_time:
            raise ValueError(
                "Options 'skip_steps' and 'skip_time' are mutually exclusive."
                " Only one of these options can be non-null."
            )
