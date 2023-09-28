from typing import Callable, Dict, Optional, Union

from ..device.device import Device
from ..solution.solution import Solution
from .options import SolverOptions
from .solver import TDGLSolver


def solve(
    device: Device,
    options: SolverOptions,
    applied_vector_potential: Union[Callable, float] = 0,
    terminal_currents: Union[Callable, Dict[str, float], None] = None,
    disorder_epsilon: Union[float, Callable] = 1,
    seed_solution: Optional[Solution] = None,
) -> Union[Solution, None]:
    """Solve a TDGL model.

    Args:
        device: The :class:`tdgl.Device` to solve.
        options: An instance :class:`tdgl.SolverOptions` specifying the solver
            parameters.
        applied_vector_potential: A function or :class:`tdgl.Parameter` that computes
            the applied vector potential as a function of position ``(x, y, z)``,
            or of position and time ``(x, y, z, *, t)``. If a float ``B`` is given,
            the applied vector potential will be that of a uniform magnetic field with
            strength ``B`` ``field_units``.
        terminal_currents: A dict of ``{terminal_name: current}`` or a callable with signature
            ``func(time: float) -> {terminal_name: current}``, where ``current`` is a float
            in units of ``current_units`` and ``time`` is the dimensionless time.
        disorder_epsilon: A float in range [-1, 1], or a callable with signature
            ``disorder_epsilon(r: Tuple[float, float]) -> epsilon``, where ``epsilon``
            is a float in range [-1, 1]. Setting
            :math:`\\epsilon(\\mathbf{r})=T_c(\\mathbf{r})/T_c - 1 < 1` suppresses the
            critical temperature at position :math:`\\mathbf{r}`, which can be used
            to model inhomogeneity.
        seed_solution: A :class:`tdgl.Solution` instance to use as the initial state
            for the simulation.

    Returns:
        A :class:`tdgl.Solution` instance. Returns ``None`` if the simulation was
        cancelled during the thermalization stage.
    """
    solver = TDGLSolver(
        device=device,
        options=options,
        applied_vector_potential=applied_vector_potential,
        terminal_currents=terminal_currents,
        disorder_epsilon=disorder_epsilon,
        seed_solution=seed_solution,
    )
    return solver.solve()
