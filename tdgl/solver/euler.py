import itertools
import logging
from typing import Tuple, Union

import numpy as np
import scipy.sparse as sp
from numpy import errstate

from .options import SolverOptions

logger = logging.getLogger("solver")


def solve_for_psi_squared(
    *,
    psi: np.ndarray,
    abs_sq_psi: np.ndarray,
    mu: np.ndarray,
    epsilon: np.ndarray,
    gamma: float,
    u: float,
    dt: float,
    psi_laplacian: sp.spmatrix,
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """Solve for :math:`\\psi_i^{n+1}` and :math:`|\\psi_i^{n+1}|^2` given
    :math:`\\psi_i^n` and :math:`\\mu_i^n`.

    Args:
        psi: The current value of the order parameter, :math:`\\psi_^n`
        abs_sq_psi: The current value of the superfluid density, :math:`|\\psi^n|^2`
        mu: The current value of the electric potential, :math:`\\mu^n`
        epsilon: The disorder parameter, :math:`\\epsilon`
        gamma: The inelastic scattering parameter, :math:`\\gamma`.
        u: The ratio of relaxation times for the order parameter, :math:`u`
        dt: The time step
        psi_laplacian: The covariant Laplacian for the order parameter

    Returns:
        ``None`` if the calculation failed to converge, otherwise the new order
        parameter :math:`\\psi^{n+1}` and superfluid density :math:`|\\psi^{n+1}|^2`.
    """
    if isinstance(psi, np.ndarray):
        np_ = np
    else:
        import cupy  # type: ignore

        assert isinstance(psi, cupy.ndarray)
        np_ = cupy
    U = np_.exp(-1j * mu * dt)
    z = U * gamma**2 / 2 * psi
    with errstate(all="raise"):
        try:
            w = z * abs_sq_psi + U * (
                psi
                + (dt / u)
                * np_.sqrt(1 + gamma**2 * abs_sq_psi)
                * ((epsilon - abs_sq_psi) * psi + psi_laplacian @ psi)
            )
            c = w.real * z.real + w.imag * z.imag
            two_c_1 = 2 * c + 1
            w2 = np_.absolute(w) ** 2
            discriminant = two_c_1**2 - 4 * np_.absolute(z) ** 2 * w2
        except Exception:
            logger.debug("Unable to solve for |psi|^2.", exc_info=True)
            return None
    if np_.any(discriminant < 0):
        return None
    new_sq_psi = (2 * w2) / (two_c_1 + np_.sqrt(discriminant))
    psi = w - z * new_sq_psi
    return psi, new_sq_psi


def adaptive_euler_step(
    step: int,
    psi: np.ndarray,
    abs_sq_psi: np.ndarray,
    mu: np.ndarray,
    epsilon: np.ndarray,
    gamma: float,
    u: float,
    dt: float,
    psi_laplacian: sp.spmatrix,
    options: SolverOptions,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Update the order parameter and time step in an adaptive Euler step.

    Args:
        step: The solve step index, :math:`n`
        psi: The current value of the order parameter, :math:`\\psi_^n`
        abs_sq_psi: The current value of the superfluid density, :math:`|\\psi^n|^2`
        mu: The current value of the electric potential, :math:`\\mu^n`
        epsilon: The disorder parameter, :math:`\\epsilon`
        gamma: The inelastic scattering parameter, :math:`\\gamma`.
        u: The ratio of relaxation times for the order parameter, :math:`u`
        dt: The tentative time step, which will be updated
        psi_laplacian: The covariant Laplacian for the order parameter
        options: The solver options for the simulation

    Returns:
        :math:`\\psi^{n+1}`, :math:`|\\psi^{n+1}|^2`, and :math:`\\Delta t^{n}`.
    """
    kwargs = dict(
        psi=psi,
        abs_sq_psi=abs_sq_psi,
        mu=mu,
        epsilon=epsilon,
        gamma=gamma,
        u=u,
        dt=dt,
        psi_laplacian=psi_laplacian,
    )
    result = solve_for_psi_squared(**kwargs)
    for retries in itertools.count():
        if result is not None:
            break  # First evaluation of |psi|^2 was successful.
        if (not options.adaptive) or retries > options.max_solve_retries:
            raise RuntimeError(
                f"Solver failed to converge in {options.max_solve_retries}"
                f" retries at step {step} with dt = {dt:.2e}."
                f" Try using a smaller dt_init."
            )
        kwargs["dt"] = dt = dt * options.adaptive_time_step_multiplier
        result = solve_for_psi_squared(**kwargs)
    psi, new_sq_psi = result
    return psi, new_sq_psi, dt
