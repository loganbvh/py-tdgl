"""Fused numba kernels for the CPU time step.

Each kernel does in one pass what the NumPy path does with several sparse products and temporaries.
They parallelize over sites or edges with ``numba.prange``, so the thread count is ``NUMBA_NUM_THREADS``.
"""

import numba
import numpy as np


@numba.njit(fastmath=False, parallel=True, cache=True)
def psi_step(
    psi: np.ndarray,
    abs_sq_psi: np.ndarray,
    mu: np.ndarray,
    epsilon: np.ndarray,
    gamma: float,
    u: float,
    dt: float,
    lap_indptr: np.ndarray,
    lap_indices: np.ndarray,
    lap_data: np.ndarray,
    new_psi: np.ndarray,
    new_sq_psi: np.ndarray,
) -> bool:
    """Same update as ``TDGLSolver.solve_for_psi_squared``, with the covariant Laplacian (CSR) fused in.

    Returns False where the NumPy path would return None: a negative discriminant or a non-finite value.
    """
    n = psi.shape[0]
    bad = 0
    g2h = gamma * gamma / 2
    for i in numba.prange(n):
        lap = 0j
        for k in range(lap_indptr[i], lap_indptr[i + 1]):
            lap += lap_data[k] * psi[lap_indices[k]]
        p = psi[i]
        s = abs_sq_psi[i]
        U = np.exp(-1j * mu[i] * dt)
        z = U * g2h * p
        w = z * s + U * (
            p + (dt / u) * np.sqrt(1 + gamma * gamma * s) * ((epsilon[i] - s) * p + lap)
        )
        c = w.real * z.real + w.imag * z.imag
        two_c_1 = 2 * c + 1
        w2 = w.real * w.real + w.imag * w.imag
        z2 = z.real * z.real + z.imag * z.imag
        disc = two_c_1 * two_c_1 - 4 * z2 * w2
        if not (disc >= 0) or not np.isfinite(w2):
            bad += 1
            continue
        sq = (2 * w2) / (two_c_1 + np.sqrt(disc))
        new_sq_psi[i] = sq
        new_psi[i] = w - z * sq
    return bad == 0


@numba.njit(fastmath=False, parallel=True, cache=True)
def supercurrent(
    psi: np.ndarray,
    edges0: np.ndarray,
    grad_indptr: np.ndarray,
    grad_indices: np.ndarray,
    grad_data: np.ndarray,
    out: np.ndarray,
) -> None:
    """``Im(conj(psi[e0]) * (psi_gradient @ psi))`` on every edge, without temporaries."""
    for e in numba.prange(out.shape[0]):
        g = 0j
        for k in range(grad_indptr[e], grad_indptr[e + 1]):
            g += grad_data[k] * psi[grad_indices[k]]
        q = psi[edges0[e]]
        out[e] = q.real * g.imag - q.imag * g.real
