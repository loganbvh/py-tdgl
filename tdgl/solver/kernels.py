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


@numba.njit(fastmath=False, parallel=True, cache=True)
def _implicit_matvec(x, out, a, s, fixed, indptr, indices, data):
    # (I - a S L) x on free rows, identity on fixed rows
    for i in numba.prange(x.shape[0]):
        if fixed[i]:
            out[i] = x[i]
            continue
        acc = 0j
        for k in range(indptr[i], indptr[i + 1]):
            acc += data[k] * x[indices[k]]
        out[i] = x[i] - a * s[i] * acc


@numba.njit(fastmath=False, parallel=True, cache=True)
def _cdot(x, y):
    # sum conj(x) * y
    acc = 0j
    for i in numba.prange(x.shape[0]):
        acc += np.conj(x[i]) * y[i]
    return acc


@numba.njit(fastmath=False, parallel=True, cache=True)
def _norm2(x):
    acc = 0.0
    for i in numba.prange(x.shape[0]):
        acc += x[i].real * x[i].real + x[i].imag * x[i].imag
    return acc


@numba.njit(fastmath=False, parallel=True, cache=True)
def implicit_laplacian_solve(
    rhs, x0, a, s, fixed, indptr, indices, data, rtol, maxiter
):
    """Solve ``(I - a S L) x = rhs`` (identity on fixed rows) by Jacobi-preconditioned BiCGSTAB from ``x0``.

    Returns ``(x, iterations, relative residual)``; iterations is -1 on breakdown.
    """
    n = rhs.shape[0]
    dinv = np.empty(n, dtype=np.complex128)
    for i in numba.prange(n):
        d = 1.0 + 0j
        if not fixed[i]:
            for k in range(indptr[i], indptr[i + 1]):
                if indices[k] == i:
                    d = 1.0 - a * s[i] * data[k]
        dinv[i] = 1.0 / d
    x = x0.copy()
    r = np.empty(n, dtype=np.complex128)
    _implicit_matvec(x, r, a, s, fixed, indptr, indices, data)
    for i in numba.prange(n):
        r[i] = rhs[i] - r[i]
    bnorm = np.sqrt(_norm2(rhs))
    if bnorm == 0.0:
        bnorm = 1.0
    res = np.sqrt(_norm2(r)) / bnorm
    if res <= rtol:
        return x, 0, res
    r0 = r.copy()
    rho = alpha = omega = 1.0 + 0j
    v = np.zeros(n, dtype=np.complex128)
    p = np.zeros(n, dtype=np.complex128)
    y = np.empty(n, dtype=np.complex128)
    z = np.empty(n, dtype=np.complex128)
    t = np.empty(n, dtype=np.complex128)
    sv = np.empty(n, dtype=np.complex128)
    for it in range(1, maxiter + 1):
        rho_new = _cdot(r0, r)
        if rho_new == 0:
            return x, -1, res
        beta = (rho_new / rho) * (alpha / omega)
        rho = rho_new
        for i in numba.prange(n):
            p[i] = r[i] + beta * (p[i] - omega * v[i])
            y[i] = dinv[i] * p[i]
        _implicit_matvec(y, v, a, s, fixed, indptr, indices, data)
        den = _cdot(r0, v)
        if den == 0:
            return x, -1, res
        alpha = rho / den
        for i in numba.prange(n):
            sv[i] = r[i] - alpha * v[i]
        if np.sqrt(_norm2(sv)) / bnorm <= rtol:
            for i in numba.prange(n):
                x[i] += alpha * y[i]
            return x, it, np.sqrt(_norm2(sv)) / bnorm
        for i in numba.prange(n):
            z[i] = dinv[i] * sv[i]
        _implicit_matvec(z, t, a, s, fixed, indptr, indices, data)
        # Jacobi-preconditioned BiCGSTAB: omega from the plain residual norms
        tt = _cdot(t, t)
        if tt == 0:
            return x, -1, res
        omega = _cdot(t, sv) / tt
        for i in numba.prange(n):
            x[i] += alpha * y[i] + omega * z[i]
            r[i] = sv[i] - omega * t[i]
        res = np.sqrt(_norm2(r)) / bnorm
        if res <= rtol:
            return x, it, res
    return x, maxiter, res


@numba.njit(fastmath=False, parallel=True, cache=True)
def psi_finish(
    psi,
    abs_sq_psi,
    mu,
    epsilon,
    gamma,
    u,
    dt,
    psi_star,
    fixed,
    lap_indptr,
    lap_indices,
    lap_data,
    new_psi,
    new_sq_psi,
):
    """The |psi|^2 quadratic of ``psi_step`` with ``U psi_star`` in place of ``U [psi + dt/u ...]``.

    ``psi_star`` already holds ``psi + dt/u sqrt(1 + gamma^2 |psi|^2) ((eps - |psi|^2) psi + L psi_star)``. Fixed
    rows keep the explicit update so they match the stock scheme.
    """
    n = psi.shape[0]
    bad = 0
    g2h = gamma * gamma / 2
    for i in numba.prange(n):
        p = psi[i]
        s = abs_sq_psi[i]
        U = np.exp(-1j * mu[i] * dt)
        z = U * g2h * p
        if fixed[i]:
            lap = 0j
            for k in range(lap_indptr[i], lap_indptr[i + 1]):
                lap += lap_data[k] * psi[lap_indices[k]]
            inner = p + (dt / u) * np.sqrt(1 + gamma * gamma * s) * (
                (epsilon[i] - s) * p + lap
            )
        else:
            inner = psi_star[i]
        w = z * s + U * inner
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
