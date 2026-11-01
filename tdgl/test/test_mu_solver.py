import numpy as np
import pytest

import tdgl


def _run(device, field, terminal_psi, **opts):
    # Fixed dt: with adaptive steps, whether a retry fires at the explicit stability edge depends on rounding.
    options = tdgl.SolverOptions(
        solve_time=4,
        dt_init=1e-3,
        adaptive=False,
        field_units="uT",
        current_units="uA",
        save_every=100,
        terminal_psi=terminal_psi,
        **opts,
    )
    return tdgl.solve(
        device,
        options,
        applied_vector_potential=field,
        terminal_currents=dict(source=5.0, drain=-5.0),
    )


def _gauge_free(solution):
    # psi up to a global phase, and mu up to a constant
    psi = solution.tdgl_data.psi
    mu = solution.tdgl_data.mu
    k = np.argmax(np.abs(psi))
    return psi * np.exp(-1j * np.angle(psi[k])), mu - mu.mean()


# terminal_psi = 1 is left out: with psi fixed at the terminals, the constant that stock SuperLU leaves in mu is
# physical rather than a gauge, and it is set by rounding, so even two stock runs need not agree.
@pytest.mark.parametrize("terminal_psi", [0])
def test_symmetric_mu_solve_matches_superlu(transport_device, terminal_psi):
    ref = _run(transport_device, 1, terminal_psi, sparse_solver="superlu")
    new = _run(transport_device, 1, terminal_psi, sparse_solver="superlu_sym")
    psi_ref, mu_ref = _gauge_free(ref)
    psi_new, mu_new = _gauge_free(new)
    assert np.allclose(np.abs(new.tdgl_data.psi), np.abs(ref.tdgl_data.psi), atol=1e-9)
    assert np.allclose(psi_new, psi_ref, atol=1e-9)
    assert np.allclose(mu_new, mu_ref, atol=1e-9)
    assert np.allclose(
        new.dynamics.phase_difference(0, 1),
        ref.dynamics.phase_difference(0, 1),
        atol=1e-9,
    )
