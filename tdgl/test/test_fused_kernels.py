import numpy as np
import pytest

import tdgl


def _run(device, field, terminal_psi, time_dependent, **opts):
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
    applied = tdgl.sources.ConstantField(
        field, field_units=options.field_units, length_units=device.length_units
    )
    if time_dependent:
        applied = applied * tdgl.sources.LinearRamp(tmin=0, tmax=3)
    return tdgl.solve(
        device,
        options,
        applied_vector_potential=applied,
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
@pytest.mark.parametrize("time_dependent", [False, True])
def test_fused_kernels_match_numpy(transport_device, terminal_psi, time_dependent):
    ref = _run(transport_device, 1, terminal_psi, time_dependent, fused_kernels=False)
    new = _run(transport_device, 1, terminal_psi, time_dependent, fused_kernels=True)
    psi_ref, mu_ref = _gauge_free(ref)
    psi_new, mu_new = _gauge_free(new)
    assert np.allclose(psi_new, psi_ref, rtol=0, atol=1e-9)
    assert np.allclose(mu_new, mu_ref, rtol=0, atol=1e-9)
    assert np.allclose(
        new.dynamics.phase_difference(0, 1),
        ref.dynamics.phase_difference(0, 1),
        atol=1e-10,
    )
