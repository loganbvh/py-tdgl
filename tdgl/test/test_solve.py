import numpy as np
import pytest

import tdgl


@pytest.mark.parametrize("current", [1.0, 10.0, lambda t: 10])
@pytest.mark.parametrize("field", [0, 1])
def test_source_drain_current(transport_device, current, field):

    device = transport_device
    dt = 1e-3
    total_time = 100

    options = tdgl.SolverOptions(
        dt_init=dt,
        dt_max=10 * dt,
        solve_time=total_time,
        save_every=100,
    )
    if callable(current):

        def terminal_currents(t):
            return dict(source=current(0), drain=-current(0))

    else:
        terminal_currents = dict(source=current, drain=-current)
    solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=field,
        field_units="uT",
        terminal_currents=terminal_currents,
        current_units="uA",
    )

    if callable(current):
        current = current(0)

    ys = np.linspace(-5, 5, 501)
    measured_currents = []
    for x0 in [-8, -2, 0, 2, 8]:
        coords = np.array([x0 * np.ones_like(ys), ys]).T
        measured_currents.append(
            solution.current_through_path(coords, with_units=False)
        )
    measured_currents = np.array(measured_currents)
    assert np.allclose(measured_currents, current, rtol=0.1)
