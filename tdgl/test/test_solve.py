import os
import tempfile

import numpy as np
import pytest

import tdgl


@pytest.mark.parametrize("current", [1.0, 10.0, lambda t: 10])
@pytest.mark.parametrize("field", [None, 0, 1])
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
    if field is not None:
        field = tdgl.sources.ConstantField(field)
    with tempfile.TemporaryDirectory() as directory:
        fname = os.path.join(directory, "output.h5")
        solution = tdgl.solve(
            device,
            fname,
            options,
            applied_vector_potential=field,
            field_units="uT",
            source_drain_current=current,
            current_units="uA",
            include_screening=False,
        )

    if callable(current):
        current = current(0)

    ys = np.linspace(-5, 5, 501)
    for x0 in [-8, -2, 0, 2, 8]:
        coords = np.array([x0 * np.ones_like(ys), ys]).T
        measured_current = np.sum(
            solution.interp_current_density(coords, units="uA/um")[:, 0]
            * np.diff(ys, prepend=0)
        )
        assert np.isclose(measured_current, current, rtol=0.1)
