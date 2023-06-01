import numpy as np
import pytest

import tdgl
from tdgl.solver.options import SolverOptionsError


@pytest.mark.parametrize("current", [5.0, lambda t: 10])
@pytest.mark.parametrize("field", [0, 1])
@pytest.mark.parametrize("terminal_psi", [0, 1])
def test_source_drain_current(transport_device, current, field, terminal_psi):
    device = transport_device
    total_time = 100

    options = tdgl.SolverOptions(
        solve_time=total_time,
        field_units="uT",
        current_units="uA",
        save_every=100,
        terminal_psi=terminal_psi,
    )
    if callable(current):

        def terminal_currents(t):
            return dict(source=current(0), drain=-current(0))

    else:
        terminal_currents = dict(source=current, drain=-current)

    with pytest.raises(ValueError):
        solution = tdgl.solve(
            device,
            options,
            disorder_epsilon=2,
            applied_vector_potential=field,
            terminal_currents=terminal_currents,
        )
    solution = tdgl.solve(
        device,
        options,
        disorder_epsilon=lambda r: 1,
        applied_vector_potential=field,
        terminal_currents=terminal_currents,
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


@pytest.mark.skip
@pytest.mark.parametrize(
    "use_numba, use_jax", [(False, True), (True, False), (False, False)]
)
def test_screening(box_device, use_numba, use_jax):
    device = box_device
    total_time = 5

    with pytest.raises(SolverOptionsError):
        tdgl.SolverOptions(solve_time=total_time, dt_init=1e-3, dt_max=1e-4).validate()
    with pytest.raises(SolverOptionsError):
        tdgl.SolverOptions(solve_time=total_time, terminal_psi=2).validate()
    with pytest.raises(SolverOptionsError):
        tdgl.SolverOptions(solve_time=total_time, screening_step_size=-1).validate()
    with pytest.raises(SolverOptionsError):
        tdgl.SolverOptions(solve_time=total_time, screening_tolerance=-1).validate()
    with pytest.raises(SolverOptionsError):
        tdgl.SolverOptions(solve_time=total_time, screening_step_drag=2).validate()
    with pytest.raises(SolverOptionsError):
        tdgl.SolverOptions(
            solve_time=total_time, adaptive_time_step_multiplier=2
        ).validate()
    with pytest.raises(SolverOptionsError):
        tdgl.SolverOptions(
            solve_time=total_time, screening_use_jax=True, screening_use_numba=True
        ).validate()

    options = tdgl.SolverOptions(
        solve_time=total_time,
        field_units="uT",
        current_units="uA",
        screening_use_jax=use_jax,
        screening_use_numba=use_numba,
    )
    field = tdgl.sources.ConstantField(50)

    options.include_screening = False
    solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=field,
    )

    circle = tdgl.geometry.circle(2, points=401)
    centers = [(0, 0), (-1.5, -2.5), (2.5, 2), (0, 1)]

    fluxoids_without_screening = []
    for r0 in centers:
        fluxoid = solution.polygon_fluxoid(circle + np.atleast_2d(r0), with_units=False)
        # Without screening the fluxoid will not be quantized.
        fluxoids_without_screening.append(abs(sum(fluxoid)))

    options.include_screening = True
    solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=field,
    )

    fluxoids_with_screening = []
    for r0 in centers:
        fluxoid = solution.polygon_fluxoid(circle + np.atleast_2d(r0), with_units=False)
        fluxoids_with_screening.append(abs(sum(fluxoid)))

    assert np.all(fluxoids_with_screening < fluxoids_without_screening)
