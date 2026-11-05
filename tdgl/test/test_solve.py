import numba
import numpy as np
import pytest

try:
    import cupy  # type: ignore
except ImportError:
    cupy = None

import tdgl
from tdgl.finite_volume.operators import MeshOperators
from tdgl.geometry import box, circle
from tdgl.solver.options import SolverOptionsError, SparseSolver


@pytest.mark.parametrize("current", [5.0, lambda t: 10])
@pytest.mark.parametrize("field", [0, 1])
@pytest.mark.parametrize(
    "terminal_psi, time_dependent, gpu, vectorized",
    [(0, True, False, True), (1, False, False, False), (1, True, True, True)],
)
def test_source_drain_current(
    transport_device,
    current,
    field,
    terminal_psi,
    time_dependent,
    gpu,
    vectorized,
):
    device = transport_device
    total_time = 10
    skip_time = 1

    if gpu and cupy is None:
        options = tdgl.SolverOptions(
            solve_time=total_time,
            skip_time=skip_time,
            field_units="uT",
            current_units="uA",
            save_every=100,
            terminal_psi=terminal_psi,
            gpu=gpu,
        )
        with pytest.raises(SolverOptionsError):
            options.validate()
        return

    options = tdgl.SolverOptions(
        solve_time=total_time,
        skip_time=skip_time,
        field_units="uT",
        current_units="uA",
        save_every=100,
        terminal_psi=terminal_psi,
        gpu=gpu,
    )

    options.sparse_solver = "unknown"
    with pytest.raises(SolverOptionsError):
        options.validate()
    options.sparse_solver = "superlu"
    options.validate()

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

    if vectorized:

        def disorder_epsilon(r):
            return 1.0 * np.ones(len(r))

    else:

        def disorder_epsilon(r):
            return 1.0

    if time_dependent:
        ramp = tdgl.sources.LinearRamp(tmin=1, tmax=8)
        constant_field = tdgl.sources.ConstantField(
            field,
            field_units=options.field_units,
            length_units=device.length_units,
        )
        field = ramp * constant_field
        field = constant_field * ramp

        _disorder_epsilon = disorder_epsilon

        def disorder_epsilon(r, *, t, vectorized=vectorized):
            return _disorder_epsilon(r)

    solution = tdgl.solve(
        device,
        options,
        disorder_epsilon=disorder_epsilon,
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


@pytest.fixture
def screening_device() -> tdgl.Device:
    length_units = "um"
    xi = 0.1
    london_lambda = 0.075
    thickness = 0.05

    height = 1
    width = 2

    layer = tdgl.Layer(
        coherence_length=xi, london_lambda=london_lambda, thickness=thickness
    )
    film = tdgl.Polygon("film", points=tdgl.geometry.box(width, height, points=301))
    device = tdgl.Device(
        "bar",
        layer=layer,
        film=film,
        length_units=length_units,
    )
    device.make_mesh(max_edge_length=xi / 2, smooth=100)
    return device


def test_screening(screening_device: tdgl.Device):
    numba.set_num_threads(2)
    device = screening_device

    fluxoid_curves = [
        circle(0.25, center=(0, 0)),
        circle(0.1, center=(0.15, 0.25)),
        circle(0.3, center=(0.6, -0.1)),
        box(0.5, center=(-0.5, 0)),
        box(0.5, center=(-0.6, -0.2)),
    ]

    options = tdgl.SolverOptions(
        solve_time=2,
        field_units="mT",
        current_units="uA",
        include_screening=False,
        monitor=True,
    )

    no_screening_solution = tdgl.solve(device, options, applied_vector_potential=0.1)
    K = no_screening_solution.current_density
    K_max = np.sqrt(K[:, 0] ** 2 + K[:, 1] ** 2).max().to("uA / um").magnitude

    assert np.isclose(K_max, 450, rtol=5e-2)

    for curve in fluxoid_curves:
        fluxoid = no_screening_solution.polygon_fluxoid(curve)
        total_fluxoid = sum(fluxoid).magnitude
        error = abs(total_fluxoid / fluxoid.flux_part.magnitude)
        assert error > 1

    options.include_screening = True
    options.screening_tolerance = 1e-6
    options.dt_max = 1e-3

    screening_solution = tdgl.solve(device, options, applied_vector_potential=0.1)
    K = screening_solution.current_density
    K_max = np.sqrt(K[:, 0] ** 2 + K[:, 1] ** 2).max().to("uA / um").magnitude
    assert np.isclose(K_max, 270, rtol=2e-2)

    for curve in fluxoid_curves:
        fluxoid = screening_solution.polygon_fluxoid(curve)
        total_fluxoid = sum(fluxoid).magnitude
        error = abs(total_fluxoid / fluxoid.flux_part.magnitude)
        assert error < 5e-2


def test_tdgl_solver_invalid_current_units(transport_device):
    options = tdgl.SolverOptions(solve_time=1, current_units="mT")
    with pytest.raises(ValueError, match="current_units"):
        tdgl.TDGLSolver(transport_device, options)


def test_solve_for_psi_squared_invalid_psi_type():
    with pytest.raises(TypeError, match="psi must be a numpy or cupy array"):
        tdgl.TDGLSolver.solve_for_psi_squared(
            psi=[1.0],
            abs_sq_psi=None,
            mu=None,
            epsilon=None,
            gamma=1,
            u=1,
            dt=1,
            psi_laplacian=None,
        )


def test_tdgl_solver_missing_optional_dependencies(monkeypatch, transport_device):
    # validate() would raise first, so skip it to reach the solver's own checks.
    monkeypatch.setattr(tdgl.SolverOptions, "validate", lambda self: None)

    monkeypatch.setattr("tdgl.solver.solver.cupy", None)
    options = tdgl.SolverOptions(solve_time=1, gpu=True)
    with pytest.raises(SolverOptionsError, match="CuPy"):
        tdgl.TDGLSolver(transport_device, options)

    monkeypatch.setattr("tdgl.solver.solver.pypardiso", None)
    options = tdgl.SolverOptions(solve_time=1, sparse_solver=SparseSolver.PARDISO)
    with pytest.raises(SolverOptionsError, match="pypardiso"):
        tdgl.TDGLSolver(transport_device, options)


def test_mesh_operators_require_cupy(monkeypatch, transport_device):
    monkeypatch.setattr("tdgl.finite_volume.operators.cupy", None)
    mesh = transport_device.mesh

    operators = MeshOperators(mesh, SparseSolver.SUPERLU, use_cupy=True)
    with pytest.raises(SolverOptionsError, match="CuPy"):
        operators.build_operators()

    operators = MeshOperators(mesh, SparseSolver.CUPY, use_cupy=False)
    with pytest.raises(SolverOptionsError, match="SparseSolver.CUPY requires"):
        operators.build_operators()
