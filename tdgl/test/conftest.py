import pytest

import tdgl
from tdgl.geometry import box, circle


@pytest.fixture(scope="package")
def transport_device():
    london_lambda = 2
    xi = 1
    d = 0.1

    layer = tdgl.Layer(coherence_length=xi, london_lambda=london_lambda, thickness=d)
    film = (
        tdgl.Polygon("film", points=box(10)).union(box(30, 4, points=400)).resample(501)
    )
    hole = tdgl.Polygon("hole1", points=circle(1.5, center=(2, 2)))
    source = tdgl.Polygon(points=box(1e-2, 4, center=(-15, 0))).set_name("source")
    drain = source.scale(xfact=-1).set_name("drain")

    device = tdgl.Device(
        "film",
        layer=layer,
        film=film,
        holes=[hole, hole.scale(xfact=-1, yfact=-1).set_name("hole2")],
        terminals=[source, drain],
        probe_points=[(-10, 0), (10, 0)],
    )

    assert device.mesh is None
    assert device.points is None
    assert device.triangles is None
    assert device.edges is None
    assert device.edge_lengths is None
    assert device.areas is None
    assert device.probe_point_indices is None
    assert device.boundary_sites() is None

    _ = device.mesh_stats_dict()
    _ = device.mesh_stats()

    device.make_mesh(min_points=2000, smooth=100, max_edge_length=xi / 2)

    _ = device.areas
    _ = device.boundary_sites()
    _ = device.mesh_stats_dict()
    _ = device.mesh_stats()

    return device


@pytest.fixture(scope="package")
def transport_device_solution(transport_device):
    device = transport_device
    dt = 1e-3
    total_time = 100

    options = tdgl.SolverOptions(
        dt_init=dt,
        solve_time=total_time,
        save_every=100,
        field_units="uT",
        current_units="uA",
        include_screening=False,
    )
    field = tdgl.sources.ConstantField(10)
    solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=field,
        terminal_currents=dict(source=10, drain=-10),
    )
    return solution


@pytest.fixture(scope="package")
def box_device():
    london_lambda = 1.5
    xi = 1.5
    d = 0.1
    layer = tdgl.Layer(coherence_length=xi, london_lambda=london_lambda, thickness=d)
    film = tdgl.Polygon("film", points=box(10))
    device = tdgl.Device("film", layer=layer, film=film)
    device.make_mesh(min_points=2000, smooth=40, max_edge_length=xi / 2)
    return device


@pytest.fixture(scope="package")
def box_device_solution_no_screening(box_device):
    device = box_device
    dt = 1e-3
    total_time = 20

    options = tdgl.SolverOptions(
        dt_init=dt,
        solve_time=total_time,
        save_every=100,
        field_units="uT",
        current_units="uA",
        include_screening=False,
    )
    solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=tdgl.sources.ConstantField(50),
        terminal_currents=None,
    )
    return solution


@pytest.fixture(scope="package")
def box_device_solution_with_screening(box_device):
    device = box_device
    dt = 1e-3
    total_time = 20

    options = tdgl.SolverOptions(
        dt_init=dt,
        solve_time=total_time,
        save_every=100,
        field_units="uT",
        current_units="uA",
        include_screening=True,
    )
    solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=tdgl.sources.ConstantField(50),
        terminal_currents=None,
    )
    return solution
