import os
import tempfile

import pytest

import tdgl
from tdgl.geometry import box, circle


@pytest.fixture(scope="package")
def transport_device():
    london_lambda = 5
    xi = 1
    d = 0.5

    layer = tdgl.Layer(coherence_length=xi, london_lambda=london_lambda, thickness=d)
    film = tdgl.Polygon("film", points=box(10)).union(box(30, 4))
    hole = tdgl.Polygon("hole1", points=circle(1.5, center=(2, 2)))
    source = tdgl.Polygon(points=box(1e-2, 4, center=(-15, 0)))
    drain = source.scale(xfact=-1)

    device = tdgl.Device(
        "film",
        layer=layer,
        film=film,
        holes=[hole, hole.scale(xfact=-1, yfact=-1).rename("hole2")],
        source_terminal=source,
        drain_terminal=drain,
    )
    device.make_mesh(min_points=1000, optimesh_steps=40, max_edge_length=xi / 2)
    return device


@pytest.fixture(scope="package")
def transport_device_solution(transport_device):
    device = transport_device
    dt = 1e-3
    total_time = 100

    options = tdgl.SolverOptions(
        dt_min=dt,
        dt_max=10 * dt,
        total_time=total_time,
        adaptive_window=1,
        save_every=100,
    )
    field = tdgl.sources.ConstantField(10)
    with tempfile.TemporaryDirectory() as directory:
        fname = os.path.join(directory, "output.h5")
        solution = tdgl.solve(
            device,
            fname,
            options,
            applied_vector_potential=field,
            field_units="uT",
            gamma=10,
            source_drain_current=10,
            current_units="uA",
            include_screening=False,
        )

    return solution


@pytest.fixture(scope="package")
def box_device():
    london_lambda = 5
    xi = 1.5
    d = 0.1
    layer = tdgl.Layer(coherence_length=xi, london_lambda=london_lambda, thickness=d)
    film = tdgl.Polygon("film", points=box(10))
    device = tdgl.Device("film", layer=layer, film=film)
    device.make_mesh(min_points=2000, optimesh_steps=40, max_edge_length=xi / 2)
    return device


@pytest.fixture(scope="package")
def box_device_solution_no_screening(box_device):
    device = box_device
    dt = 1e-3
    total_time = 20

    options = tdgl.SolverOptions(
        dt_min=dt,
        dt_max=10 * dt,
        skip_time=0 * total_time / 2,
        total_time=total_time,
        adaptive_window=1,
        save_every=100,
    )
    solution = tdgl.solve(
        device,
        "output1.h5",
        options,
        applied_vector_potential=tdgl.sources.ConstantField(50),
        field_units="uT",
        gamma=10,
        source_drain_current=0,
        current_units="uA",
        include_screening=False,
    )
    return solution


@pytest.fixture(scope="package")
def box_device_solution_with_screening(box_device):
    device = box_device
    dt = 1e-3
    total_time = 20

    options = tdgl.SolverOptions(
        dt_min=dt,
        dt_max=10 * dt,
        skip_time=0 * total_time / 2,
        total_time=total_time,
        adaptive_window=1,
        save_every=100,
    )
    solution = tdgl.solve(
        device,
        "output1.h5",
        options,
        applied_vector_potential=tdgl.sources.ConstantField(50),
        field_units="uT",
        gamma=10,
        source_drain_current=0,
        current_units="uA",
        include_screening=True,
    )
    return solution
