import os
import tempfile

import pytest

import tdgl


@pytest.fixture(scope="module")
def tempdir():
    tmp = tempfile.TemporaryDirectory()
    yield tmp.__enter__()
    tmp.cleanup()


@pytest.fixture(scope="module")
def solution(transport_device, tempdir):
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
    fname = os.path.join(tempdir, "output.h5")
    solution = tdgl.solve(
        device,
        fname,
        options,
        applied_vector_potential=field,
        field_units="uT",
        gamma=10,
        source_drain_current=lambda t: 10,
        current_units="uA",
        include_screening=False,
    )

    return solution


def test_save_and_load_solution(solution):
    solution.to_hdf5()
    loaded_solution = tdgl.Solution.from_hdf5(solution.path)
    assert loaded_solution == solution
