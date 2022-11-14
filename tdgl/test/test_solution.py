import os
import tempfile

import pint
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
        dt_init=dt,
        solve_time=total_time,
        save_every=100,
    )
    field = tdgl.sources.ConstantField(10)
    fname = os.path.join(tempdir, "output.h5")

    def terminal_currents(time):
        return dict(source=10, drain=-10)

    solution = tdgl.solve(
        device,
        fname,
        options,
        applied_vector_potential=field,
        field_units="uT",
        terminal_currents=terminal_currents,
        current_units="uA",
        include_screening=False,
    )

    return solution


def test_save_and_load_solution(solution):
    solution.to_hdf5()
    loaded_solution = tdgl.Solution.from_hdf5(solution.path)
    assert loaded_solution == solution


@pytest.mark.parametrize("hole", ["hole1", "hole2", "invalid"])
@pytest.mark.parametrize("with_units", [False, True])
@pytest.mark.parametrize("units", ["Phi_0", "mT * um**2"])
def test_hole_fluxoid(solution, hole, with_units, units):
    if hole == "invalid":
        with pytest.raises(KeyError):
            _ = solution.hole_fluxoid(hole, units=units, with_units=with_units)
    else:
        fluxoid = solution.hole_fluxoid(hole, units=units, with_units=with_units)
        assert isinstance(fluxoid, tdgl.solution.Fluxoid)
        if with_units:
            assert isinstance(sum(fluxoid), pint.Quantity)
        else:
            assert isinstance(sum(fluxoid), float)
