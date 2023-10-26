import os
import tempfile

import h5py
import numpy as np
import pint
import pytest

import tdgl
from tdgl.solution.data import DynamicsData


@pytest.fixture(scope="module")
def tempdir():
    tmp = tempfile.TemporaryDirectory()
    yield tmp.__enter__()
    tmp.cleanup()


@pytest.fixture(scope="module")
def solution(transport_device, tempdir):
    device = transport_device
    total_time = 100

    fname = os.path.join(tempdir, "output.h5")
    options = tdgl.SolverOptions(
        solve_time=total_time,
        output_file=fname,
        save_every=100,
        field_units="uT",
        current_units="uA",
    )
    field = tdgl.sources.ConstantField(1)

    def terminal_currents(time):
        return dict(source=0.1, drain=-0.1)

    solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=field,
        terminal_currents=terminal_currents,
    )

    _, phases = solution.boundary_phases()[device.film.name]
    assert np.isclose((phases[-1] - phases[0]) / (2 * np.pi), 0, atol=5e-2)

    _, phases = solution.boundary_phases(delta=True)[device.film.name]
    assert np.isclose(phases[-1] / (2 * np.pi), 0, atol=5e-2)
    return solution


def test_save_and_load_solution(solution, tempdir):
    path = os.path.join(tempdir, "output-1.h5")
    solution.to_hdf5(path)
    loaded_solution = tdgl.Solution.from_hdf5(path)
    assert loaded_solution == solution

    solution.to_hdf5()
    loaded_solution = tdgl.Solution.from_hdf5(solution.path)
    assert loaded_solution == solution


@pytest.mark.parametrize("hole", ["hole1", "hole2", "invalid"])
@pytest.mark.parametrize("with_units", [False, True])
@pytest.mark.parametrize("units", ["Phi_0", "mT * um**2"])
@pytest.mark.parametrize("interp_method", ["linear", "cubic"])
def test_hole_fluxoid(solution, hole, with_units, units, interp_method):
    if hole == "invalid":
        with pytest.raises(KeyError):
            _ = solution.hole_fluxoid(
                hole, units=units, with_units=with_units, interp_method=interp_method
            )
    else:
        fluxoid = solution.hole_fluxoid(
            hole, units=units, with_units=with_units, interp_method=interp_method
        )
        assert isinstance(fluxoid, tdgl.Fluxoid)
        if with_units:
            assert isinstance(sum(fluxoid), pint.Quantity)
        else:
            assert isinstance(sum(fluxoid), float)


def test_tdgl_data(solution: tdgl.Solution):
    tdgl_data = solution.tdgl_data
    with h5py.File(solution.path, "r+") as f:
        tdgl_data.to_hdf5(f.create_group("tdgl_data"))


def test_dynamics(solution: tdgl.Solution):
    dynamics = solution.dynamics
    ix = dynamics.time_slice()
    ts = dynamics.time[ix]
    assert np.array_equal(ts, dynamics.time)

    ix = dynamics.time_slice(tmin=10, tmax=90)
    ts = dynamics.time[ix]
    assert np.all(ts >= 10)
    assert np.all(ts <= 90)

    V0 = dynamics.mean_voltage()
    d2 = dynamics.resample()
    dt = np.diff(d2.time)
    assert np.allclose(dt, dt[0])
    V1 = d2.mean_voltage()

    assert np.isclose(V0, V1, rtol=1e-2)

    with tdgl.non_gui_backend():
        _ = dynamics.plot(legend=True)
        _ = dynamics.plot_dt()

    with h5py.File(solution.path, "r+") as f:
        dynamics.to_hdf5(f.create_group("dynamics"))
        loaded_dynamics = DynamicsData.from_hdf5(f["dynamics"])

    assert loaded_dynamics == dynamics

    time = solution.times
    assert len(time) == (solution.data_range[1] + 1)
    assert solution.closest_solve_step(0) == 0

    _ = DynamicsData.from_solution(solution.path, probe_points=None, progress_bar=True)
    _ = DynamicsData.from_solution(
        solution.path, probe_points=solution.device.probe_points, progress_bar=False
    )


@pytest.mark.parametrize("dataset", [None, "supercurrent", "normal_current", "invalid"])
@pytest.mark.parametrize("interp_method", ["linear", "cubic", "invalid"])
def test_get_current_through_paths(solution: tdgl.Solution, dataset, interp_method):
    ys = np.linspace(-2, 2, 101)
    xs = 7.5 * np.ones_like(ys)
    paths = [
        np.array([+xs, ys]).T,
        np.array([-xs, ys]).T,
    ]

    Isrc = solution.terminal_currents(0)["source"]

    if dataset == "invalid" or interp_method == "invalid":
        with pytest.raises(ValueError):
            times, currents = tdgl.get_current_through_paths(
                solution.path,
                paths[0],
                dataset=dataset,
                interp_method=interp_method,
            )
    else:
        (fig, ax), (times, currents) = tdgl.plot_current_through_paths(
            solution.path,
            paths[0],
            dataset=dataset,
            interp_method=interp_method,
            progress_bar=False,
        )
        assert len(currents) == 1
        assert len(times) == len(currents[0])
        for cs in currents:
            assert isinstance(cs[0], pint.Quantity)
            if dataset is None:
                assert np.allclose(cs.m[1:], Isrc, rtol=3e-2)


@pytest.mark.parametrize("units", [None, "A * m**2"])
@pytest.mark.parametrize("with_units", [False, True])
def test_magnetic_moment(solution: tdgl.Solution, units, with_units):
    device = solution.device
    m = solution.magnetic_moment(units=units, with_units=with_units)
    if with_units:
        assert isinstance(m, pint.Quantity)
        if units is None:
            units = f"{solution.current_units} * {device.length_units}**2"
        assert m.units == device.ureg(units)
    else:
        assert isinstance(m, float)
