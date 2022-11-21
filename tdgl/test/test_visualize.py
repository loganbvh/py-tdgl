import os
import subprocess
import tempfile
from typing import List

import matplotlib.pyplot as plt
import pytest

import tdgl
from tdgl import visualize
from tdgl.solution.plot_solution import non_gui_backend


@pytest.fixture(scope="module")
def tempdir():
    tmp = tempfile.TemporaryDirectory()
    yield tmp.__enter__()
    tmp.cleanup()


@pytest.fixture(scope="module")
def solution(transport_device, tempdir):
    device = transport_device
    dt = 1e-3
    total_time = 10

    options = tdgl.SolverOptions(
        dt_init=dt,
        solve_time=total_time,
        save_every=100,
    )
    field = tdgl.sources.ConstantField(10)
    fname = os.path.join(tempdir, "output.h5")
    solution = tdgl.solve(
        device,
        options,
        output_file=fname,
        applied_vector_potential=field,
        field_units="uT",
        terminal_currents=dict(source=10, drain=-10),
        current_units="uA",
    )
    return solution


def run_cmd(cmd: List[str]) -> None:
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stderr)
    print(result.stdout)


def test_visualize_help():
    run_cmd(["python", "-m", "tdgl.visualize", "-h"])


def test_no_args():
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd(["python", "-m", "tdgl.visualize"])


def test_bad_input():
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd(["python", "-m", "tdgl.visualize"])


@pytest.mark.parametrize(
    "observables", [None, "all", "complex_field phase vorticity supercurrent"]
)
@pytest.mark.parametrize("allow_save", [False, True])
@pytest.mark.parametrize("silent", [False, True])
@pytest.mark.parametrize("verbose", [False, True])
def test_interactive(solution, observables, verbose, silent, allow_save):
    parser = visualize.make_parser()
    cmd = ["--input", solution.path, "interactive"]
    if verbose:
        cmd.insert(2, "--verbose")
    if silent:
        cmd.insert(2, "--silent")
    if allow_save:
        cmd.append("--allow-save")
    if observables is not None:
        cmd.extend(["--observables"] + observables.split(" "))
    args = parser.parse_args(cmd)
    with non_gui_backend():
        tdgl.visualize.main(args)
        plt.close("all")
