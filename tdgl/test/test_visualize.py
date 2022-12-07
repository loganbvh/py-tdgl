import os
import subprocess
import tempfile
from typing import List

import matplotlib.pyplot as plt
import pytest

import tdgl
from tdgl import visualize
from tdgl.solution.plot_solution import non_gui_backend
from tdgl.visualization.animate import animate
from tdgl.visualization.interactive_plot import Quantity, _default_quantities


@pytest.fixture(scope="module")
def tempdir():
    tmp = tempfile.TemporaryDirectory()
    yield tmp.__enter__()
    tmp.cleanup()


@pytest.fixture(scope="module")
def solution(transport_device, tempdir):
    device = transport_device
    total_time = 10

    fname = os.path.join(tempdir, "output.h5")
    options = tdgl.SolverOptions(
        solve_time=total_time,
        output_file=fname,
        field_units="uT",
        current_units="uA",
    )
    field = tdgl.sources.ConstantField(10)
    solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=field,
        terminal_currents=dict(source=10, drain=-10),
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
    "quantities", [None, "all"] + [name.lower() for name in Quantity.get_keys()]
)
@pytest.mark.parametrize("allow_save", [False, True])
@pytest.mark.parametrize("silent", [False, True])
@pytest.mark.parametrize("verbose", [False, True])
def test_interactive(solution, quantities, verbose, silent, allow_save):
    parser = visualize.make_parser()
    cmd = ["--input", solution.path, "interactive"]
    if verbose:
        cmd.insert(2, "--verbose")
    if silent:
        cmd.insert(2, "--silent")
    if allow_save:
        cmd.append("--allow-save")
    if quantities is not None:
        cmd.extend(["--quantities"] + quantities.split(" "))
    args = parser.parse_args(cmd)
    with non_gui_backend():
        tdgl.visualize.main(args)
        plt.close("all")


@pytest.mark.parametrize("quantities", [None, "order_parameter phase", "supercurrent"])
@pytest.mark.parametrize("ext", ["-m.gif"])
@pytest.mark.parametrize("max_cols", [4, 2])
def test_animate(solution, quantities, ext, max_cols):
    kwargs = dict(
        output_file=solution.path.replace(".h5", ext),
        full_title=False,
        dpi=200,
        fps=20,
        max_cols=max_cols,
    )
    if quantities is None:
        kwargs["quantities"] = _default_quantities
    if quantities is not None:
        kwargs["quantities"] = quantities.split(" ")
    animate(solution.path, **kwargs)


@pytest.mark.parametrize(
    "quantities", [None, "all", "order_parameter phase vorticity supercurrent"]
)
def test_animate_cli(solution, quantities):
    parser = visualize.make_parser()
    cmd = [
        "--input",
        solution.path,
        "animate",
        "--output",
        solution.path.replace(".h5", "-cli.gif"),
        "--min-frame",
        "2",
        "--max-frame",
        "-1",
        "--fps",
        "30",
        "--dpi",
        "200",
    ]
    if quantities is not None:
        cmd.extend(["--quantities"] + quantities.split(" "))
    args = parser.parse_args(cmd)
    with non_gui_backend():
        tdgl.visualize.main(args)
        plt.close("all")
