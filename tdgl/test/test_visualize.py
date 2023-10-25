import os
import subprocess
import tempfile
from typing import List

import matplotlib.pyplot as plt
import pytest

import tdgl
from tdgl import visualize
from tdgl.visualization import DEFAULT_QUANTITIES, Quantity, create_animation


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
@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("dimensionless", [True, False])
def test_interactive(solution, quantities, verbose, dimensionless):
    parser = visualize.make_parser()
    cmd = ["--input", solution.path]
    if verbose:
        cmd.append("--verbose")
    if dimensionless:
        cmd.append("--dimensionless")
    cmd.append("--axis-labels")
    cmd.append("interactive")
    if quantities is not None:
        cmd.extend(["--quantities"] + quantities.split(" "))
    args = parser.parse_args(cmd)
    with tdgl.non_gui_backend():
        tdgl.visualize.main(args)
        plt.close("all")


@pytest.mark.parametrize("quantities", [None, "order_parameter phase", "supercurrent"])
@pytest.mark.parametrize("ext", ["-m.gif"])
@pytest.mark.parametrize("max_cols", [4, 2])
def test_animation(solution, quantities, ext, max_cols):
    kwargs = dict(
        output_file=solution.path.replace(".h5", ext),
        full_title=False,
        dpi=200,
        fps=20,
        axes_off=True,
        shading="flat",
        max_cols=max_cols,
    )
    if quantities is None:
        kwargs["quantities"] = DEFAULT_QUANTITIES
    if quantities is not None:
        kwargs["quantities"] = quantities.split(" ")
    create_animation(solution.path, **kwargs)


@pytest.mark.parametrize(
    "quantities",
    [
        "all",
        # None,
        # "order_parameter phase vorticity supercurrent",
    ],
)
@pytest.mark.parametrize("autoscale", [True, False])
@pytest.mark.parametrize("dimensionless", [True, False])
def test_animate_cli(solution, quantities, autoscale, dimensionless):
    parser = visualize.make_parser()
    cmd = ["--input", solution.path]
    if dimensionless:
        cmd.append("--dimensionless")
    if autoscale:
        cmd.append("--autoscale")
    cmd.append("--axis-labels")
    cmd.extend(
        [
            "--output",
            solution.path.replace(".h5", "-cli.gif"),
            "--shading",
            "flat",
            "--axis-labels",
            "--dpi",
            "200",
            "animate",
            "--min-frame",
            "2",
            "--max-frame",
            "-1",
            "--fps",
            "30",
        ]
    )

    if quantities is not None:
        cmd.extend(["--quantities"] + quantities.split(" "))
    args = parser.parse_args(cmd)
    with tdgl.non_gui_backend():
        tdgl.visualize.main(args)
        plt.close("all")


@pytest.mark.parametrize(
    "quantities",
    [
        "all",
        # None,
        # "order_parameter phase vorticity supercurrent",
    ],
)
@pytest.mark.parametrize("autoscale", [True, False])
@pytest.mark.parametrize("dimensionless", [True, False])
def test_snapshot_cli(solution, quantities, autoscale, dimensionless):
    parser = visualize.make_parser()
    cmd = ["--input", solution.path]
    if dimensionless:
        cmd.append("--dimensionless")
    if autoscale:
        cmd.append("--autoscale")
    cmd.append("--axis-labels")
    cmd.extend(
        [
            "--shading",
            "flat",
            "--axis-labels",
            "--dpi",
            "200",
            "snapshot",
            "--times",
        ]
        + "1 2 5 10".split(" ")
    )

    if quantities is not None:
        cmd.extend(["--quantities"] + quantities.split(" "))
    args = parser.parse_args(cmd)
    with tdgl.non_gui_backend():
        tdgl.visualize.main(args)
        plt.close("all")


@pytest.mark.parametrize("output", [None, "auto"])
def test_convert_cli(solution, output):
    parser = visualize.make_parser()
    cmd = ["--input", solution.path]
    output = solution.path.replace(".h5", "-converted.xdmf")
    output_h5 = os.path.join(
        os.getcwd(), os.path.basename(output).replace(".xdmf", ".h5")
    )
    if output == "auto":
        cmd.extend(["--output", output])
    cmd.extend(["convert", "--format", "xdmf"])
    args = parser.parse_args(cmd)
    with tdgl.non_gui_backend():
        tdgl.visualize.main(args)
        plt.close("all")
    if os.path.exists(output):
        os.remove(output)
    if os.path.exists(output_h5):
        os.remove(output_h5)
