# flake8: noqa
import os
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

import tdgl
from tdgl.geometry import box


def make_device():
    length_units = "um"
    length = 20
    width = 10
    layer = tdgl.Layer(london_lambda=4, coherence_length=0.1, thickness=0.3)
    film = tdgl.Polygon("film", points=box(width, length)).resample(2001)

    source = tdgl.Polygon("source", points=box(width, length / 100)).translate(
        dy=length / 2
    )
    drain = source.translate(dy=-length).set_name("drain")

    device = tdgl.Device(
        "strip",
        layer=layer,
        film=film,
        terminals=[source, drain],
        probe_points=[(0, 0.75 * length / 2), (0, -0.75 * length / 2)],
        length_units=length_units,
    )
    return device


def make_mesh(device: tdgl.Device):
    t0 = time.perf_counter()
    device.make_mesh(max_edge_length=device.layer.coherence_length / 2)
    t1 = time.perf_counter()
    print(f"Finished making mesh in {t1-t0:.2f} seconds")
    print(device.mesh_stats_dict())


def epsilon(r, r0=(0, 0), radius=1, epsilon0=-1, epsilon1=1):
    """Set the disorder parameter $\\epsilon$ for position r."""
    x, y = r
    x0, y0 = r0
    if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) <= radius:
        return epsilon0
    return epsilon1


def main(solver: str):
    device = make_device()
    make_mesh(device)

    current = 150

    options = tdgl.SolverOptions(
        solve_time=20,
        current_units="uA",
        field_units="mT",
        save_every=100,
        dt_max=2e-2,
        sparse_solver=solver,
    )

    solution = tdgl.solve(
        device,
        options,
        terminal_currents=dict(source=current, drain=-current),
        disorder_epsilon=lambda r: epsilon(
            r, r0=(-2.5, 0), radius=1, epsilon0=-1, epsilon1=1
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", type=str, choices=["superlu", "umfpack", "pardiso", "cupy"]
    )
    args = parser.parse_args()
    main(args.solver)
