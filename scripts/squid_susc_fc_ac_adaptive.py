import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from itertools import count
from operator import itemgetter
from typing import Any, Dict, Tuple

import adaptive
import h5py
import numpy as np
import superscreen as sc
from mpi4py.futures import MPIPoolExecutor

import squids
import tdgl
from tdgl.core.visualization.helpers import get_data_range

squid_funcs = {
    "ibm-small": squids.ibm.small.make_squid,
    "ibm-medium": squids.ibm.medium.make_squid,
    "ibm-large": squids.ibm.large.make_squid,
    "ibm-xlarge": squids.ibm.xlarge.make_squid,
    "huber": squids.huber.make_squid,
    "hypres-small": squids.hypres.small.make_squid,
}


logger = logging.getLogger(os.path.basename(__file__).replace(".py", ""))
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def make_film(
    radius,
    xi,
    lambda_,
    d,
    loop_radius,
    min_points,
    optimesh_steps,
) -> tdgl.Device:
    layer = tdgl.Layer(coherence_length=xi, london_lambda=lambda_, thickness=d, z0=0)
    film = tdgl.Polygon("film", points=tdgl.geometry.circle(radius, points=101))
    abstract_regions = [
        tdgl.Polygon("loop", points=tdgl.geometry.circle(loop_radius, points=301)),
    ]
    device = tdgl.Device(
        "box",
        layer=layer,
        film=film,
        abstract_regions=abstract_regions,
        length_units="um",
    )
    device.make_mesh(min_points=min_points, optimesh_steps=optimesh_steps)
    return device


def make_squid(name, min_points, optimesh_steps, angle) -> sc.Device:
    squid = squid_funcs[name]().rotate(angle)
    squid.make_mesh(min_points=min_points, optimesh_steps=optimesh_steps)
    return squid


def get_base_squid_solution(squid, iterations) -> sc.Solution:
    return sc.solve(
        squid,
        circulating_currents=dict(fc_center=1.0),
        current_units="mA",
        field_units="mT",
        iterations=iterations,
    )[-1]


def applied_potential(
    x,
    y,
    z,
    *,
    path_to_solution,
    I_fc,
    r0=(0, 0, 0),
    current_units="mA",
    field_units="mT",
    length_units="um",
) -> np.ndarray:
    solution = sc.Solution.from_file(path_to_solution, compute_matrices=True)
    r0 = np.atleast_2d(r0)
    if len(z) == 1:
        z = z[0] * np.ones_like(x)
    positions = np.array([x.squeeze(), y.squeeze(), z.squeeze()]).T - r0
    I_circ = solution.circulating_currents["fc_center"] * tdgl.ureg(
        solution.current_units
    )
    I_fc = I_fc * tdgl.ureg(current_units)
    scale = (I_fc / I_circ).to_base_units().magnitude
    A = solution.vector_potential_at_position(
        positions,
        units=f"{field_units} * {length_units}",
        with_units=False,
    )
    return scale * A


def applied_field_from_film(
    x, y, z, *, tdgl_solution: tdgl.Solution, field_units: str = "mT"
):
    if len(z) == 1:
        z = z[0] * np.ones_like(x)
    positions = np.array([x.squeeze(), y.squeeze(), z.squeeze()]).T
    return tdgl_solution.field_at_position(
        positions,
        vector=False,
        units=field_units,
        with_units=False,
        return_sum=True,
    )


def calculate_pl_flux(
    squid: sc.Device, tdgl_solution: tdgl.Solution, iterations: int = 5
) -> float:
    applied_field = sc.Parameter(
        applied_field_from_film,
        tdgl_solution=tdgl_solution,
        field_units="mT",
    )
    solution = sc.solve(
        squid,
        applied_field=applied_field,
        iterations=iterations,
    )[-1]
    fluxoid = solution.hole_fluxoid("pl_center")
    return sum(fluxoid).to("Phi_0").magnitude


def get_ac_flux(
    I_fc_rms: float,
    *,
    squid: sc.Device,
    device: tdgl.Device,
    solver_options: tdgl.SolverOptions,
    base_path: str,
    squid_solution_path: str,
    metadata: Dict[str, Any],
    squid_position: Tuple[float, float, float] = (0, 0, 0),
    squid_iterations: int = 5,
    gamma: float = 10,
    cycles: float = 1.0,
    points_per_cycle: int = 25,
    skip_cycles: float = 0.0,
    seed_solutions: bool = True,
    screening: bool = False,
    field_units: str = "mT",
    current_units: str = "uA",
    length_units: str = "um",
):
    assert skip_cycles < cycles
    for i in count():
        path = os.path.join(base_path, f"{i:03}")
        if not os.path.exists(path):
            os.mkdir(path)
            break
    npoints = int(points_per_cycle * cycles) + 1
    thetas = np.linspace(0, 2 * np.pi * cycles, npoints)
    I_fc = I_fc_rms * np.sqrt(2) * np.cos(thetas)

    all_flux = []

    args_as_dict = metadata.copy()
    args_as_dict["I_fc_rms"] = I_fc_rms

    prev_solution = None
    start_time = datetime.now()

    with h5py.File(
        os.path.join(path, "steady-state.h5"),
        "x",
        libver="latest",
    ) as f:
        device.mesh.save_to_hdf5(f.create_group("mesh"))

    steps = solver_options.min_steps
    total_time = solver_options.total_time

    solution_paths = []

    for i, current in enumerate(I_fc):

        solver_options.min_steps = steps
        solver_options.total_time = total_time

        A_applied = tdgl.Parameter(
            applied_potential,
            path_to_solution=squid_solution_path,
            r0=squid_position,
            I_fc=current,
            field_units=field_units,
            current_units=current_units,
            length_units=length_units,
        )

        tdgl_solution = tdgl.solve(
            device,
            A_applied,
            os.path.join(path, f"output-{i}.h5"),
            options=solver_options,
            field_units=field_units,
            gamma=gamma,
            source_drain_current=0,
            include_screening=screening,
            seed_solution=prev_solution,
        )
        tdgl_solution.to_hdf5()

        if seed_solutions:
            prev_solution = tdgl_solution
            if total_time is not None:
                total_time = max(10, int(solver_options.total_time / 5))
            else:
                steps = max(100, int(solver_options.min_steps / 5))

        flux = calculate_pl_flux(squid, tdgl_solution, iterations=squid_iterations)
        all_flux.append(flux)

        solution_paths.append(tdgl_solution.path)

        with h5py.File(tdgl_solution.path, "r+", libver="latest") as f:
            for key, val in args_as_dict.items():
                if val is not None:
                    f.attrs[key] = val
            f.attrs["pl_fluxoid_in_Phi_0"] = flux

            i_start, i_end = get_data_range(f)

            with h5py.File(os.path.join(path, "steady-state.h5"), "r+") as out:
                data_grp = out.require_group("data")
                f["data"].copy(str(i_end), data_grp, name=str(i))
                for key, val in args_as_dict.items():
                    if val is not None:
                        out.attrs[key] = val
                data_grp[str(i)].attrs["pl_fluxoid_in_Phi_0"] = flux

    end_time = datetime.now()

    Phi = np.array(all_flux)
    susc = (
        np.trapz((Phi * np.exp(-1j * thetas))[int(skip_cycles * points_per_cycle) :])
        / I_fc_rms
    )

    json_data = {}
    json_data["args"] = args_as_dict.copy()
    json_data["timing"] = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_seconds": (end_time - start_time).total_seconds(),
    }
    json_data["I_fc"] = I_fc.tolist()
    json_data["flux"] = Phi.tolist()
    json_data["susc"] = {"real": susc.real, "imag": susc.imag}

    with open(os.path.join(path, "results.json"), "w") as f:
        json.dump(json_data, f, indent=4, sort_keys=True)

    for p in solution_paths:
        try:
            os.remove(p)
        except Exception as e:
            logger.warning(f"Unable to remove {p!r}: {e}.")

    return {"abs_susc": np.abs(susc), **json_data}


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--ntasks", type=int, default=None)
    parser.add_argument("--max-loss", type=float, default=0.1)
    parser.add_argument(
        "--max-wall-time", type=float, default=1, help="Max wall time in hours."
    )

    sample_grp = parser.add_argument_group("sample")
    squid_grp = parser.add_argument_group("squid")
    tdgl_grp = parser.add_argument_group("tdgl")

    squid_grp.add_argument(
        "--squid-type",
        type=str,
        default="hypres-small",
        choices=list(squid_funcs),
    )
    squid_grp.add_argument(
        "--squid-points",
        type=int,
        default=3000,
        help="Minimum number of points in the SQUID mesh.",
    )
    squid_grp.add_argument(
        "--squid-optimesh",
        type=int,
        default=50,
        help="Number of optimesh steps for the SQUID mesh.",
    )
    squid_grp.add_argument(
        "--squid-angle",
        type=float,
        default=0,
        help="Angle by which to rotate the SQUID device, in degrees.",
    )
    squid_grp.add_argument(
        "--squid-position",
        type=float,
        nargs=3,
        default=0,
        help="SQUID (x, y, z) position in microns.",
    )
    squid_grp.add_argument(
        "--squid-iterations",
        type=int,
        default=5,
        help="Number of superscreen solve iterations.",
    )

    sample_grp.add_argument(
        "--film-radius", type=float, default=15, help="Film radius in microns."
    )
    sample_grp.add_argument(
        "--film-points",
        type=int,
        default=4000,
    )
    sample_grp.add_argument(
        "--film-optimesh",
        type=int,
        default=100,
    )
    sample_grp.add_argument(
        "--d", default=0.1, type=float, help="Film thickness in microns."
    )
    sample_grp.add_argument(
        "--lam",
        default=2,
        type=float,
        help="London penetration depth in microns.",
    )
    sample_grp.add_argument(
        "--xi",
        default=1,
        type=float,
        help="Coherence length in microns.",
    )
    sample_grp.add_argument(
        "--gamma", default=10, type=float, help="TDGL gamma parameter."
    )
    sample_grp.add_argument(
        "--pinning", default=0, type=float, help="Pinning sites per square micron."
    )
    sample_grp.add_argument(
        "--screening", action="store_true", help="Include screening."
    )

    tdgl_grp.add_argument("--directory", type=str, help="Output directory.")
    tdgl_grp.add_argument(
        "--I_fc",
        nargs=2,
        type=float,
        help="RMS field coil current in mA: start, stop.",
    )
    tdgl_grp.add_argument(
        "--cycles",
        default=1,
        type=float,
        help="Number of AC field cycles to simulate.",
    )
    tdgl_grp.add_argument(
        "--skip-cycles",
        default=0,
        type=float,
        help="Number of AC field cycles to skip when calculating complex susceptibility.",
    )
    tdgl_grp.add_argument(
        "--points-per-cycle",
        type=float,
        default=10,
        help="Number of current points per AC cycle.",
    )
    tdgl_grp.add_argument(
        "--seed-solutions",
        action="store_true",
        help="Seed each simulation with the previous solution.",
    )
    tdgl_grp.add_argument(
        "--dt-min", default=1e-3, type=float, help="Min. GL ODE time step."
    )
    tdgl_grp.add_argument(
        "--dt-max", default=None, type=float, help="Max. GL ODE time step."
    )
    tdgl_grp.add_argument(
        "--total-time",
        type=float,
        default=None,
        help="Total solve time in units of GL tau.",
    )
    tdgl_grp.add_argument("--steps", type=float, default=None, help="GL ODE steps.")
    tdgl_grp.add_argument(
        "--save-every", default=100, type=int, help="Save interval in steps."
    )
    tdgl_grp.add_argument(
        "--field-units",
        type=str,
        default="mT",
    )
    tdgl_grp.add_argument(
        "--current-units",
        type=str,
        default="mA",
    )

    args = parser.parse_args()
    args_as_dict = vars(args)
    for k, v in args_as_dict.items():
        print(f"{k}: {v}")

    base_path = os.path.abspath(args.directory)
    # if os.path.exists(path) and os.listdir(path):
    #     raise ValueError(f"Path {path!r} exists and is not empty.")
    os.makedirs(base_path, exist_ok=True)

    logger.info("Building SQUID and calculating field coil vector potential.")
    squid = make_squid(
        args.squid_type, args.squid_points, args.squid_optimesh, args.squid_angle
    )
    squid_solution_path = os.path.join(base_path, "squid_solution")
    if not os.path.exists(squid_solution_path):
        squid_solution = get_base_squid_solution(squid, args.squid_iterations)
        squid_solution.to_file(squid_solution_path)
    logger.info(repr(squid))

    field_units = args.field_units
    current_units = args.current_units
    length_units = "um"

    device = make_film(
        args.film_radius,
        args.xi,
        args.lam,
        args.d,
        3,
        args.film_points,
        args.film_optimesh,
    )

    I_fc_min, I_fc_max = args.I_fc

    steps = args.steps
    total_time = args.total_time

    options = tdgl.SolverOptions(
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        total_time=total_time,
        min_steps=steps,
        save_every=args.save_every,
    )

    function = partial(
        get_ac_flux,
        squid=squid,
        device=device,
        solver_options=options,
        base_path=base_path,
        squid_solution_path=squid_solution_path,
        metadata=args_as_dict,
        squid_position=args.squid_position,
        squid_iterations=args.squid_iterations,
        gamma=args.gamma,
        cycles=args.cycles,
        points_per_cycle=args.points_per_cycle,
        skip_cycles=args.skip_cycles,
        seed_solutions=args.seed_solutions,
        screening=args.screening,
        field_units=field_units,
        current_units=current_units,
        length_units=length_units,
    )

    learner = adaptive.Learner1D(function, bounds=(I_fc_min, I_fc_max))
    learner = adaptive.DataSaver(learner, arg_picker=itemgetter("abs_susc"))

    timeout = adaptive.runner.stop_after(hours=args.max_wall_time)

    def goal(learner_):
        return timeout(learner_) or learner_.loss() < args.max_loss

    if args.mpi:
        executor = MPIPoolExecutor()
    else:
        executor = ProcessPoolExecutor()

    runner = adaptive.Runner(
        learner,
        goal=goal,
        executor=executor,
        ntasks=args.ntasks,
        shutdown_executor=True,
    )
    save_kwargs = dict(fname=os.path.join(base_path, "learner.pickle"))
    runner.start_periodic_saving(save_kwargs=save_kwargs, interval=30)
    runner.ioloop.run_until_complete(runner.task)
    learner.save(save_kwargs["fname"])


if __name__ == "__main__":
    main()
