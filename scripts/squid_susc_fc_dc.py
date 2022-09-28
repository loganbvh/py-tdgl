import json
import logging
import os
import sys
from datetime import datetime

import h5py
import numpy as np
import superscreen as sc

import squids
import tdgl

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


def main():

    import argparse

    parser = argparse.ArgumentParser()

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
        nargs=3,
        type=float,
        help="RMS field coil current in mA: start, stop, num_steps.",
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

    start_time = datetime.now()

    path = os.path.abspath(args.directory)
    # if os.path.exists(path) and os.listdir(path):
    #     raise ValueError(f"Path {path!r} exists and is not empty.")
    os.makedirs(path, exist_ok=True)

    logger.info("Building SQUID and calculating field coil vector potential.")
    squid = make_squid(
        args.squid_type, args.squid_points, args.squid_optimesh, args.squid_angle
    )
    squid_solution = get_base_squid_solution(squid, args.squid_iterations)
    squid_solution_path = os.path.join(path, "squid_solution")
    squid_solution.to_file(squid_solution_path)

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

    start, stop, num = args.I_fc
    I_fc = np.linspace(start, stop, int(num))

    all_flux = []

    options = tdgl.SolverOptions(
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        total_time=args.total_time,
        min_steps=args.steps,
        save_every=args.save_every,
    )

    for i, current in enumerate(I_fc):

        A_applied = tdgl.Parameter(
            applied_potential,
            path_to_solution=squid_solution_path,
            r0=args.squid_position,
            I_fc=current,
            field_units=field_units,
            current_units=current_units,
            length_units=length_units,
        )

        tdgl_solution = tdgl.solve(
            device,
            A_applied,
            os.path.join(path, f"output-{i}.h5"),
            options,
            pinning_sites=args.pinning,
            field_units=field_units,
            gamma=args.gamma,
            source_drain_current=0,
            include_screening=args.screening,
            rng_seed=42,
        )
        tdgl_solution.to_hdf5()

        flux = calculate_pl_flux(squid, tdgl_solution, iterations=args.squid_iterations)
        all_flux.append(flux)

        with h5py.File(tdgl_solution.path, "r+") as f:
            for key, val in args_as_dict.items():
                if val is not None:
                    f.attrs[key] = val
            f.attrs["pl_fluxoid_in_Phi_0"] = flux

    end_time = datetime.now()

    json_data = {}
    json_data["args"] = args_as_dict.copy()
    json_data["timing"] = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_seconds": (end_time - start_time).total_seconds(),
    }
    json_data["I_fc"] = I_fc.tolist()
    json_data["flux"] = all_flux
    json_data["susc"] = (1e3 * np.array(all_flux) / I_fc).tolist()

    with open(os.path.join(path, "results.json"), "w") as f:
        json.dump(json_data, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
