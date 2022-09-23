import os
from datetime import datetime
import json

import h5py
import numpy as np

import tdgl


def make_device(radius, xi, lambda_, d, min_points, optimesh_steps):
    layer = tdgl.Layer(coherence_length=xi, london_lambda=lambda_, thickness=d, z0=0)
    film = tdgl.Polygon("film", points=tdgl.geometry.circle(radius, points=201))
    device = tdgl.Device("box", layer=layer, film=film, length_units="um")
    device.make_mesh(min_points=min_points, optimesh_steps=optimesh_steps)
    return device


def make_pickup_loop(radius, z0):
    pl = tdgl.Polygon(points=tdgl.geometry.circle(radius, points=201))
    pl_points, pl_triangles = pl.make_mesh(min_points=1000, optimesh_steps=50)
    pl_centroids = tdgl.fem.centroids(pl_points, pl_triangles)
    pl_areas = tdgl.fem.triangle_areas(pl_points, pl_triangles)
    pl_centroids = np.append(
        pl_centroids, z0 * np.ones_like(pl_centroids[:, :1]), axis=1
    )
    return pl_centroids, pl_areas


def applied_potential(x, y, z, *, fc_radius, fc_center, fc_current):
    loop_center = np.atleast_2d(fc_center)
    loop_radius = fc_radius
    loop_current = fc_current
    positions = np.array([x, y, z]).T
    potential = tdgl.em.current_loop_vector_potential(
        positions,
        loop_center=loop_center,
        current=loop_current,
        loop_radius=loop_radius,
        length_units="um",
        current_units="mA",
    )
    return potential.to("mT * um").m


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", type=str, help="Output directory.")
    parser.add_argument(
        "--I_fc",
        nargs=3,
        type=float,
        help="Field coil current in mA: start, stop, num_steps.",
    )
    parser.add_argument(
        "--r_fc",
        default=2.5,
        type=float,
        help="Field coil radius in microns.",
    )
    parser.add_argument(
        "--r_pl",
        default=0.6,
        type=float,
        help="Pickup loop radius in microns.",
    )
    parser.add_argument(
        "--z0",
        default=None,
        type=float,
        help="z-position of the field coil and pickup loop.",
    )
    parser.add_argument(
        "--film-radius",
        default=30,
        type=float,
        help="Film radius in microns",
    )
    parser.add_argument(
        "--d", default=0.1, type=float, help="Film thickness in microns."
    )
    parser.add_argument(
        "--lam",
        default=2,
        type=float,
        help="London penetration depth in microns.",
    )
    parser.add_argument(
        "--xi",
        default=1,
        type=float,
        help="Coherence length in microns.",
    )
    parser.add_argument("--gamma", default=10, type=float, help="TDGL gamma parameter.")
    parser.add_argument(
        "--min-points", default=1500, type=int, help="Minimum number of mesh vertices."
    )
    parser.add_argument(
        "--optimesh-steps", default=100, type=int, help="Number of optimesh steps."
    )
    parser.add_argument("--dt", default=1e-2, type=float, help="GL ODE time step.")
    parser.add_argument("--steps", default=5e3, type=float, help="GL ODE steps.")
    parser.add_argument(
        "--save-every", default=10, type=int, help="Save interval in steps."
    )
    parser.add_argument(
        "--pinning", default=0, type=float, help="Pinning sites per square micron."
    )
    parser.add_argument("--screening", action="store_true", help="Include screening.")

    args = parser.parse_args()
    args_as_dict = vars(args)
    print(args_as_dict)

    start_time = datetime.now()

    path = os.path.abspath(args.directory)
    # if os.path.exists(path) and os.listdir(path):
    #     raise ValueError(f"Path {path} exists and is not empty.")
    os.makedirs(path, exist_ok=True)

    device = make_device(
        args.film_radius,
        args.xi,
        args.lam,
        args.d,
        args.min_points,
        args.optimesh_steps,
    )

    z0 = args.z0
    if z0 is None:
        z0 = args.r_fc

    pl_points, pl_areas = make_pickup_loop(args.r_pl, z0)

    fc_center = (0, 0, z0)
    r_fc = args.r_fc
    start, stop, num = args.I_fc
    I_fc = np.linspace(start, stop, int(num))

    all_flux = []

    for i, current in enumerate(I_fc):

        A_applied = tdgl.Parameter(
            applied_potential,
            fc_center=fc_center,
            fc_radius=r_fc,
            fc_current=current,
        )

        solution = tdgl.solve(
            device,
            A_applied,
            os.path.join(path, f"output-{i}.h5"),
            pinning_sites=args.pinning,
            field_units="mT",
            gamma=args.gamma,
            dt=args.dt,
            skip=0,
            max_steps=int(args.steps),
            save_every=args.save_every,
            source_drain_current=0,
            include_screening=args.screening,
            rng_seed=42,
        )
        solution.to_hdf5()

        field = solution.field_at_position(
            pl_points,
            units="Phi_0 / um**2",
            with_units=False,
            return_sum=False,
        )
        flux = (field * pl_areas).sum()
        all_flux.append(flux)

        with h5py.File(solution.path, "r+") as f:
            for key, val in args_as_dict.items():
                f.attrs[key] = val
            f.attrs["flux_in_Phi_0"] = flux

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
