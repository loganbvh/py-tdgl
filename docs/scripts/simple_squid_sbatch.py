import os
from datetime import datetime
from typing import Sequence

import h5py
import numpy as np

import tdgl
from tdgl.geometry import box, circle


def make_squid():
    length_units = "nm"
    xi = 50  # GL coherence length
    london_lambda = 200  # London penetration depth
    d = 20  # Film thickness

    ri = 75  # SQUID annulus inner radis
    ro = 150  # SQUID annulus outer radius
    link_width = 30  # width of the weak links
    link_angle = 60  # included angle of the triangular constriction
    link_radius = 10  # radius of curvature of the weak links
    # Length and width of the current leads
    lead_length = 2 * ro
    lead_width = ro

    # Define the material properties of the superconducting layer
    layer = tdgl.Layer(coherence_length=xi, london_lambda=london_lambda, thickness=d)

    # Define the device geometry:
    # Leads
    top_lead = tdgl.Polygon(points=box(lead_width, lead_length)).translate(
        dy=lead_length / 2
    )
    bottom_lead = top_lead.scale(yfact=-1)
    # Weak links
    right_notch = (
        tdgl.Polygon(points=box(ro))
        .rotate(45)
        .translate(dx=(ri + ro * np.sqrt(2) / 2 + link_width - link_radius - 2))
        .scale(yfact=np.tan(np.radians(link_angle / 2)))
        .buffer(-link_radius)
        .buffer(link_radius, join_style="round")
    )
    left_notch = right_notch.scale(xfact=-1)
    # SQUID
    film = tdgl.Polygon("film", points=circle(ro)).union(top_lead, bottom_lead)
    film = film.difference(right_notch, left_notch).resample(501)
    hole = tdgl.Polygon("hole", points=circle(ri)).resample(101)

    # Define the current terminals and voltage measurement points
    source = tdgl.Polygon(
        "source", points=box(lead_width, lead_length / 100)
    ).translate(dy=lead_length)
    drain = source.scale(yfact=-1).set_name("drain")
    voltage_points = [(0, +0.6 * lead_length), (0, -0.6 * lead_length)]

    # Build the Device
    device = tdgl.Device(
        "squid",
        layer=layer,
        film=film,
        holes=[hole],
        terminals=[source, drain],
        voltage_points=voltage_points,
        length_units=length_units,
    )
    return device, ro


def constant_field_vector_potential(
    x,
    y,
    z,
    *,
    Bz: float,
    loop_radius: float,
    lead_angle: float,
) -> np.ndarray:
    """Calculates the vector potential [Ax, Ay] for uniform out-of-plane
    magnetic field Bz. For all points further than loop_radius from the
    origin, the magnetic field will be scaled by cos(lead_angle).

    Args:
        x, y, z: Spatial coordinates at which to evaluate A.
        Bz: The constant out-of-plane magnetic field.
        loop_radius: The outer radius of the SQUID loop.
        lead_angle: The angle of the leads, in degrees, relative
            to the x-y plane.

    Returns:
        The applied vector potential.
    """
    # Evaluate the vector potential
    A = Bz / 2 * np.array([-y, x, np.zeros_like(x)]).T
    # For all points outside the SQUID loop, i.e., points in the leads,
    # multiply the vector potential by cos(lead_angle).
    r = np.sqrt(x**2 + y**2)
    A[r > loop_radius] *= np.cos(np.radians(lead_angle))
    return A


def simulate_iv_curve(
    currents: Sequence[float],
    field: float,
    h5file: h5py.File,
    min_points: int = 3000,
    lead_angle: float = 0.0,
    solve_time: float = 500.0,
    current_units: str = "uA",
    field_units: str = "mT",
):

    options = tdgl.SolverOptions(
        solve_time=solve_time,
        dt_init=1e-7,
        output_file=None,
        field_units=field_units,
        current_units=current_units,
        save_every=500,
    )

    device, ro = make_squid()
    device.make_mesh(
        min_points=min_points,
        max_edge_length=device.coherence_length / 3,
        smooth=100,
    )

    zero_current_solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=tdgl.Parameter(
            constant_field_vector_potential,
            Bz=field,
            loop_radius=ro,
            lead_angle=lead_angle,
        ),
    )

    zero_current_solution.to_hdf5(h5file.create_group("zero_current_solution"))
    # dynamics_grp = h5file.create_group("dynamics", track_order=True)

    voltages = []

    for i, current in enumerate(currents):
        solution = tdgl.solve(
            device,
            options,
            applied_vector_potential=tdgl.Parameter(
                constant_field_vector_potential,
                Bz=field,
                loop_radius=ro,
                lead_angle=lead_angle,
            ),
            terminal_currents=dict(source=current, drain=-current),
            seed_solution=zero_current_solution,
        )
        mean_voltage = solution.dynamics.mean_voltage(tmin=solve_time / 2)
        voltages.append(mean_voltage)
        # solution.dynamics.to_hdf5(dynamics_grp, str(i))
        # dynamics_grp[str(i)].attrs["current"] = current
        # dynamics_grp[str(i)].attrs["voltage"] = mean_voltage

    h5file["voltages"] = np.array(voltages)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--currents",
        type=float,
        nargs=3,
        help="(start, stop, num_points) for currents, in ``current_units``.",
    )
    parser.add_argument(
        "--fields",
        type=float,
        nargs=3,
        help="(start, stop, num_points) for the applied field in ``field_units``.",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=str, default=None, help="Output directory.")
    parser.add_argument(
        "--min-points", type=int, default=3000, help="Minimum number of mesh points."
    )
    parser.add_argument(
        "--lead-angle",
        type=float,
        default=60,
        help="Angle of the leads, in degrees, relative to the x-y plane.",
    )
    parser.add_argument(
        "--solve-time", type=float, default=500, help="Total solve time."
    )
    parser.add_argument("--current-units", type=str, default="uA")
    parser.add_argument("--field-units", type=str, default="mT")

    args = parser.parse_args()
    print(vars(args))

    start, stop, num = args.currents
    currents = np.linspace(start, stop, int(num))

    start, stop, num = args.fields
    fields = np.linspace(start, stop, int(num))
    index = int(args.index)

    os.makedirs(args.output, exist_ok=True)
    fname = os.path.join(args.output, f"results-{args.index:03d}.h5")

    start_time = datetime.now()

    with h5py.File(fname, "x") as h5file:

        for k, v in vars(args).items():
            if v is not None:
                h5file.attrs[k] = v
        h5file.attrs["start_time"] = start_time.isoformat()

        h5file["currents"] = currents
        h5file["fields"] = fields

        simulate_iv_curve(
            currents,
            fields[index],
            h5file,
            solve_time=args.solve_time,
            min_points=args.min_points,
            lead_angle=args.lead_angle,
            current_units=args.current_units,
            field_units=args.field_units,
        )

        end_time = datetime.now()
        h5file.attrs["end_time"] = end_time.isoformat()
        h5file.attrs["total_seconds"] = (end_time - start_time).total_seconds()


if __name__ == "__main__":
    main()
