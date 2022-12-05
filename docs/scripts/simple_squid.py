import multiprocessing as mp
from datetime import datetime
from functools import partial
from typing import Sequence, Union

import h5py
import joblib
import numpy as np

import tdgl
from tdgl.geometry import box, circle


def make_squid():
    length_units = "nm"
    xi = 50  # GL coherence length
    london_lambda = 200  # London penetration depth
    d = 20  # Film thickness

    ri = 75  # SQUID annulus inner radis
    ro = 125  # SQUID annulus outer radius
    link_width = 20  # width of the weak links
    link_radius = 10  # radius of curvature of the weak links
    # Length and width of the current leads
    lead_length = 3 * ro
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
        .scale(yfact=0.5)
        .buffer(-link_radius)
        .buffer(link_radius, join_style="round")
    )
    left_notch = right_notch.scale(xfact=-1)
    # SQUID
    film = tdgl.Polygon("film", points=circle(ro)).union(top_lead, bottom_lead)
    film = film.difference(right_notch, left_notch).resample(301)
    hole = tdgl.Polygon("hole", points=circle(ri)).resample(101)

    # Define the current terminals and voltage measurement points
    source = tdgl.Polygon(
        "source", points=box(lead_width, lead_length / 100)
    ).translate(dy=lead_length)
    drain = source.scale(yfact=-1).set_name("drain")
    voltage_points = [(0, +0.5 * lead_length), (0, -0.5 * lead_length)]

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


def terminal_currents(t, *, t0, t1, max_current):
    """Returns the terminal currents for a given time t.

    The current will be zero for t < t0, ramped linearly to
    max_current for t0 <= t < t1, then held constant at max_current
    for t >= t1.

    Args:
        t: Dimensionless time.
        t0: The time at which to begin ramping the current from zero.
        t1: The time at which the current reaches max_current.
    """
    if t < t0:
        # Allow system to equilibrate
        current = 0
    elif t < t1:
        # Linearly ramp current from 0 to 'max_current'
        current = max_current * (t - t0) / (t1 - t0)
    else:
        current = max_current
    return dict(source=current, drain=-current)


def get_voltage(
    max_current: float,
    *,
    field: float,
    device: tdgl.Device,
    options: tdgl.SolverOptions,
    t0: float,
    t1: float,
    ro: float,
    lead_angle: float,
) -> float:
    solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=tdgl.Parameter(
            constant_field_vector_potential,
            Bz=field,
            loop_radius=ro,
            lead_angle=lead_angle,
        ),
        terminal_currents=partial(
            terminal_currents,
            t0=t0,
            t1=t1,
            max_current=max_current,
        ),
    )
    return solution.dynamics.mean_voltage(tmin=options.solve_time / 2)


def simulate_iv_curve(
    currents: Sequence[float],
    min_points: int = 3000,
    output_file: Union[str, None] = None,
    field: float = 0.0,
    lead_angle: float = 0.0,
    solve_time: float = 1000.0,
    t0: float = 100.0,
    t1: float = 150.0,
    ntasks: Union[int, None] = None,
    current_units: str = "uA",
    field_units: str = "mT",
):
    currents = np.asarray(currents)

    options = tdgl.SolverOptions(
        solve_time=solve_time,
        dt_init=1e-7,
        output_file=output_file,
        field_units=field_units,
        current_units=current_units,
        save_every=250,
    )

    device, ro = make_squid()
    device.make_mesh(min_points=min_points, smooth=100)

    ncpus = joblib.cpu_count(only_physical_cores=True)
    if ntasks is None:
        ntasks = max(1, ncpus - 1)
    ntasks = min(ntasks, ncpus)

    func = partial(
        get_voltage,
        field=field,
        device=device,
        options=options,
        t0=t0,
        t1=t1,
        ro=ro,
        lead_angle=lead_angle,
    )
    with mp.Pool(processes=ntasks) as pool:
        results = pool.map(func, currents)

    return device, np.array(results)


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
        "--field", type=float, default=0, help="Applied field in ``field_units``."
    )
    parser.add_argument("--output", type=str, default=None, help="Output file.")
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
        "--ntasks", type=int, default=None, help="Number of processes to use."
    )
    parser.add_argument("--current-units", type=str, default="uA")
    parser.add_argument("--field-units", type=str, default="mT")

    args = parser.parse_args()
    print(vars(args))

    start, stop, num = args.currents
    currents = np.linspace(start, stop, int(num))

    now = datetime.now().strftime("%y%m%d_%H%M%S")
    output_file = f"squid-{now}.h5"

    device, voltage = simulate_iv_curve(
        currents,
        min_points=args.min_points,
        output_file=args.output,
        field=args.field,
        lead_angle=args.lead_angle,
        ntasks=args.ntasks,
        current_units=args.current_units,
        field_units=args.field_units,
    )

    with h5py.File(output_file, "x") as f:
        device.to_hdf5(f.create_group("device"))
        f["current"] = currents
        f["voltage"] = voltage
        for k, v in vars(args).items():
            if v is not None:
                f.attrs[k] = v


if __name__ == "__main__":
    main()
