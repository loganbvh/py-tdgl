import argparse
import logging
import os

from .visualization import (
    DEFAULT_QUANTITIES,
    InteractivePlot,
    MultiInteractivePlot,
    Quantity,
    convert_to_xdmf,
    create_animation,
    generate_snapshots,
    monitor_solution,
)

logger = logging.getLogger("visualize")


def make_parser() -> argparse.ArgumentParser:
    quantities_args = ("-q", "--quantities")
    quantities_kwargs = dict(
        type=lambda s: str(s).upper(),
        choices=Quantity.get_keys() + ["ALL"],
        nargs="*",
        help="Name(s) of the quantities to show.",
    )

    parser = argparse.ArgumentParser(description="Visualize TDGL simulation data.")

    # Common arguments
    parser.add_argument("-i", "--input", type=str, help="H5 file to visualize.")
    parser.add_argument("-o", "--output", type=str, help="Output file path.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Run in verbose mode.",
    )
    parser.add_argument(
        "--shading",
        type=str,
        choices=["flat", "gouraud"],
        default="gouraud",
        help="Shading method, see matplotlib.pyplot.tripcolor.",
    )
    parser.add_argument(
        "--dimensionless", action="store_true", help="Use dimensionless x-y units."
    )
    parser.add_argument(
        "--xlim", type=float, nargs=2, default=None, help="x-axis limits"
    )
    parser.add_argument(
        "--ylim", type=float, nargs=2, default=None, help="y-axis limits"
    )
    parser.add_argument(
        "--axis-labels", action="store_true", help="Add x-y axis labels."
    )
    parser.add_argument(
        "--autoscale",
        action="store_true",
        help="Autoscale colorbar limits at each frame.",
    )
    parser.add_argument(
        "--axes-off",
        action="store_true",
        help="Turn the axes off.",
    )
    parser.add_argument(
        "--title-off",
        action="store_true",
        help="Turn figure title off.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        help="Figure size (width, height) in inches.",
    )
    parser.add_argument(
        "-d", "--dpi", type=float, default=200, help="Resolution in dots per inch."
    )

    subparsers = parser.add_subparsers()

    # Interactive plot
    interactive_parser = subparsers.add_parser(
        "interactive", help="Create an interactive plot of one or more quantities."
    )
    interactive_parser.add_argument(*quantities_args, **quantities_kwargs)
    interactive_parser.set_defaults(func=visualize_tdgl)

    # Matplotlib animate
    animate_parser = subparsers.add_parser(
        "animate", help="Create an animation of the TDGL data."
    )
    animate_parser.add_argument(
        "-f", "--fps", type=int, default=30, help="Frame rate of the animation."
    )
    animate_parser.add_argument(
        "--min-frame",
        type=int,
        default=0,
        help="The first frame to render.",
    )
    animate_parser.add_argument(
        "--max-frame",
        type=int,
        default=-1,
        help="The last frame to render (-1 indicates the last step in the simulation).",
    )
    animate_parser.add_argument(*quantities_args, **quantities_kwargs)
    animate_parser.set_defaults(func=animate_tdgl)

    # Live monitoring during solve
    monitor_parser = subparsers.add_parser(
        "monitor", help="Visualize the results of a simulation as it is running."
    )
    monitor_parser.add_argument(
        "--interval", type=float, default=1, help="Monitor update interval in seconds."
    )
    monitor_parser.add_argument(*quantities_args, **quantities_kwargs)
    monitor_parser.set_defaults(func=monitor_tdgl)

    # Convert data format
    convert_parser = subparsers.add_parser(
        "convert", help="Convert a Solution from HDF5 to another data format."
    )
    convert_parser.add_argument(
        "--format",
        type=str,
        choices=["xdmf"],
        help="Data format into which to convert the Solution.",
    )
    convert_parser.set_defaults(func=convert_tdgl)

    # Generate snapshots
    snap_parser = subparsers.add_parser(
        "snapshot", help="Generate snapshots of a Solution."
    )
    snap_parser.add_argument(
        "-t",
        "--times",
        type=float,
        nargs="+",
        help="The time(s) at which to generate a snapshot.",
    )
    snap_parser.add_argument(*quantities_args, **quantities_kwargs)
    snap_parser.set_defaults(func=snapshot_tdgl)

    return parser


def animate_tdgl(args: argparse.Namespace) -> None:
    kwargs = dict(
        input_file=args.input,
        output_file=args.output,
        logger=logger,
        shading=args.shading,
        dpi=args.dpi,
        fps=args.fps,
        min_frame=args.min_frame,
        max_frame=args.max_frame,
        autoscale=args.autoscale,
        dimensionless=args.dimensionless,
        xlim=args.xlim,
        ylim=args.ylim,
        axis_labels=args.axis_labels,
        axes_off=args.axes_off,
        title_off=args.title_off,
    )
    if args.figsize is not None:
        kwargs["figure_kwargs"] = dict(figsize=args.figsize)
    if args.quantities is None or "ALL" in args.quantities:
        kwargs["quantities"] = DEFAULT_QUANTITIES
    else:
        kwargs["quantities"] = args.quantities
    create_animation(**kwargs)


def visualize_tdgl(args: argparse.Namespace) -> None:
    kwargs = dict(
        input_file=args.input,
        shading=args.shading,
        dimensionless=args.dimensionless,
        xlim=args.xlim,
        ylim=args.ylim,
        axis_labels=args.axis_labels,
        logger=logger,
    )
    if args.figsize is not None:
        kwargs["figure_kwargs"] = dict(figsize=args.figsize)
    if args.quantities is None:
        InteractivePlot(**kwargs).show()
        return
    if "ALL" not in args.quantities:
        kwargs["quantities"] = args.quantities
    MultiInteractivePlot(**kwargs).show()


def monitor_tdgl(args: argparse.Namespace) -> None:
    dirname = os.path.dirname(args.input)
    fname = os.path.basename(args.input) + ".tmp"
    h5path = os.path.join(dirname, fname)
    kwargs = dict(
        h5path=h5path,
        update_interval=args.interval,
        shading=args.shading,
        xlim=args.xlim,
        ylim=args.ylim,
        autoscale=args.autoscale,
        dimensionless=args.dimensionless,
    )
    if args.figsize is not None:
        kwargs["figure_kwargs"] = dict(figsize=args.figsize)
    if args.quantities is None or "ALL" in args.quantities:
        kwargs["quantities"] = DEFAULT_QUANTITIES
    else:
        kwargs["quantities"] = args.quantities
    monitor_solution(**kwargs)


def convert_tdgl(args: argparse.Namespace) -> None:
    # I may add other convert_funcs if they are requested.
    convert_funcs = {"xdmf": convert_to_xdmf}
    convert_func = convert_funcs[args.format.lower()]
    convert_func(
        path_to_solution=args.input,
        xdmf_path=args.output,
        dimensionless=args.dimensionless,
    )


def snapshot_tdgl(args: argparse.Namespace) -> None:
    if args.output is not None:
        logger.warning("Argument --output is ignored by the snapshot command.")
    dirname = os.path.dirname(args.input)
    basename = os.path.basename(args.input)
    name, _ = os.path.splitext(basename)
    output_file = os.path.join(dirname, f"{name}_[t={{time:.0f}}].png")

    kwargs = dict(
        input_path=args.input,
        times=args.times,
        shading=args.shading,
        autoscale=args.autoscale,
        dimensionless=args.dimensionless,
        xlim=args.xlim,
        ylim=args.ylim,
        axis_labels=args.axis_labels,
        axes_off=args.axes_off,
        title_off=args.title_off,
    )
    if args.figsize is not None:
        kwargs["figure_kwargs"] = dict(figsize=args.figsize)
    if args.quantities is None or "ALL" in args.quantities:
        kwargs["quantities"] = DEFAULT_QUANTITIES
    else:
        kwargs["quantities"] = args.quantities

    figures = generate_snapshots(**kwargs)
    for time, (fig, axes) in zip(args.times, figures):
        fig.savefig(output_file.format(time=time), dpi=args.dpi, bbox_inches="tight")


def main(args: argparse.Namespace) -> None:
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main(make_parser().parse_args())
