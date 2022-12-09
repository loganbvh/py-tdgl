import argparse
import logging

from .visualization import (
    DEFAULT_QUANTITIES,
    InteractivePlot,
    MultiInteractivePlot,
    Quantity,
    create_animation,
)

logger = logging.getLogger("visualize")


def make_parser():
    parser = argparse.ArgumentParser(description="Visualize TDGL simulation data.")
    subparsers = parser.add_subparsers()
    parser.add_argument("--input", type=str, help="H5 file to visualize.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Run in verbose mode.",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        default=False,
        help="Run in silent mode.",
    )

    interactive_parser = subparsers.add_parser(
        "interactive", help="Create an interactive plot of one or more quantities."
    )
    interactive_parser.add_argument(
        "-q",
        "--quantities",
        type=lambda s: str(s).upper(),
        choices=Quantity.get_keys() + ["ALL"],
        nargs="*",
        help=(
            "Name(s) of the quantities to show. Because ``quantities`` takes a "
            "variable number of arguments, it must be the last argument provided."
        ),
    )

    interactive_parser.set_defaults(func=visualize_tdgl)

    animate_parser = subparsers.add_parser(
        "animate", help="Create an animation of the TDGL data."
    )
    animate_parser.add_argument(
        "-o", "--output", type=str, help="Output file for animation."
    )
    animate_parser.add_argument(
        "-f", "--fps", type=int, default=30, help="Frame rate of the animation."
    )
    animate_parser.add_argument(
        "-d", "--dpi", type=float, default=200, help="Resolution in dots per inch."
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
    animate_parser.add_argument(
        "--axes-off",
        action="store_true",
        help="Turn the axes off.",
    )
    animate_parser.add_argument(
        "--title-off",
        action="store_true",
        help="Turn figure title off.",
    )
    animate_parser.add_argument(
        "-q",
        "--quantities",
        type=lambda s: str(s).upper(),
        choices=Quantity.get_keys() + ["ALL"],
        nargs="*",
        help=(
            "Name(s) of the quantities to show. Because 11quantities11 takes a "
            "variable number of arguments, it must be the last argument provided."
        ),
    )

    animate_parser.set_defaults(func=animate_tdgl)

    return parser


def animate_tdgl(args):
    kwargs = dict(
        input_file=args.input,
        output_file=args.output,
        logger=logger,
        silent=args.silent,
        dpi=args.dpi,
        fps=args.fps,
        min_frame=args.min_frame,
        max_frame=args.max_frame,
        axes_off=args.axes_off,
        title_off=args.title_off,
    )
    if args.quantities is None or "ALL" in args.quantities:
        kwargs["quantities"] = DEFAULT_QUANTITIES
    else:
        kwargs["quantities"] = args.quantities
    create_animation(**kwargs)


def visualize_tdgl(args):
    if args.quantities is None:
        InteractivePlot(input_file=args.input, logger=logger).show()
        return
    kwargs = dict(input_file=args.input, logger=logger)
    if "ALL" not in args.quantities:
        kwargs["quantities"] = args.quantities
    MultiInteractivePlot(**kwargs).show()


def main(args):
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.disabled = args.silent
    args.func(args)


if __name__ == "__main__":
    main(make_parser().parse_args())
