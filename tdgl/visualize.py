import argparse
import logging

from .visualization.animate import multi_animate
from .visualization.defaults import Observable
from .visualization.interactive_plot import InteractivePlot, MultiInteractivePlot

logger = logging.getLogger("visualize")
console_stream = logging.StreamHandler()
console_stream.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(console_stream)


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
        "interactive", help="Create an interactive plot of one or more observables."
    )
    interactive_parser.add_argument(
        "-a",
        "--allow-save",
        action="store_true",
        default=False,
        help="Allow saving file.",
    )
    interactive_parser.add_argument(
        "-o",
        "--observables",
        type=lambda s: str(s).upper(),
        choices=Observable.get_keys() + ["ALL"],
        nargs="*",
        help="Name(s) of the observable(s) to show.",
    )

    interactive_parser.set_defaults(func=visualize_tdgl)

    animate_parser = subparsers.add_parser(
        "animate", help="Create an animation of the TDGL data."
    )
    animate_parser.add_argument("--output", type=str, help="Output file for animation.")
    animate_parser.add_argument(
        "-f", "--fps", type=int, default=30, help="Frame rate of the animation."
    )
    animate_parser.add_argument(
        "-d", "--dpi", type=float, default=200, help="Resolution in dots per inch."
    )
    animate_parser.add_argument(
        "-s",
        "--skip",
        type=int,
        default=0,
        help="Number of frames to skip at the beginning of the animation.",
    )
    animate_parser.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        default=False,
        help="Enable NVIDIA GPU acceleration.",
    )
    animate_parser.add_argument(
        "-o",
        "--observables",
        type=lambda s: str(s).upper(),
        choices=Observable.get_keys() + ["ALL"],
        nargs="*",
        help="Name(s) of the observable(s) to show.",
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
        gpu=args.gpu,
        skip=args.skip,
    )
    if args.observables is not None and "ALL" not in args.observables:
        kwargs["observables"] = args.observables
    multi_animate(**kwargs)


def visualize_tdgl(args):
    if args.observables is None:
        InteractivePlot(
            input_file=args.input,
            enable_save=args.allow_save,
            logger=logger,
        ).show()
        return
    kwargs = dict(
        input_file=args.input,
        logger=logger,
    )
    if "ALL" not in args.observables:
        kwargs["observables"] = args.observables
    MultiInteractivePlot(**kwargs).show()


def main(args):
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.disabled = args.silent
    args.func(args)


if __name__ == "__main__":
    main(make_parser().parse_args())
