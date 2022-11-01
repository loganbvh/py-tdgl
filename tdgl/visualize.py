import argparse
import logging

from .enums import Observable
from .visualization.animate import Animate, MultiAnimate

# from .visualization.iv_plot import IvPlot
# from .visualization.ic_dist import IcDist
# from .visualization.ic_vs_b import IcVsB
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
        "-o",
        "--observables",
        type=lambda s: str(s).upper(),
        choices=Observable.get_keys() + ["ALL"],
        nargs="*",
        help="Name(s) of the observable(s) to show.",
    )

    interactive_parser.add_argument(
        "-a",
        "--allow-save",
        action="store_true",
        default=False,
        help="Allow saving file.",
    )

    interactive_parser.set_defaults(func=visualize_tdgl)

    animate_parser = subparsers.add_parser(
        "animate", help="Create an animation of the TDGL data."
    )

    animate_parser.add_argument("--output", type=str, help="Output file for animation.")

    animate_parser.add_argument(
        "-o",
        "--observables",
        type=lambda s: str(s).upper(),
        choices=Observable.get_keys() + ["ALL"],
        nargs="*",
        help="Name(s) of the observable(s) to show.",
    )

    animate_parser.add_argument(
        "-f", "--fps", type=int, default=10, help="Frame rate of the animation."
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

    animate_parser.set_defaults(func=animate_tdgl)

    # iv_parser = subparsers.add_parser("iv", help="Plot an IV curve.")

    # iv_parser.add_argument(
    #     "-o", "--output", type=str, default=None, help="Save the IV curve to file."
    # )

    # iv_parser.add_argument(
    #     "-m",
    #     "--marker-size",
    #     type=int,
    #     default=None,
    #     help="Size of the markers in the curve.",
    # )

    # iv_parser.add_argument(
    #     "-S",
    #     "--save",
    #     type=str,
    #     default=None,
    #     help="Output file for saving the data.",
    # )

    # iv_parser.set_defaults(func=iv)

    # ic_vs_b_parser = subparsers.add_parser(
    #     "ic-vs-b",
    #     help="Plot the critical current as a function of the magnetic field.",
    # )

    # ic_vs_b_parser.add_argument(
    #     "-t",
    #     "--threshold",
    #     type=float,
    #     default=0.01,
    #     help="Voltage level to measure the critical current.",
    # )

    # ic_vs_b_parser.add_argument(
    #     "-o",
    #     "--output",
    #     type=str,
    #     default=None,
    #     help="Save the Ic vs B curve to file.",
    # )

    # ic_vs_b_parser.add_argument(
    #     "-d",
    #     "--data-output",
    #     type=str,
    #     default=None,
    #     help="Save Ic vs B data to file.",
    # )

    # ic_vs_b_parser.set_defaults(func=ic_vs_b)

    # iv_parser.set_defaults(func=iv)

    # ic_dist = subparsers.add_parser(
    #     "ic-dist", help="plot the critical current distribution"
    # )

    # ic_dist.add_argument(
    #     "-t",
    #     "--threshold",
    #     type=float,
    #     default=0.01,
    #     help="voltage level to measure the critical current",
    # )

    # ic_dist.add_argument(
    #     "-o",
    #     "--output",
    #     type=str,
    #     default=None,
    #     help="save the Ic vs B curve to file",
    # )

    # ic_dist.add_argument("-b", "--bins", type=int, default=10, help="number of bins")

    # ic_dist.add_argument(
    #     "-n",
    #     "--normalized",
    #     action="store_true",
    #     default=False,
    #     help="if the histogram should be normalized to a " "probability distribution",
    # )

    # ic_dist.add_argument(
    #     "-d",
    #     "--data-output",
    #     type=str,
    #     default=None,
    #     help="save Ic vs B data to file",
    # )

    # ic_dist.set_defaults(func=ic_dist)

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
    if len(kwargs["observables"]) == 1 and "ALL" not in kwargs["observables"][0]:
        kwargs["observable"] = Observable.from_key(args.observables[0])
        Animate(**kwargs).build()
        return
    if "ALL" not in args.observables:
        kwargs["observables"] = args.observables
    MultiAnimate(**kwargs).build()


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


# def iv(args):
#     IvPlot(
#         input_path=args.input,
#         output_file=args.output,
#         save=args.save,
#         logger=logger,
#     ).show()


# def ic_vs_b(args):
#     IcVsB(
#         input_path=args.input,
#         output_file=args.output,
#         threshold=args.threshold,
#         data_file=args.data_output,
#     ).show()


# def ic_dist(args):
#     IcDist(
#         input_path=args.input,
#         output_file=args.output,
#         threshold=args.threshold,
#         data_file=args.data_output,
#         bins=args.bins,
#         normalized=args.normalized,
#     ).show()


def main(args):
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.disabled = args.silent
    args.func(args)


if __name__ == "__main__":
    main(make_parser().parse_args())
