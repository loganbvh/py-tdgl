#!/usr/bin/env python
import argparse
import logging

from .enums import Observable
from .visualization.animate import Animate, MultiAnimate
from .visualization.ic_dist import IcDist
from .visualization.ic_vs_b import IcVsB
from .visualization.interactive_plot import InteractivePlot, MultiInteractivePlot
from .visualization.iv_plot import IvPlot


class Visualize:
    def __init__(self):
        # Set the current frame and field
        self.frame = 0
        self.observable = Observable.COMPLEX_FIELD

        # Parse command line args
        parser = argparse.ArgumentParser(description="visualize TDGL simulation data")

        parser.add_argument(
            "input", metavar="INPUT", type=str, help="data file to visualize"
        )

        parser.add_argument(
            "--observables",
            type=str,
            nargs="*",
        )

        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            default=False,
            help="run in verbose mode",
        )

        parser.add_argument(
            "-s",
            "--silent",
            action="store_true",
            default=False,
            help="run in silent mode",
        )

        parser.add_argument(
            "-a",
            "--allow-save",
            action="store_true",
            default=False,
            help="allow saving file",
        )

        parser.set_defaults(func=self.visualize_tdgl)
        subparsers = parser.add_subparsers()

        animate_parser = subparsers.add_parser(
            "animate", help="create an animation of the TDGL data"
        )

        animate_parser.add_argument(
            "output", metavar="OUTPUT", type=str, help="output file for animation"
        )

        animate_parser.add_argument(
            "-o",
            "--observable",
            type=lambda s: str(s).upper(),
            choices=Observable.get_keys(),
            default="COMPLEX_FIELD",
            help="specify the observable to display in the animation",
        )

        animate_parser.add_argument(
            "-f", "--fps", type=int, default=10, help="frame rate of the animation"
        )

        animate_parser.add_argument(
            "-d", "--dpi", type=float, default=200, help="resolution in dots per inch"
        )

        animate_parser.add_argument(
            "-g",
            "--gpu",
            action="store_true",
            default=False,
            help="enable NVIDIA GPU acceleration",
        )

        animate_parser.set_defaults(func=self.animate_tdgl)

        iv_parser = subparsers.add_parser("iv", help="plot an IV curve")

        iv_parser.add_argument(
            "-o", "--output", type=str, default=None, help="save the IV curve to file"
        )

        iv_parser.add_argument(
            "-m",
            "--marker-size",
            type=int,
            default=None,
            help="size 1of the markers in the curve",
        )

        iv_parser.add_argument(
            "-S",
            "--save",
            type=str,
            default=None,
            help="specify outfile for saving the data",
        )

        iv_parser.set_defaults(func=self.iv)

        ic_vs_b_parser = subparsers.add_parser(
            "ic-vs-b",
            help="plot the critical current as a function of the magnetic field",
        )

        ic_vs_b_parser.add_argument(
            "-t",
            "--threshold",
            type=float,
            default=0.01,
            help="voltage level to measure the critical current",
        )

        ic_vs_b_parser.add_argument(
            "-o",
            "--output",
            type=str,
            default=None,
            help="save the Ic vs B curve to file",
        )

        ic_vs_b_parser.add_argument(
            "-d",
            "--data-output",
            type=str,
            default=None,
            help="save Ic vs B data to file",
        )

        ic_vs_b_parser.set_defaults(func=self.ic_vs_b)

        iv_parser.set_defaults(func=self.iv)

        ic_dist = subparsers.add_parser(
            "ic-dist", help="plot the critical current distribution"
        )

        ic_dist.add_argument(
            "-t",
            "--threshold",
            type=float,
            default=0.01,
            help="voltage level to measure the critical current",
        )

        ic_dist.add_argument(
            "-o",
            "--output",
            type=str,
            default=None,
            help="save the Ic vs B curve to file",
        )

        ic_dist.add_argument(
            "-b", "--bins", type=int, default=10, help="number of bins"
        )

        ic_dist.add_argument(
            "-n",
            "--normalized",
            action="store_true",
            default=False,
            help="if the histogram should be normalized to a "
            "probability distribution",
        )

        ic_dist.add_argument(
            "-d",
            "--data-output",
            type=str,
            default=None,
            help="save Ic vs B data to file",
        )

        ic_dist.set_defaults(func=self.ic_dist)

        # Get arguments
        self.args = parser.parse_args()

        # Create a logger
        self.logger = logging.getLogger("visualize")
        console_stream = logging.StreamHandler()
        console_stream.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        self.logger.addHandler(console_stream)

        # Set log level to DEBUG in verbose mode and INFO in non-verbose mode
        self.logger.setLevel(logging.DEBUG if self.args.verbose else logging.INFO)

        # Disable logging if silent mode is enabled
        self.logger.disabled = self.args.silent

        self.args.func()

    def visualize_tdgl(self):
        if len(self.args.observables) == 0:
            InteractivePlot(
                input_file=self.args.input,
                enable_save=self.args.allow_save,
                logger=self.logger,
            ).show()
            return
        kwargs = dict(
            input_file=self.args.input,
            logger=self.logger,
        )
        if "all" not in self.args.observables:
            kwargs["observables"] = self.args.observables
        MultiInteractivePlot(**kwargs).show()

    def animate_tdgl(self):
        kwargs = dict(
            input_file=self.args.input,
            output_file=self.args.output,
            logger=self.logger,
            silent=self.args.silent,
            dpi=self.args.dpi,
            fps=self.args.fps,
            gpu=self.args.gpu,
        )
        if len(self.args.observables) == 0:
            kwargs["observable"] = Observable.from_key(self.args.observable)
            Animate(**kwargs).build()
            return
        if "all" not in self.args.observables:
            kwargs["observables"] = self.args.observables
        MultiAnimate(**kwargs).build()

    def iv(self):
        IvPlot(
            input_path=self.args.input,
            output_file=self.args.output,
            save=self.args.save,
            logger=self.logger,
        ).show()

    def ic_vs_b(self):
        IcVsB(
            input_path=self.args.input,
            output_file=self.args.output,
            threshold=self.args.threshold,
            data_file=self.args.data_output,
        ).show()

    def ic_dist(self):
        IcDist(
            input_path=self.args.input,
            output_file=self.args.output,
            threshold=self.args.threshold,
            data_file=self.args.data_output,
            bins=self.args.bins,
            normalized=self.args.normalized,
        ).show()


if __name__ == "__main__":
    Visualize()
