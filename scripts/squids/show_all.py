import argparse

import matplotlib.pyplot as plt

from . import huber, hypres, ibm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--same-scale",
        default=False,
        action="store_true",
        help="Whether to plot all devices on the same scale.",
    )
    args = parser.parse_args()

    squid_funcs = [
        ibm.small.make_squid,
        ibm.medium.make_squid,
        ibm.large.make_squid,
        ibm.xlarge.make_squid,
        huber.make_squid,
        hypres.small.make_squid,
    ]

    fig, axes = plt.subplots(
        1,
        len(squid_funcs),
        figsize=(len(squid_funcs) * 3, 3),
        sharex=args.same_scale,
        sharey=args.same_scale,
        constrained_layout=True,
    )

    for ax, make_squid in zip(axes, squid_funcs):
        squid = make_squid()
        squid.plot(ax=ax, legend=False)
        ax.set_title(make_squid.__module__)
    plt.show()
