import argparse
import logging

import matplotlib.pyplot as plt
import superscreen as sc

from . import huber, hypres, ibm


def get_mutual(squid, label, iterations, fc_lambda=None):
    if fc_lambda is not None:
        squid.layers["BE"].london_lambda = fc_lambda
    print(squid)
    fluxoid_polys = sc.make_fluxoid_polygons(squid)
    fig, ax = squid.plot()
    for name, poly in fluxoid_polys.items():
        ax.plot(*sc.geometry.close_curve(poly).T, label=name + "_fluxoid")
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_title(label)
    return squid.mutual_inductance_matrix(iterations=iterations, units="Phi_0 / A")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-points",
        type=int,
        default=5_000,
        help="Minimum number of vertices in the mesh.",
    )
    parser.add_argument(
        "--solve-dtype",
        type=str,
        help="Device solve_dtype.",
        default="float64",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of solver iterations.",
    )
    parser.add_argument(
        "--optimesh-steps",
        type=int,
        default=40,
        help="Number of optimesh steps to perform.",
    )
    parser.add_argument(
        "--fc-lambda",
        type=float,
        default=None,
        help="London penetration depth for the field coil layer.",
    )
    args = parser.parse_args()

    squid_funcs = {
        "ibm-small": ibm.small.make_squid,
        "ibm-medium": ibm.medium.make_squid,
        "ibm-large": ibm.large.make_squid,
        "ibm-xlarge": ibm.xlarge.make_squid,
        "huber": huber.make_squid,
        "hypres-small": hypres.small.make_squid,
    }

    mutuals = {}
    for make_squid in squid_funcs.values():
        squid = make_squid()
        squid.make_mesh(
            min_points=args.min_points,
            optimesh_steps=args.optimesh_steps,
        )
        squid.solve_dtype = args.solve_dtype
        M = get_mutual(
            squid,
            make_squid.__module__,
            args.iterations,
            fc_lambda=args.fc_lambda,
        )
        mutuals[make_squid.__module__] = M
        print(M)

    for label, mutual in mutuals.items():
        print()
        print(label)
        print("-" * len(label))
        print(mutual)
        print(mutual.to("pH"))
        print("-" * len(repr(mutual)))

    plt.show()
