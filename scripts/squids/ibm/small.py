"""IBM SQUID susceptometer, 100 nm inner radius pickup loop."""

import numpy as np
import superscreen as sc

from .layers import ibm_squid_layers


def make_squid(interp_points=81, align_layers="middle"):
    pl_length = 2.5
    ri_pl = 0.1
    ro_pl = 0.3
    ri_fc = 0.5
    ro_fc = 1.0125
    pl_center = sc.Polygon(
        "pl_center",
        layer="W1",
        points=sc.geometry.box(0.20, pl_length, center=(0, -pl_length / 2 + ri_pl)),
    )

    pl = sc.Polygon(
        "pl",
        layer="W1",
        points=sc.geometry.box(
            2 * ro_pl, pl_length + ro_pl, center=(0, -(pl_length + 0.3) / 2 + 3 * ri_pl)
        ),
    ).union(
        np.array(
            [
                [-0.30, -1.10],
                [-0.385, -1.7],
                [-0.64, -2.57],
                [+0.62, -2.57],
                [+0.35, -1.67],
                [+0.30, -1.15],
            ]
        )
    )

    pl_shield1 = sc.Polygon(
        "pl_shield1",
        layer="W2",
        points=np.array(
            [
                [+0.35, -ri_pl],
                [-0.35, -ri_pl],
                [-0.98, -2.65],
                [-1.05, -2.80],
                [+1.05, -2.80],
                [+0.98, -2.65],
            ]
        ),
    )

    pl_shield2 = sc.Polygon(
        "pl_shield2",
        layer="BE",
        points=np.array(
            [
                [+0.5, -1.5 - ri_pl],
                [-0.5, -1.5 - ri_pl],
                [-0.84, -2.70],
                [+0.84, -2.70],
            ]
        ),
    )

    fc = sc.Polygon(
        "fc",
        layer="BE",
        points=sc.geometry.circle(ro_fc, center=(0, 0.01)),
    ).union(
        np.array(
            [
                [2.30, -0.35],
                [2.00, -0.04],
                [1.19, 0.54],
                [0.60, 0.80],
                [0.40, -0.9],
                [1.1, -1.30],
                [1.35, -1.9],
            ]
        )
    )

    fc_shield = sc.Polygon(
        "fc_shield",
        layer="W1",
        points=np.array(
            [
                [2.5, -0.45],
                [2.15, -0.15],
                [2.00, -0.04],
                [1.31, 0.43],
                [0.81, -0.08],
                [0.66, -1.23],
                [1.25, -2.65],
            ]
        ),
    )

    fc_center = sc.Polygon(
        "fc_center", layer="BE", points=sc.geometry.circle(ri_fc)
    ).union(
        np.array(
            [
                [1.7, -0.47],
                [0.95, 0.02],
                [0.6, 0.11],
                [0.4, 0.28],
                [0.33, -0.34],
                [0.69, -0.44],
                [1.4, -0.9],
            ]
        )
    )

    bbox = sc.Polygon(
        "bounding_box",
        layer="BE",
        points=sc.geometry.box(3.9, 4.25, points_per_side=5, center=(0.65, -0.9)),
    )

    films = [fc_shield, fc, pl_shield1, pl_shield2, pl]
    holes = [fc_center, pl_center]
    for polygon in films + holes:
        if "shield" in polygon.name:
            polygon.points = polygon.resample(interp_points // 2)
        else:
            polygon.points = polygon.resample(interp_points)

    return sc.Device(
        "ibm_100nm",
        layers=ibm_squid_layers(align=align_layers),
        films=films,
        holes=holes,
        abstract_regions=[bbox],
        length_units="um",
    )
