"""IBM SQUID susceptometer, 300 nm inner radius pickup loop."""

import numpy as np
import superscreen as sc

from .layers import ibm_squid_layers


def make_squid(interp_points=201, align_layers="middle"):
    pl_length = 2.2
    ri_pl = 0.3
    ro_pl = 0.5
    ri_fc = 1.0
    ro_fc = 1.5
    pl_center = sc.Polygon(
        "pl_center",
        layer="W1",
        points=sc.geometry.circle(ri_pl),
    ).union(sc.geometry.box(0.2, pl_length, center=(0, -pl_length / 2 - 0.9 * ri_pl)))

    pl = sc.Polygon("pl", layer="W1", points=sc.geometry.circle(ro_pl)).union(
        np.array(
            [
                [+0.3, -0.4],
                [-0.3, -0.4],
                [-0.87, -2.8],
                [+0.85, -2.8],
            ]
        )
    )

    pl_shield2 = sc.Polygon(
        "pl_shield2",
        layer="BE",
        points=np.array(
            [
                [+0.75, -(2.3 - ri_pl)],
                [-0.75, -(2.3 - ri_pl)],
                [-0.99, -3.0],
                [+0.96, -3.0],
            ]
        ),
    )

    pl_shield1 = sc.Polygon(
        "pl_shield1",
        layer="W2",
        points=np.array(
            [
                [+0.3, -0.4],
                [-0.3, -0.4],
                [-1.0, -2.7],
                [-1.2, -3.2],
                [+1.2, -3.2],
                [+1.0, -2.7],
            ]
        ),
    )

    fc_center = sc.Polygon(
        "fc_center", layer="BE", points=sc.geometry.circle(ri_fc)
    ).union(
        np.array(
            [
                [2.2, -1.2],
                [1.7, -0.45],
                [0.97, 0.0],
                [0.8, -0.5],
                [1.23, -0.78],
                [1.4, -0.9],
                [1.85, -1.55],
            ]
        )
    )

    fc = sc.Polygon("fc", layer="BE", points=sc.geometry.circle(ro_fc)).union(
        np.array(
            [
                [3.0, -1.05],
                [2.0, 0.0],
                [1.68, 0.2],
                [1.2, 0.52],
                [0.85, -1.18],
                [1.12, -1.35],
                [1.55, -2.35],
            ]
        )
    )

    fc_shield = sc.Polygon(
        "fc_shield",
        layer="W1",
        points=np.array(
            [
                [3.25, -1.25],
                [2.96, -0.9],
                [2.0, 0.0],
                [1.67, 0.19],
                [1.11, -0.37],
                [0.9, -1.4],
                [1.5, -2.9],
            ]
        ),
    )

    bbox = sc.Polygon(
        "bounding_box", layer="BE", points=sc.geometry.box(5, 5, center=(0.85, -0.85))
    )

    films = [fc_shield, fc, pl_shield1, pl_shield2, pl]
    holes = [fc_center, pl_center]
    for polygon in films + holes:
        if "shield" in polygon.name:
            polygon.points = polygon.resample(interp_points // 2)
        else:
            polygon.points = polygon.resample(interp_points)

    return sc.Device(
        "ibm_300nm",
        layers=ibm_squid_layers(align=align_layers),
        films=films,
        holes=holes,
        abstract_regions=[bbox],
        length_units="um",
    )
