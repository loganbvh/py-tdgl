"""IBM SQUID susceptometer, 1 um inner radius pickup loop."""

import numpy as np
import superscreen as sc

from .layers import ibm_squid_layers


def make_squid(interp_points=201, align_layers="middle"):
    pl_length = 4
    ri_pl = 1.0
    ro_pl = 1.5
    ri_fc = 2.5
    ro_fc = 3.5
    pl_center = sc.Polygon(
        "pl_center",
        layer="W1",
        points=sc.geometry.circle(ri_pl),
    ).union(sc.geometry.box(0.2, pl_length, center=(0, -pl_length / 2 - 0.9 * ri_pl)))

    pl = sc.Polygon("pl", layer="W1", points=sc.geometry.circle(ro_pl)).union(
        np.array(
            [
                [1.5, -5.7],
                [0.41, -1],
                [-0.41, -1],
                [-1.5, -5.7],
            ]
        )
    )

    pl_shield1 = sc.Polygon(
        "pl_shield1",
        layer="W2",
        points=np.array(
            [
                [+1.0, -2.8],
                [+0.6, -(ri_pl + 0.4)],
                [-0.6, -(ri_pl + 0.4)],
                [-1.0, -2.8],
                [-2.6, -6.4],
                [-2.75, -6.9],
                [+2.75, -6.9],
                [+2.6, -6.4],
            ]
        ),
    )

    pl_shield2 = sc.Polygon(
        "pl_shield2",
        layer="BE",
        points=np.array(
            [
                [+1.25, -(2.55 + ro_pl)],
                [-1.25, -(2.55 + ro_pl)],
                [-2.0, -6.2],
                [+2.0, -6.2],
            ]
        ),
    )

    fc_center = sc.Polygon(
        "fc_center", layer="BE", points=sc.geometry.circle(ri_fc)
    ).union(
        np.array(
            [
                [4.3, -4.2],
                [2.1, -1.0],
                [1.8, -1.6],
                [3.85, -4.55],
            ]
        )
    )

    fc = sc.Polygon("fc", layer="BE", points=sc.geometry.circle(ro_fc)).union(
        np.array(
            [
                [5.8, -3.9],
                [2.8, -0.9],
                [1.5, -2.3],
                [3.2, -6.0],
            ]
        )
    )

    fc_shield = sc.Polygon(
        "fc_shield",
        layer="W1",
        points=np.array(
            [
                [6.4, -4.05],
                [3.45, -1.4],
                [1.65, -3.3],
                [3.1, -6.8],
            ]
        ),
    )

    bbox = sc.Polygon(
        "bounding_box",
        layer="BE",
        points=sc.geometry.box(10.5, 11, center=(1.35, -1.75)),
    )

    films = [fc_shield, fc, pl_shield1, pl_shield2, pl]
    holes = [fc_center, pl_center]
    for polygon in films + holes:
        if "shield" in polygon.name:
            polygon.points = polygon.resample(interp_points // 2)
        else:
            polygon.points = polygon.resample(interp_points)

    return sc.Device(
        "ibm_1000nm",
        layers=ibm_squid_layers(align=align_layers),
        films=films,
        holes=holes,
        abstract_regions=[bbox],
        length_units="um",
    )
