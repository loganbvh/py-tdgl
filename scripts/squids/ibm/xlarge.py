"""IBM SQUID susceptometer, 3 um inner radius pickup loop."""

import numpy as np
import superscreen as sc

from .layers import ibm_squid_layers


def make_squid(
    interp_points: int = 310,
    align_layers: str = "middle",
    d_I1: float = 0.4,
    d_I2: float = 0.4,
) -> sc.Device:
    pl_length = 11.5
    ri_pl = 3.0
    ro_pl = 3.5
    ri_fc = 6.0
    ro_fc = 8.8

    pl_center = sc.Polygon(
        "pl_center",
        layer="W1",
        points=sc.geometry.circle(ri_pl),
    ).union(sc.geometry.box(0.314, pl_length, center=(0, -pl_length / 2 - 0.9 * ri_pl)))

    pl = sc.Polygon("pl", layer="W1", points=sc.geometry.circle(ro_pl)).union(
        np.array(
            [
                [+0.8, -2.7],
                [-0.8, -2.7],
                [-4.6, -15.0],
                [+4.6, -15.0],
            ]
        )
    )

    pl_shield1 = sc.Polygon(
        "pl_shield1",
        layer="W2",
        points=np.array(
            [
                [+2.6, -6.3],
                [+1.3, -3.6],
                [-1.3, -3.6],
                [-2.6, -6.3],
                [-6.0, -16.0],
                [+6.0, -16.0],
            ]
        ),
    )

    pl_shield2 = sc.Polygon(
        "pl_shield2",
        layer="BE",
        points=np.array(
            [
                [+4.5, -13.2],
                [-4.5, -13.2],
                [-5.3, -15.5],
                [+5.3, -15.5],
            ]
        ),
    )

    fc_center = sc.Polygon(
        "fc_center", layer="BE", points=sc.geometry.circle(ri_fc)
    ).union(
        np.array(
            [
                [8.5, -10.3],
                [4.15, -4.15],
                [3.55, -4.75],
                [7.75, -10.75],
            ]
        )
    )

    fc = sc.Polygon("fc", layer="BE", points=sc.geometry.circle(ro_fc)).union(
        np.array(
            [
                [12.0, -9.6],
                [7.5, -4.8],
                [4.2, -4.2],
                [3.2, -7.8],
                [6.0, -13.5],
            ]
        )
    )

    fc_shield = sc.Polygon(
        "fc_shield",
        layer="W1",
        points=np.array(
            [
                [13.3, -10.2],
                [7.7, -4.8],
                [3.3, -8.1],
                [6.1, -15.0],
            ]
        ),
    )

    bbox = sc.Polygon(
        "bounding_box", layer="BE", points=sc.geometry.box(24, 28, center=(2, -3.5))
    )

    films = [fc_shield, fc, pl_shield1, pl_shield2, pl]
    holes = [fc_center, pl_center]
    abstract_regions = [
        bbox,
        sc.Polygon(
            "circle",
            layer="W1",
            points=sc.geometry.circle(ri_pl / 2, points=interp_points // 5),
        ),
    ]
    for polygon in films + holes:
        if "shield" in polygon.name:
            polygon.points = polygon.resample(interp_points // 2)
        else:
            polygon.points = polygon.resample(interp_points)

    return sc.Device(
        "ibm_3000nm",
        layers=ibm_squid_layers(
            align=align_layers,
            d_I1=d_I1,
            d_I2=d_I2,
        ),
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
        length_units="um",
    )
