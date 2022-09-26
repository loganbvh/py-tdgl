import numpy as np
import superscreen as sc


def huber_geometry(interp_points=101):

    ri_pl = 1.7
    ro_pl = 2.7
    w_pl_center = 1.18
    w_pl_outer = 3.10
    pl_angle = 0
    pl_total_length = 15
    y0_pl_leads = -(pl_total_length - ro_pl)

    x0_pl_center = w_pl_center / 2
    theta0_pl_center = np.arcsin(x0_pl_center / ri_pl)
    thetas_pl_center = (
        np.linspace(theta0_pl_center, 2 * np.pi - theta0_pl_center, 101) - np.pi / 2
    )

    x0_pl_outer = w_pl_outer / 2
    theta0_pl_outer = np.arcsin(x0_pl_outer / ro_pl)
    thetas_pl_outer = (
        np.linspace(theta0_pl_outer, 2 * np.pi - theta0_pl_outer, 101) - np.pi / 2
    )

    pl_points = np.concatenate(
        [
            [[-w_pl_outer / 2, y0_pl_leads]],
            ro_pl
            * np.stack([np.cos(thetas_pl_outer), np.sin(thetas_pl_outer)], axis=1)[
                ::-1
            ],
            [[w_pl_outer / 2, y0_pl_leads]],
            [[-w_pl_outer / 2, y0_pl_leads]],
        ]
    )
    pl_points = sc.geometry.rotate(pl_points, pl_angle)

    pl_center = np.concatenate(
        [
            [[w_pl_center / 2, y0_pl_leads + (ro_pl - ri_pl)]],
            ri_pl
            * np.stack([np.cos(thetas_pl_center), np.sin(thetas_pl_center)], axis=1),
            [[-w_pl_center / 2, y0_pl_leads + (ro_pl - ri_pl)]],
        ]
    )
    pl_center = sc.geometry.rotate(pl_center, pl_angle)

    pl_shield = np.concatenate(
        [
            [[-(w_pl_outer / 2 + 0.25), -(ri_pl + 0.5)]],
            [[-w_pl_outer / 2, -(ri_pl + 0.25)]],
            [[+w_pl_outer / 2, -(ri_pl + 0.25)]],
            [[+(w_pl_outer / 2 + 0.25), -(ri_pl + 0.5)]],
            [[+(w_pl_outer / 2 + 0.25), y0_pl_leads - 0.5]],
            [[-(w_pl_outer / 2 + 0.25), y0_pl_leads - 0.5]],
            [[-(w_pl_outer / 2 + 0.25), -(ri_pl + 0.5)]],
        ]
    )
    pl_shield = sc.geometry.rotate(pl_shield, pl_angle)

    ri_fc = 5.5
    ro_fc = 8.0
    w_fc_outer = 7.0
    w_fc_center = 1.6
    fc_angle = 45

    fc_center_length = 6
    y0_fc_center_leads = -(fc_center_length + ri_fc)

    x0_fc_center = w_fc_center / 2
    theta0_fc_center = np.arcsin(x0_fc_center / ri_fc)
    # y0_fc_center = ri_fc * np.cos(theta0_fc_center)
    thetas_fc_center = (
        np.linspace(theta0_fc_center, 2 * np.pi - theta0_fc_center, 101) - np.pi / 2
    )

    fc_center_points = np.concatenate(
        [
            [[-w_fc_center / 2, y0_fc_center_leads]],
            ri_fc
            * np.stack([np.cos(thetas_fc_center), np.sin(thetas_fc_center)], axis=1)[
                ::-1
            ],
            [[w_fc_center / 2, y0_fc_center_leads]],
            [[-w_fc_center / 2, y0_fc_center_leads]],
        ]
    )
    fc_center_points = sc.geometry.rotate(fc_center_points, fc_angle)

    fc_outer_length = 6
    y0_fc_outer_leads = -(fc_outer_length + ro_fc)

    x0_fc_outer = w_fc_outer / 2
    theta0_fc_outer = np.arcsin(x0_fc_outer / ro_fc)
    # y0_fc_outer = ri_fc * np.cos(theta0_fc_outer)
    thetas_fc_outer = (
        np.linspace(theta0_fc_outer, 2 * np.pi - theta0_fc_outer, 101) - np.pi / 2
    )

    fc_outer_points = np.concatenate(
        [
            [[-w_fc_outer / 2, y0_fc_outer_leads]],
            ro_fc
            * np.stack([np.cos(thetas_fc_outer), np.sin(thetas_fc_outer)], axis=1)[
                ::-1
            ],
            [[w_fc_outer / 2, y0_fc_outer_leads]],
            [[-w_fc_outer / 2, y0_fc_outer_leads]],
        ]
    )
    fc_outer_points = sc.geometry.rotate(fc_outer_points, fc_angle)

    w_fc_shield = 10
    w0_fc_shield = 2
    y0_fc_shield = -(ro_fc + 1)
    y1_fc_shield = -(ri_fc - 0.5)

    fc_shield = np.concatenate(
        [
            [[-w_fc_shield / 2, y0_fc_outer_leads - 1]],
            [[-w_fc_shield / 2, y0_fc_shield]],
            [[-w0_fc_shield / 2, y1_fc_shield]],
            [[+w0_fc_shield / 2, y1_fc_shield]],
            [[+w_fc_shield / 2, y0_fc_shield]],
            [[+w_fc_shield / 2, y0_fc_outer_leads - 1]],
            [[-w_fc_shield / 2, y0_fc_outer_leads - 1]],
        ]
    )
    fc_shield = sc.geometry.rotate(fc_shield, fc_angle)

    polygons = {
        "pl": pl_points,
        "pl_shield": pl_shield,
        "pl_center": pl_center,
        "fc": fc_outer_points,
        "fc_center": fc_center_points,
        "fc_shield": fc_shield,
    }

    if interp_points is not None:
        from scipy.interpolate import splev, splprep

        new_polygons = {}
        for name, points in polygons.items():
            x, y = np.array(points).T
            tck, u = splprep([x, y], s=0, k=1)
            new_points = splev(np.linspace(0, 1, interp_points), tck)
            new_polygons[name] = np.stack(new_points, axis=1)
        polygons = new_polygons

    return polygons


def make_squid():

    interp_points = 151

    # See Nick Koshnick thesis
    # bottom of page 29 and table 3.2 on page 32).

    d_be = 0.2
    d_i1 = 0.350
    d_w1 = 0.23
    d_i2 = 0.350
    d_w2 = 0.25

    # # align ==  "middle"
    # z0_w2 = d_w2 / 2
    # z0_w1 = z0_w2 + d_w2 / 2 + d_i2 + d_w1 / 2
    # z0_be = d_w2 + d_i2 + d_w1 + d_i1 + d_be / 2

    # align == "bottom"
    z0 = 0
    z0_w2 = z0
    z0_w1 = z0_w2 + d_w2 + d_i2
    z0_be = z0_w2 + d_w2 + d_i2 + d_w1 + d_i1

    polygons = huber_geometry(interp_points=interp_points)

    layers = [
        sc.Layer("BE", london_lambda=0.08, thickness=d_be, z0=z0_be),
        sc.Layer("W1", london_lambda=0.08, thickness=d_w1, z0=z0_w1),
        sc.Layer("W2", london_lambda=0.08, thickness=d_w2, z0=z0_w2),
    ][::-1]

    films = [
        sc.Polygon("fc", layer="BE", points=polygons["fc"]),
        sc.Polygon("fc_shield", layer="W1", points=polygons["fc_shield"]),
        sc.Polygon("pl", layer="W1", points=polygons["pl"]),
        sc.Polygon("pl_shield", layer="W2", points=polygons["pl_shield"]),
    ]

    holes = [
        sc.Polygon("fc_center", layer="BE", points=polygons["fc_center"]),
        sc.Polygon("pl_center", layer="W1", points=polygons["pl_center"]),
    ]

    bbox = np.array(
        [
            [-9.0, -15.0],
            [-9.0, 9.0],
            [14.5, 9.0],
            [14.5, -15.0],
        ]
    )

    abstract_regions = [
        sc.Polygon("bounding_box", layer="W1", points=bbox),
    ]

    device = sc.Device(
        "huber_squid",
        layers=layers,
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
        length_units="um",
    )
    return device
