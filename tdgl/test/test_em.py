import numpy as np
import pytest

import tdgl
from tdgl import em
from tdgl.geometry import circle

ureg = tdgl.ureg


@pytest.mark.parametrize("length_units", ["um", "nm"])
@pytest.mark.parametrize("current_units", ["uA", "mA"])
@pytest.mark.parametrize("z_eval", [1, 2, 5, 10])
def test_current_loop(z_eval, current_units, length_units):
    r_loop = 10
    z_loop = 0
    r_eval = 5
    eval_points = 3000
    current = 1

    pl = circle(r_eval, points=1001)
    pl_edge_centers = ((pl + np.roll(pl, -1, axis=0)) / 2)[:-1]
    pl_edges = np.diff(pl, axis=0) * ureg(length_units)
    pl_mesh = tdgl.Polygon(points=pl).make_mesh(min_points=eval_points)
    pl_points = np.concatenate(
        [pl_mesh.sites, z_eval * np.ones((len(pl_mesh.sites), 1))], axis=1
    )
    pl_areas = pl_mesh.areas * ureg(length_units) ** 2

    B = em.current_loop_field(
        pl_points,
        current=current,
        loop_radius=r_loop,
        loop_center=np.array([0, 0, z_loop]),
        num_segments=201,
        length_units=length_units,
        current_units=current_units,
    )

    A = em.current_loop_vector_potential(
        np.concatenate(
            [pl_edge_centers, z_eval * np.ones_like(pl_edge_centers[:, :1])], axis=1
        ),
        current=current,
        loop_radius=r_loop,
        loop_center=np.array([0, 0, z_loop]),
        length_units=length_units,
        current_units=current_units,
    )

    flux_from_B = np.sum(B[:, 2] * pl_areas).to("Phi_0").magnitude
    flux_from_A = np.trapz((A[:, :2] * pl_edges).sum(axis=1)).to("Phi_0").magnitude

    assert np.isclose(flux_from_A, flux_from_B, rtol=1e-2)
