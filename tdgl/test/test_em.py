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
    flux_from_A = np.trapezoid((A[:, :2] * pl_edges).sum(axis=1)).to("Phi_0").magnitude

    assert np.isclose(flux_from_A, flux_from_B, rtol=1e-2)


def test_biot_savart_input_validation():
    eval_positions = np.array([[2.0, 0.0, 1.0]])
    current_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    current_vectors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    currents = np.array([1.0, 1.0])

    B = em.biot_savart(
        eval_positions,
        current_positions=current_positions,
        current_vectors=current_vectors,
        currents=currents,
    )
    assert B.shape == (1, 3)

    # Shape (m, 1) is documented as valid for currents.
    B = em.biot_savart(
        eval_positions,
        current_positions=current_positions,
        current_vectors=current_vectors,
        currents=currents[:, np.newaxis],
    )
    assert B.shape == (1, 3)

    with pytest.raises(ValueError, match="eval_positions"):
        em.biot_savart(
            np.zeros((2, 2)),
            current_positions=current_positions,
            current_vectors=current_vectors,
            currents=currents,
        )
    with pytest.raises(ValueError, match="current_positions"):
        em.biot_savart(
            eval_positions,
            current_positions=np.zeros((2, 2)),
            current_vectors=current_vectors,
            currents=currents,
        )
    with pytest.raises(ValueError, match="current_vectors"):
        em.biot_savart(
            eval_positions,
            current_positions=current_positions,
            current_vectors=np.zeros((1, 3)),
            currents=currents,
        )
    with pytest.raises(ValueError, match=r"currents must all have"):
        em.biot_savart(
            eval_positions,
            current_positions=current_positions,
            current_vectors=current_vectors,
            currents=np.zeros(3),
        )


def test_biot_savart_2d_input_validation():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    current_densities = np.ones((3, 2))

    B = em.biot_savart_2d(
        0.5,
        0,
        1,
        positions=positions,
        current_densities=current_densities,
        areas=np.full(3, 0.1),
    )
    assert B.shape == (1, 3)

    with pytest.raises(ValueError, match="positions"):
        em.biot_savart_2d(
            0.5, 0, 1, positions=np.ones((3, 3)), current_densities=current_densities
        )
    with pytest.raises(ValueError, match="current_densities"):
        em.biot_savart_2d(
            0.5, 0, 1, positions=positions, current_densities=np.ones((2, 2))
        )
    with pytest.raises(ValueError, match="areas"):
        em.biot_savart_2d(
            0.5,
            0,
            1,
            positions=positions,
            current_densities=current_densities,
            areas=np.ones(4),
        )


def test_uniform_Bz_vector_potential_input_validation():
    positions = np.zeros((4, 3))

    A = em.uniform_Bz_vector_potential(positions, 1.0)
    assert A.shape == (4, 3)

    with pytest.raises(TypeError, match="Bz"):
        em.uniform_Bz_vector_potential(positions, [1.0])
    with pytest.raises(ValueError, match="positions"):
        em.uniform_Bz_vector_potential(np.zeros((4, 2)), 1.0)
