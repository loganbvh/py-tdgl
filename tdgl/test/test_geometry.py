import numpy as np
import pytest

from tdgl.geometry import box, path_vectors, rotate


def test_path_vectors_unit_normals():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    edge_lengths, normals = path_vectors(path)
    assert np.allclose(edge_lengths, [1.0, 1.0])
    assert normals.shape == (2, 2)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.allclose(normals, [[0.0, -1.0], [1.0, 0.0]])


def test_rotate():
    coords = box(2, 2)
    rotated = rotate(coords, 90)
    assert rotated.shape == coords.shape
    # Rotating by 90 degrees counterclockwise maps (x, y) to (-y, x).
    assert np.allclose(rotated, coords[:, ::-1] * [-1, 1])


def test_rotate_invalid_shape():
    with pytest.raises(ValueError, match="coords"):
        rotate(np.zeros(5), 90)
    with pytest.raises(ValueError, match="coords"):
        rotate(np.zeros((5, 3)), 90)
