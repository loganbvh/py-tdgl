import numpy as np

from tdgl.geometry import path_vectors


def test_path_vectors_unit_normals():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    edge_lengths, normals = path_vectors(path)
    assert np.allclose(edge_lengths, [1.0, 1.0])
    assert normals.shape == (2, 2)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.allclose(normals, [[0.0, -1.0], [1.0, 0.0]])
