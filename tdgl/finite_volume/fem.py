from typing import Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from matplotlib.path import Path


def in_polygon(
    poly_points: np.ndarray,
    query_points: np.ndarray,
    radius: float = 0,
) -> Union[bool, np.ndarray]:
    """Returns a boolean array indicating which points in ``query_points``
    lie inside the polygon defined by ``poly_points``.

    Args:
        poly_points: Shape ``(m, 2)`` array of polygon vertex coordinates.
        query_points: Shape ``(n, 2)`` array of "query points".
        radius: See :meth:`matplotlib.path.Path.contains_points`.

    Returns:
        A shape ``(n, )`` boolean array indicating which ``query_points``
        lie inside the polygon. If only a single query point is given, then
        a single boolean value is returned.
    """
    query_points, poly_points = np.atleast_2d(query_points, poly_points)
    path = Path(poly_points)
    bool_array = path.contains_points(query_points, radius=radius).squeeze()
    if len(bool_array.shape) == 0:
        bool_array = bool_array.item()
    return bool_array


def centroids(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Returns x, y coordinates for triangle centroids (centers of mass).

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangle indices.

    Returns:
        Shape (m, 2) array of triangle centroid (center of mass) coordinates
    """
    return points[triangles].sum(axis=1) / 3


def triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Calculates the area of each triangle.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices
        triangles: Shape (m, 3) array of triangle indices

    Returns:
        Shape (m, ) array of triangle areas
    """
    xy = points[triangles]
    # s1 = xy[:, 2, :] - xy[:, 1, :]
    # s2 = xy[:, 0, :] - xy[:, 2, :]
    # s3 = xy[:, 1, :] - xy[:, 0, :]
    # which can be simplified to
    # s = xy[:, [2, 0, 1]] - xy[:, [1, 2, 0]]  # 3D
    s = xy[:, [2, 0]] - xy[:, [1, 2]]  # 2D
    a = np.linalg.det(s)
    return a * 0.5


def mass_matrix(
    points: np.ndarray,
    triangles: np.ndarray,
    sparse: bool = False,
) -> Union[np.ndarray, sp.csc_matrix]:
    """The mass matrix defines an effective area for each vertex.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangle indices.
        sparse: Whether to return a sparse matrix or numpy ndarray.

    Returns:
        Shape (n, n) sparse mass matrix or shape (n,) vector of diagonals.
    """
    # Adapted from spharaphy.TriMesh:
    # https://spharapy.readthedocs.io/en/latest/modules/trimesh.html
    # https://gitlab.com/uwegra/spharapy/-/blob/master/spharapy/trimesh.py
    N = points.shape[0]
    if sparse:
        mass = sp.lil_matrix((N, N), dtype=float)
    else:
        mass = np.zeros((N, N), dtype=float)

    tri_areas = triangle_areas(points, triangles)

    for a, t in zip(tri_areas / 3, triangles):
        mass[t[0], t[0]] += a
        mass[t[1], t[1]] += a
        mass[t[2], t[2]] += a

    if sparse:
        # Use csc_matrix because we will eventually invert the mass matrix,
        # and csc is efficient for inversion.
        return mass.tocsc()
    return mass.diagonal()


def edge_lengths(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [
            points[triangles[:, [0, 1]]],
            points[triangles[:, [1, 2]]],
            points[triangles[:, [2, 0]]],
        ]
    )
    return la.norm(np.diff(edges, axis=1), axis=2)


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Normalizes ``vector``."""
    return vector / la.norm(vector, axis=-1)[:, np.newaxis]


def path_vectors(path: np.ndarray) -> Tuple[float, np.ndarray]:
    """Computes the total length and the unit normals for a path.

    Args:
        path: Shape ``(n, 2)`` array of coordinates representing a continuous path.

    Returns:
        The total path length and an array of of unit vectors normal to
        each edge in the path.
    """
    dr = np.diff(path, axis=0)
    normals = np.cross(dr, [0, 0, 1])
    unit_normals = unit_vector(normals)
    total_length = np.sum(la.norm(dr, axis=1))
    unit_normals = np.concatenate([unit_normals, unit_normals[-1:]], axis=0)
    return total_length, unit_normals[:, :2]
