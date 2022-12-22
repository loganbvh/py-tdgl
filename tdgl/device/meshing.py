import logging
from typing import List, Tuple, Union

import numpy as np
from meshpy import triangle
from scipy import spatial
from shapely.geometry.polygon import Polygon

from ..finite_volume.util import get_edge_lengths
from ..geometry import ensure_unique

logger = logging.getLogger(__name__)


def generate_mesh(
    poly_coords: np.ndarray,
    hole_coords: Union[List[np.ndarray], None] = None,
    min_points: Union[int, None] = None,
    max_edge_length: Union[float, None] = None,
    convex_hull: bool = False,
    boundary: Union[np.ndarray, None] = None,
    min_angle: float = 32.5,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a Delaunay mesh for a given set of polygon vertex coordinates.

    Additional keyword arguments are passed to ``triangle.build()``.

    Args:
        poly_coords: Shape ``(n, 2)`` array of polygon ``(x, y)`` coordinates.
        hole_coords: A list of arrays of hole boundary coordinates.
        min_points: The minimimum number of vertices in the resulting mesh.
        max_edge_length: The maximum distance between vertices in the resulting mesh.
        convex_hull: If True, then the entire convex hull of the polygon (minus holes)
            will be meshed. Otherwise, only the polygon interior is meshed.
        boundary: Shape ``(m, 2)`` (where ``m <= n``) array of ``(x, y)`` coordinates
            for points on the boundary of the polygon.
        min_angle: The minimum angle in the mesh's triangles. Setting a larger value
            will make the triangles closer to equilateral, but the mesh generation
            may fail if the value is too large.

    Returns:
        Mesh vertex coordinates and triangle indices.
    """
    poly_coords = ensure_unique(poly_coords)
    if hole_coords is None:
        hole_coords = []
    hole_coords = [ensure_unique(coords) for coords in hole_coords]
    # Facets is a shape (m, 2) array of edge indices.
    # coords[facets] is a shape (m, 2, 2) array of edge coordinates:
    # [(x0, y0), (x1, y1)]
    coords = np.concatenate([poly_coords] + hole_coords, axis=0)
    xmin = coords[:, 0].min()
    dx = np.ptp(coords[:, 0])
    ymin = coords[:, 1].min()
    dy = np.ptp(coords[:, 1])
    r0 = np.array([[xmin, ymin]]) + np.array([[dx, dy]]) / 2
    # Center the coordinates at (0, 0) to avoid floating point issues.
    coords = coords - r0
    indices = np.arange(len(poly_coords), dtype=int)
    if convex_hull:
        if boundary is not None:
            raise ValueError(
                "Cannot have both boundary is not None and convex_hull = True."
            )
        facets = spatial.ConvexHull(coords).simplices
    else:
        if boundary is not None:
            boundary = list(map(tuple, boundary))
            indices = [i for i in indices if tuple(coords[i]) in boundary - r0]
        facets = np.array([indices, np.roll(indices, -1)]).T
    # Create facets for the holes.
    for hole in hole_coords:
        hole_indices = np.arange(
            indices[-1] + 1, indices[-1] + 1 + len(hole), dtype=int
        )
        hole_facets = np.array([hole_indices, np.roll(hole_indices, -1)]).T
        indices = np.concatenate([indices, hole_indices], axis=0)
        facets = np.concatenate([facets, hole_facets], axis=0)

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(coords)
    mesh_info.set_facets(facets)
    if hole_coords:
        # Triangle allows you to set holes by specifying a single point
        # that lies in each hole. Here we use the centroid of the hole.
        holes = [
            np.array(Polygon(hole).centroid.coords[0]) - r0.squeeze()
            for hole in hole_coords
        ]
        mesh_info.set_holes(holes)

    if "min_angle" not in kwargs:
        kwargs["min_angle"] = min_angle

    mesh = triangle.build(mesh_info=mesh_info, **kwargs)
    points = np.array(mesh.points) + r0
    triangles = np.array(mesh.elements)
    if min_points is None and (max_edge_length is None or max_edge_length <= 0):
        return points, triangles

    kwargs = kwargs.copy()
    kwargs["max_volume"] = dx * dy / 100
    i = 1
    if min_points is None:
        min_points = 0
    if max_edge_length is None or max_edge_length <= 0:
        max_edge_length = np.inf
    max_length = get_edge_lengths(points, triangles).max()
    while (len(points) < min_points) or (max_length > max_edge_length):
        mesh = triangle.build(mesh_info=mesh_info, **kwargs)
        points = np.array(mesh.points) + r0
        triangles = np.array(mesh.elements)
        max_length = get_edge_lengths(points, triangles).max()
        logger.debug(
            f"Iteration {i}: Made mesh with {len(points)} points and "
            f"{len(triangles)} triangles with maximum edge length: "
            f"{max_length:.2e}. Target maximum edge length: {max_edge_length:.2e}."
        )
        if np.isfinite(max_edge_length):
            kwargs["max_volume"] *= min(0.98, np.sqrt(max_edge_length / max_length))
        else:
            kwargs["max_volume"] *= 0.98
        i += 1
    return points, triangles
