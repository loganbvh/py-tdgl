import logging
from typing import Optional, Tuple

import numpy as np
import optimesh
from matplotlib.tri import Triangulation
from meshpy import triangle
from scipy import spatial
from shapely.geometry import MultiLineString
from shapely.geometry.polygon import Polygon, orient
from shapely.ops import polygonize

from ..fem import edge_lengths

logger = logging.getLogger(__name__)


def ensure_unique(coords: np.ndarray) -> np.ndarray:
    # Coords is a shape (n, 2) array of vertex coordinates.
    coords = np.asarray(coords)
    # Remove duplicate coordinates, otherwise triangle.build() will segfault.
    # By default, np.unique() does not preserve order, so we have to remove
    # duplicates this way:
    _, ix = np.unique(coords, return_index=True, axis=0)
    coords = coords[np.sort(ix)]
    return coords


def generate_mesh(
    poly_coords: np.ndarray,
    hole_coords: Optional[list[np.ndarray]] = None,
    min_points: Optional[int] = None,
    max_edge_length: Optional[float] = None,
    convex_hull: bool = False,
    boundary: Optional[np.ndarray] = None,
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
    indices = np.arange(poly_coords.shape[0], dtype=int)
    if convex_hull:
        if boundary is not None:
            raise ValueError(
                "Cannot have both boundary is not None and convex_hull = True."
            )
        facets = spatial.ConvexHull(coords).simplices
    else:
        if boundary is not None:
            boundary = list(map(tuple, boundary))
            indices = [i for i in indices if tuple(coords[i]) in boundary]
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
        holes = [Polygon(hole).centroid.coords[0] for hole in hole_coords]
        mesh_info.set_holes(holes)

    mesh = triangle.build(mesh_info=mesh_info, **kwargs)
    points = np.array(mesh.points)
    triangles = np.array(mesh.elements)

    if min_points is None and (max_edge_length is None or max_edge_length <= 0):
        return points, triangles

    kwargs = kwargs.copy()
    kwargs["max_volume"] = 1
    i = 1
    if min_points is None:
        min_points = 0
    if max_edge_length is None or max_edge_length <= 0:
        max_edge_length = np.inf
    max_length = edge_lengths(points, triangles).max()
    while (points.shape[0] < min_points) or (max_length > max_edge_length):
        mesh = triangle.build(mesh_info=mesh_info, **kwargs)
        points = np.array(mesh.points)
        triangles = np.array(mesh.elements)
        max_length = edge_lengths(points, triangles).max()
        logger.debug(
            f"Iteration {i}: Made mesh with {points.shape[0]} points and "
            f"{triangles.shape[0]} triangles with maximum edge length: "
            f"{max_length:.2e}. Target maximum edge length: {max_edge_length:.2e}."
        )
        kwargs["max_volume"] *= 0.98
        i += 1
    return points, triangles


def optimize_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    steps: int,
    method: str = "cvt-block-diagonal",
    tolerance: float = 1e-3,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimizes an existing mesh using ``optimesh``.

    See ``optimesh`` documentation for additional options.

    Args:
        points: Mesh vertex coordinates.
        triangles: Mesh triangle indices.
        steps: Number of optimesh steps to perform.
        method: See ``optimesh`` documentation.
        tolerance: See ``optimesh`` documentation.
        verbose: See ``optimesh`` documentation.

    Returns:
        Optimized mesh vertex coordinates and triangle indices.
    """
    points, triangles = optimesh.optimize_points_cells(
        points,
        triangles,
        method,
        tolerance,
        steps,
        verbose=verbose,
        **kwargs,
    )
    return points, triangles


def boundary_vertices(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Returns an array of boundary vertex indices, ordered counterclockwise.

    Args:
        points: Shape ``(n, 2)`` array of vertex coordinates.
        triangles: Shape ``(m, 3)`` array of triangle indices.

    Returns:
        An array of boundary vertex indices, ordered counterclockwise.
    """
    tri = Triangulation(points[:, 0], points[:, 1], triangles)
    boundary_edges = set()
    for i, neighbors in enumerate(tri.neighbors):
        for k in range(3):
            if neighbors[k] == -1:
                boundary_edges.add((triangles[i, k], triangles[i, (k + 1) % 3]))
    edges = MultiLineString([points[edge, :] for edge in boundary_edges])
    polygons = list(polygonize(edges))
    assert len(polygons) == 1, polygons
    polygon = orient(polygons[0])
    points_list = [tuple(xy) for xy in points]
    indices = np.array([points_list.index(xy) for xy in polygon.exterior.coords])
    return indices[:-1]
