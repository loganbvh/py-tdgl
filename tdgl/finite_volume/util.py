import multiprocessing as mp
from functools import partial
from typing import List, Tuple

import joblib
import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull, Delaunay, QhullError
from shapely.geometry import MultiLineString
from shapely.ops import orient, polygonize


def get_edges(elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the edges from a list of triangle indices.

    Args:
        elements: The triangle indices, shape ``(n, 3)``.

    Returns:
        A tuple containing an integer array of edges and a boolean array
        indicating whether each edge on in the boundary.
    """
    edges = np.concatenate([elements[:, e] for e in [(0, 1), (1, 2), (2, 0)]])
    edges = np.sort(edges, axis=1)
    edges, counts = np.unique(edges, return_counts=True, axis=0)
    return edges, counts == 1


def get_edge_lengths(points: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Returns the lengths of all edges in a triangulation.

    Args:
        points: Vertex coordinates.
        elements: Triangle indices.

    Returns:
        An array of edge lengths.
    """
    edges = np.concatenate([points[elements[:, e]] for e in [(0, 1), (1, 2), (2, 0)]])
    return np.linalg.norm(np.diff(edges, axis=1), axis=2)


def get_dual_edge_lengths(
    edge_centers: np.ndarray,
    elements: np.ndarray,
    dual_sites: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """
    Compute the lengths of the dual edges.

    Args:
        edge_centers: The (x, y) coordinates of the edge centers.
        elements: The triangular elements in the tesselation.
        dual_sites: The (x, y) coordinates for the dual mesh (Voronoi sites).
        edges: The edges connecting the sites.

    Returns:
        An array of dual edge lengths.
    """
    # Create a dict with keys corresponding to the edges and values
    # corresponding to the triangles
    edge_to_element = {}
    # Iterate over all elements to create the edge_to_element dict
    edge_element_indices = [[0, 1], [1, 2], [2, 0]]
    for i, element in enumerate(elements):
        for idx in edge_element_indices:
            # Make the array hashable by converting it to a tuple
            edge = tuple(np.sort(element[idx]))
            if edge in edge_to_element:
                edge_to_element[edge].append(i)
            else:
                edge_to_element[edge] = [i]
    dual_lengths = np.zeros(edge_centers.shape[0], dtype=float)
    for i, edge in enumerate(edges):
        indices = edge_to_element[tuple(edge)]
        if len(indices) == 1:  # Boundary edges
            dual_lengths[i] = np.linalg.norm(dual_sites[indices[0]] - edge_centers[i])
        else:  # Inner edges
            dual_lengths[i] = np.linalg.norm(
                dual_sites[indices[0]] - dual_sites[indices[1]]
            )
    return dual_lengths


def generate_voronoi_vertices(
    sites: np.ndarray, elements: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the vertices of the Voronoi lattice by computing the
    circumcenters of the triangles in the tesselation.

    Args:
        sites: The (x, y) coordinates of the tesselation.
        elements: The triangular elements in the tesselation.

    Returns:
        The x and y coordinates of the Voronoi vertices, as arrays.
    """
    # Get the triangle abc
    # Convert to the coordinate system where a is in the origin
    a = sites[elements[:, 0]]
    bp = sites[elements[:, 1]] - a
    cp = sites[elements[:, 2]] - a
    denominator = 2 * (bp[:, 0] * cp[:, 1] - bp[:, 1] * cp[:, 0])
    # Compute the circumcenter
    xcp = (
        cp[:, 1] * (bp**2).sum(axis=1) - bp[:, 1] * (cp**2).sum(axis=1)
    ) / denominator
    ycp = (
        bp[:, 0] * (cp**2).sum(axis=1) - cp[:, 0] * (bp**2).sum(axis=1)
    ) / denominator
    # Convert back to the initial coordinate system
    return np.array([xcp, ycp]).T + a


def _get_polygon_indices(
    elements: np.ndarray, site_index: int
) -> Tuple[int, np.ndarray]:
    """Helper function for get_surrounding_voronoi_polygons()."""
    return np.where((elements == site_index).any(axis=1))[0]


def get_surrounding_voronoi_polygons(
    elements: np.ndarray,
    num_sites: int,
    parallel: bool = True,
    min_sites_for_multiprocessing: int = 10_000,
) -> List[np.ndarray]:
    """Find the polygons surrounding each site.

    Args:
        elements: The triangular elements in the tesselation.
        num_sites: The number of sites
        parallel: If True and the number of sites is greater than
            ``min_sites_for_multiprocessing``, then use multiprocessing.
        min_sites_for_multiprocessing: If ``parallel`` is True and ``num_sites``
            is greater than this value, then use multiprocessing.

    Returns:
        A list of arrays of Voronoi polygon indices.
    """
    # Iterate over all sites and find the triangles that the site belongs to
    # The indices for the triangles are the same as the indices for the
    # Voronoi lattice
    # This is by far the costliest step in Mesh.from_triangulation(),
    # so by default we will use multiprocessing if there are many sites.
    if (
        parallel
        and (ncpus := joblib.cpu_count(only_physical_cores=True)) > 1
        and num_sites > min_sites_for_multiprocessing
    ):
        with mp.Pool(processes=ncpus) as pool:
            results = pool.map(
                partial(_get_polygon_indices, elements), range(num_sites)
            )
        return results
    return [np.where((elements == i).any(axis=1))[0] for i in range(num_sites)]


def compute_surrounding_area(
    sites: np.ndarray,
    dual_sites: np.ndarray,
    boundary: np.ndarray,
    edges: np.ndarray,
    boundary_edge_indices: np.ndarray,
    polygons: List[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute the areas of the surrounding polygons.

    Areas of boundary points are handled by adding additional points
    on the boundary to make a convex polygon.

    Args:
        sites: The (x, y) coordinates for the sites.
        dual_sites: The (x, y) coordinates of the dual (Voronoi) sites.
        boundary: An array containing all boundary points.
        edges: The edges of the triangles.
        boundary_edge_indices: The edge indices corresponding to the boundary.
        polygons: The polygons in Voronoi diagram.

    Returns:
        An array of areas for each site in the lattice, and a list of Voronoi
        polygon vertices.
    """

    boundary_set = set(boundary)
    boundary_edges = edges[boundary_edge_indices]
    areas = np.zeros(len(polygons))
    voronoi_sites = []

    for site, polygon in enumerate(polygons):
        # Get the polygon points
        poly = dual_sites[polygon]
        # Polygon vertices may end up very close (e.g. delta = 2e-17) due to floating
        # point errors, so we need to remove near-duplicate vertices.
        _, unique = np.unique(poly.round(decimals=13), axis=0, return_index=True)
        poly = poly[unique]
        if site not in boundary_set:
            areas[site], is_convex = get_convex_polygon_area(poly)
            assert is_convex
            voronoi_sites.append(poly)
            continue
        # For points on the boundary, add vertices at the mesh site and the midpoints
        # of the two edges adjacent to the mesh site to complete the Voronoi polygons.
        connected_boundary_edges = boundary_edges[(boundary_edges == site).any(axis=1)]
        # The midpoints of the two edges adjacent to the site
        midpoints = sites[connected_boundary_edges].mean(axis=1)
        poly = np.concatenate([poly, [sites[site]], midpoints], axis=0)
        areas[site], is_convex = get_convex_polygon_area(poly)
        # If the polygon is non-convex we need to subtract the area of the
        # concave part, which is the triangle on the boundary.
        if not is_convex:
            # Does this ever actually happen?
            concave_area, is_convex = get_convex_polygon_area(
                np.concatenate([[sites[site]], midpoints], axis=0)
            )
            assert is_convex
            areas[site] -= concave_area
        voronoi_sites.append(poly)
    return areas, voronoi_sites


def get_convex_polygon_area(coords: np.ndarray) -> Tuple[float, bool]:
    """Compute the area of a convex polygon or the area of its convex hull.

    Note: The vertices do not need to be stored in any specific order.

    Args:
        coords: The (x, y) coordinates of the vertices.

    Returns:
        The area of the polygon or the convex hull.
    """
    try:
        hull = ConvexHull(coords)
    except QhullError:
        # Handle error when all points lie on a line
        return 0, True
    else:
        is_convex = len(hull.vertices) == len(coords)
        return hull.volume, is_convex


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


def orient_convex_polygon(vertices: np.ndarray) -> np.ndarray:
    """Returns counterclockwise-oriented vertices for a convex polygon.

    Args:
        vertices: The vertex positions (x, y), shape ``(n, 2)``.

    Returns:
        The ``vertices`` sorted counterclockwise.
    """
    # Sort the vertices by the angle between each vertex and some point in the
    # interior of the polygon. Here we use the mean of the vertices.
    diffs = vertices - vertices.mean(axis=0)
    return vertices[np.argsort(np.arctan2(diffs[:, 1], diffs[:, 0]))]


def convex_polygon_centroid(points: np.ndarray) -> Tuple[float, float]:
    """Calculates the ``(x, y)`` position of the centroid of a convex polygon.

    Args:
        points: An array of vertex coordinates.

    Returns:
        The ``(x, y)`` position of the centroid of the polygon defined by ``points``.
    """
    # Find a Delaunay triangulation of the polygon
    triangles = Delaunay(points).simplices
    # Find the area and centroid of each triangle
    areas = triangle_areas(points, triangles)
    centroids = points[triangles].mean(axis=1)
    # Return the weighted average of the triangle centroids.
    return np.average(centroids, weights=areas, axis=0)


def get_oriented_boundary(
    points: np.ndarray, boundary_edges: np.ndarray
) -> List[np.ndarray]:
    """Returns arrays of boundary vertex indices, ordered counterclockwise.

    Args:
        points: Shape ``(n, 2)``, float array of vertex coordinates.
        boundary_edges: Shape ``(m, 2)`` integer array of boundary edges.

    Returns:
        A list of arrays of boundary vertex indices (ordered counterclockwise).
        The length of the list will be 1 plus the number of holes in the polygon,
        as each hole has a boundary.
    """
    points_list = [tuple(xy) for xy in points]
    edges = MultiLineString([points[edge, :] for edge in boundary_edges])
    polygons = list(polygonize(edges))
    polygon_indices = []
    for p in polygons:
        polygon = orient(p)
        indices = np.array([points_list.index(xy) for xy in polygon.exterior.coords])
        polygon_indices.append(indices[:-1])
    return polygon_indices


def get_supercurrent(
    psi: np.ndarray, gradient: sp.csr_matrix, edges: np.ndarray
) -> np.ndarray:
    """Compute the supercurrent on the edges.

    Args:
        psi: The value of the complex order parameter.
        gradient: The covariant derivative matrix.
        edges: The indices for the edges.

    Returns:
        The supercurrent at each edge.
    """
    return (psi.conjugate()[edges[:, 0]] * (gradient @ psi)).imag
