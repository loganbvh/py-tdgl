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
    xe: np.ndarray,
    ye: np.ndarray,
    elements: np.ndarray,
    x_dual: np.ndarray,
    y_dual: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """
    Compute the lengths of the dual edges.

    Args:
        xe: The x coordinates for the edges.
        ye: The y coordinates for the edges.
        elements: The triangular elements in the tesselation.
        x_dual: The x coordinates for the dual mesh (Voronoi sites).
        y_dual: The y coordinates for the dual mesh (Voronoi sites).
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
    dual_lengths = np.zeros_like(xe)
    for i, edge in enumerate(edges):
        indices = edge_to_element[tuple(edge)]
        if len(indices) == 1:  # Boundary edges
            dual_lengths[i] = np.sqrt(
                (x_dual[indices[0]] - xe[i]) ** 2 + (y_dual[indices[0]] - ye[i]) ** 2
            )
        else:  # Inner edges
            dual_lengths[i] = np.sqrt(
                (x_dual[indices[0]] - x_dual[indices[1]]) ** 2
                + (y_dual[indices[0]] - y_dual[indices[1]]) ** 2
            )
    return dual_lengths


def generate_voronoi_vertices(
    x: np.ndarray, y: np.ndarray, elements: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the vertices of the Voronoi lattice by computing the
    circumcenter of the triangles in the tesselation.

    Args:
        x: The x coordinates of the tesselation.
        y: The y coordinates of the tesselation.
        elements: The triangular elements in the tesselation.

    Returns:
        The x and y coordinates of the Voronoi vertices, as arrays.
    """

    # Get the triangle abc
    # Convert to the coordinate system where a is in the origin
    a = np.array([x[elements[:, 0]], y[elements[:, 0]]])
    bp = np.array([x[elements[:, 1]], y[elements[:, 1]]]) - a
    cp = np.array([x[elements[:, 2]], y[elements[:, 2]]]) - a
    denominator = 2 * (bp[0, :] * cp[1, :] - bp[1, :] * cp[0, :])
    # Compute the circumcenter
    xcp = (
        cp[1, :] * (bp**2).sum(axis=0) - bp[1, :] * (cp**2).sum(axis=0)
    ) / denominator
    ycp = (
        bp[0, :] * (cp**2).sum(axis=0) - cp[0, :] * (bp**2).sum(axis=0)
    ) / denominator
    # Convert back to the initial coordinate system
    return xcp + a[0, :], ycp + a[1, :]


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
    x: np.ndarray,
    y: np.ndarray,
    x_dual: np.ndarray,
    y_dual: np.ndarray,
    boundary: np.ndarray,
    edges: np.ndarray,
    boundary_edge_indices: np.ndarray,
    polygons: List[np.ndarray],
) -> np.ndarray:
    """Compute the areas of the surrounding polygons.

    Areas of boundary points are handled by adding additional points
    on the boundary to make a convex polygon.

    Args:
        x: The x coordinates for the sites.
        y: The y coordinates for the sites.
        x_dual: The x coordinates for the dual mesh (Voronoi sites).
        y_dual: The y coordinates for the dual mesh (Voronoi sites).
        boundary: An array containing all boundary points.
        edges: The edges of the triangles.
        boundary_edge_indices: The edge indices corresponding to the boundary.
        polygons: The polygons in Voronoi diagram.

    Returns:
        An array of areas for each site in the lattice.
    """

    boundary_set = set(boundary)
    boundary_edges = edges[boundary_edge_indices]
    areas = np.zeros(len(polygons))

    for site, polygon in enumerate(polygons):
        # Get the polygon points
        poly_x = x_dual[polygon]
        poly_y = y_dual[polygon]
        # Handle points not on the boundary
        if site not in boundary_set:
            areas[site], is_convex = get_convex_polygon_area(poly_x, poly_y)
            assert is_convex
            continue
        # TODO: First computing a dict where the key is the boundary index
        #  and the value is a list of neighbouring
        #  points would be more effective. Consider changing to that instead.
        connected_boundary_edges = boundary_edges[(boundary_edges == site).any(axis=1)]
        x_mid = x[connected_boundary_edges].mean(axis=1)
        y_mid = y[connected_boundary_edges].mean(axis=1)
        poly_x = np.concatenate([poly_x, [x[site]], x_mid])
        poly_y = np.concatenate([poly_y, [y[site]], y_mid])
        areas[site], is_convex = get_convex_polygon_area(poly_x, poly_y)
        # If the polygon is non-convex we need to subtract the area of the
        # concave part, which is the triangle on the boundary.
        if not is_convex:
            concave_area, is_convex = get_convex_polygon_area(
                np.concatenate([[x[site]], x_mid]), np.concatenate([[y[site]], y_mid])
            )
            assert is_convex
            areas[site] -= concave_area
    return areas


def get_convex_polygon_area(x: np.ndarray, y: np.ndarray) -> Tuple[float, bool]:
    """Compute the area of a convex polygon or the area of its convex hull.

    Note: The vertices do not need to be stored in any specific order.

    Args:
        x: The x coordinates of the vertices.
        y: The y coordinates of the vertices.

    Returns:
        The area of the polygon or the convex hull.
    """
    try:
        hull = ConvexHull(np.array([x, y]).T)
    except QhullError:
        # Handle error when all points lie on a line
        return 0, True
    else:
        # is_convex = len(hull.vertices) == len(x)
        is_convex = len(hull.coplanar) == 0
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


def orient_convex_polygon_vertices(
    vertices: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Returns counterclockwise-oriented vertex indices for a convex polygon.

    Args:
        vertices: The vertex positions (x, y), shape ``(n, 2)``.
        indices: Indices into ``vertices`` that define a convex polygon,
            but for which the ordering is unknown.

    Returns:
        ``indices``, sorted so as to orient ``vertices`` counterclockwise.
    """
    # Sort the vertices by the angle between each vertex and some point in the
    # interior of the polygon. Here we use the mean of the vertices.
    vertices = vertices[indices]
    r0 = vertices.mean(axis=0)
    diffs = vertices - r0
    return indices[np.argsort(np.arctan2(diffs[:, 1], diffs[:, 0]))]


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
