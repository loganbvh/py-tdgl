import logging
import multiprocessing as mp
from functools import partial
from typing import List, Tuple

import joblib
import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull, Delaunay, QhullError
from shapely.geometry import MultiLineString
from shapely.ops import orient, polygonize

logger = logging.getLogger("tdgl.finite_volume")


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
    edges, _ = get_edges(elements)
    return np.linalg.norm(np.diff(points[edges], axis=1), axis=2).squeeze()


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
    dual_lengths = np.zeros(len(edge_centers), dtype=float)
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
    # https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates_2
    # Get the triangle ABC
    # Convert to the coordinate system where A is in the origin
    A = sites[elements[:, 0]]
    B = sites[elements[:, 1]] - A
    C = sites[elements[:, 2]] - A
    # Compute the circumcenter
    D = 2 * B[:, 0] * C[:, 1] - 2 * B[:, 1] * C[:, 0]
    Ux = (C[:, 1] * (B**2).sum(axis=1) - B[:, 1] * (C**2).sum(axis=1)) / D
    Uy = (B[:, 0] * (C**2).sum(axis=1) - C[:, 0] * (B**2).sum(axis=1)) / D
    # Convert back to the initial coordinate system
    return np.array([Ux, Uy]).T + A


def _get_polygon_indices(
    elements: np.ndarray, site_index: int
) -> Tuple[int, np.ndarray]:
    """Helper function for get_voronoi_polygon_indices()."""
    return np.where((elements == site_index).any(axis=1))[0]


def get_voronoi_polygon_indices(
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


def compute_voronoi_polygon_areas(
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
        An array of areas for each site in the lattice, and a list of
        counterclockwise-oriented Voronoi polygon vertices.
    """

    boundary_set = set(boundary)
    boundary_edges = edges[boundary_edge_indices]
    areas = np.zeros(len(polygons), dtype=float)
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
            assert is_convex  # All interior Voronoi cells are convex.
            voronoi_sites.append(orient_convex_polygon(poly))
            continue
        # For points on the boundary, add vertices at the mesh site and the midpoints
        # of the two edges adjacent to the mesh site to complete the Voronoi polygon.
        connected_boundary_edges = boundary_edges[(boundary_edges == site).any(axis=1)]
        # The midpoints of the two edges adjacent to the site
        midpoints = sites[connected_boundary_edges].mean(axis=1)
        # Orient the convex hull of the polygon in a counterclockwise fashion
        coords = orient_convex_polygon(np.concatenate([poly, midpoints], axis=0))
        coords = [tuple(xy) for xy in coords]
        # Insert the central mesh site between the two boundary edge midpoints
        # to ensure the correct ordering of coordinates.
        indices = sorted([coords.index(tuple(mid)) for mid in midpoints])
        if indices[1] == indices[0] + 1:
            # The two boundary edge midpoints are adjacent in the list of coordinates,
            # so insert the mesh site between them.
            coords.insert(indices[1], sites[site])
        else:
            # The boundary edge midpoints are the first and last elements in the
            # list of coordinates, so append the central mesh site to the end.
            if indices[0] != 0:
                # TODO: Decide whether this should be an exception.
                logger.warning(
                    f"Malformed Voronoi cell surrounding boundary site {site}."
                    " Try changing the number of boundary mesh sites using"
                    " Polygon.resample() or Polygon.buffer(eps) where eps"
                    " is 0 or a small positive float."
                )
            coords.append(sites[site])
        poly = np.array(coords)
        areas[site], is_convex = get_convex_polygon_area(poly)
        if not is_convex:
            # If the polygon is non-convex we need to subtract the area of the
            # concave part, which is the triangle formed by the mesh site and
            # the two adjacent boundary edge midpoints.
            triangle_area, is_convex = get_convex_polygon_area(
                np.concatenate([midpoints, [sites[site]]], axis=0)
            )
            assert is_convex  # This is just a triangle, so it must be convex.
            areas[site] -= triangle_area
        voronoi_sites.append(poly)
    return areas, voronoi_sites


def get_convex_polygon_area(coords: np.ndarray) -> Tuple[float, bool]:
    """Compute the area of a convex polygon or the area of its convex hull.

    Note: The vertices do not need to be stored in any specific order.

    Args:
        coords: The (x, y) coordinates of the vertices.

    Returns:
        The area of the polygon or the convex hull, and a bool indicating
        whether the polygon is convex.
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
    # interior of the polygon. Here we use the mean of the vertex positions.
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
