from typing import Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError


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
    :param xe: The x coordinates for the edges.
    :param ye: The y coordinates for the edges.
    :param elements: The triangular elements in the tesselation.
    :param x_dual: The x coordinates for the dual mesh (Voronoi sites).
    :param y_dual: The y coordinates for the dual mesh (Voronoi sites).
    :param edges: The edges connecting the sites.
    :return: An array of dual edge lengths.
    """

    # Create a dict with keys corresponding to the edges and values
    # corresponding to the triangles
    edge_to_element = {}

    # Iterate over all elements to create the edge_to_element dict
    edge_element_indices = [[0, 1], [1, 2], [2, 0]]
    for i, element in enumerate(elements):

        for idx in edge_element_indices:

            # Hash the array by converting it to a string
            edge = str(np.sort(element[idx]))

            if edge in edge_to_element:
                edge_to_element[edge].append(i)
            else:
                edge_to_element[edge] = [i]

    dual_lengths = np.zeros_like(xe)

    # Iterate over all edges
    for i, edge in enumerate(edges):

        # Get the elements from the dict
        indices = edge_to_element[str(edge)]

        # Handle boundary edges
        if len(indices) == 1:
            dual_lengths[i] = np.sqrt(
                (x_dual[indices[0]] - xe[i]) ** 2 + (y_dual[indices[0]] - ye[i]) ** 2
            )

        # Handle inner edges
        else:
            dual_lengths[i] = np.sqrt(
                (x_dual[indices[0]] - x_dual[indices[1]]) ** 2
                + (y_dual[indices[0]] - y_dual[indices[1]]) ** 2
            )

    return dual_lengths


def get_edges(elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the edges from a list of triangles.
    :param elements: The triangle elements.
    :return: A tuple containing the edges and if they are a boundary edge.
    """

    # Separate the elements into the three edges of the triangles
    edges = np.concatenate(
        [elements[:, (0, 1)], elements[:, (1, 2)], elements[:, (2, 0)]]
    )

    # Sort each edge such that the smallest index is first
    edges = np.sort(edges, axis=1)

    # Remove the duplicates of the edges
    edges, occurrences = np.unique(edges, return_counts=True, axis=0)

    return edges, occurrences == 1


def generate_voronoi_vertices(
    x: np.ndarray, y: np.ndarray, elements: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the vertices of the Voronoi lattice by computing the
    circumcenter of the triangles in the tesselation.
    :param x: The x coordinates of the tesselation.
    :param y: The y coordinates of the tesselation.
    :param elements: The triangular elements in the tesselation.
    :return: A list of vertices.
    """

    # Get the triangle abc
    # Convert to the coordinate system where a is in the origin
    a = np.array([x[elements[:, 0]], y[elements[:, 0]]])
    bp = np.array([x[elements[:, 1]], y[elements[:, 1]]]) - a
    cp = np.array([x[elements[:, 2]], y[elements[:, 2]]]) - a

    # Compute the denominator
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


def get_surrounding_voronoi_polygons(
    elements: np.ndarray, num_sites: int
) -> Dict[int, np.ndarray]:
    """
    Find the polygons surrounding each site.
    :param elements: The triangular elements in the tesselation.
    :param num_sites: The number of sites
    :return: A dict where the keys are the indices for the sites and the values
    are lists of the Voronoi polygon indices.
    """

    # Iterate over all sites and find the triangles that the site belongs to
    # The indices for the triangles are the same as the indices for the
    # Voronoi lattice
    return dict(
        (idx, (elements == idx).any(axis=1).nonzero()[0]) for idx in range(num_sites)
    )


def compute_surrounding_area(
    x: np.ndarray,
    y: np.ndarray,
    x_dual: np.ndarray,
    y_dual: np.ndarray,
    boundary: np.ndarray,
    edges: np.ndarray,
    boundary_edge_indices: np.ndarray,
    polygons: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Compute the areas of the surrounding polygons. Areas of boundary points
    are handled by adding additional points on
    the boundary to make a convex polygon.
    :param x: The x coordinates for the sites.
    :param y: The y coordinates for the sites.
    :param x_dual: The x coordinates for the dual mesh (Voronoi sites).
    :param y_dual: The y coordinates for the dual mesh (Voronoi sites).
    :param boundary: An array containing all boundary points.
    :param edges: The edges of the triangles.
    :param boundary_edge_indices: The edge indices corresponding to the boundary.
    :param polygons: The polygons in Voronoi diagram.
    :return: A list of areas for each site in the lattice.
    """

    boundary_set = set(boundary)
    boundary_edges = edges[boundary_edge_indices]

    areas = np.zeros(len(polygons))

    # Iterate over the sites
    for i, polygon in polygons.items():

        # Get the polygon points
        poly_x = x_dual[polygon]
        poly_y = y_dual[polygon]

        # Handle points not on the boundary
        if i not in boundary_set:
            areas[i], _ = get_convex_polygon_area(poly_x, poly_y)
            continue

        # TODO: First computing a dict where the key is the boundary index
        #  and the value is a list of neighbouring
        #  points would be more effective. Consider changing to that instead.
        connected_boundary_edges = boundary_edges[
            (boundary_edges == i).any(axis=1).nonzero()[0]
        ]
        x_mid = x[connected_boundary_edges].mean(axis=1)
        y_mid = y[connected_boundary_edges].mean(axis=1)
        poly_x = np.concatenate([poly_x, [x[i]], x_mid])
        poly_y = np.concatenate([poly_y, [y[i]], y_mid])

        # Compute the area of the polygon
        areas[i], is_convex = get_convex_polygon_area(poly_x, poly_y)

        # If the polygon is non-convex we need to subtract the area of the
        # concave part, which is the triangle on the boundary.
        if not is_convex:
            concave_area, _ = get_convex_polygon_area(
                np.concatenate([[x[i]], x_mid]), np.concatenate([[y[i]], y_mid])
            )
            areas[i] -= concave_area

    return areas


def get_convex_polygon_area(x: np.ndarray, y: np.ndarray) -> Tuple[float, bool]:
    """
    Compute the area of a convex polygon or the area of its convex hull.
    Note: The vertices do not need to be stored in any specific order.
    :param x: The x coordinates of the vertices.
    :param y: The y coordinates of the vertices.
    :return: The area of the polygon or the convex hull.
    """

    try:
        # Compute the convex hull
        hull = ConvexHull(np.array([x, y]).transpose())

        # Check if the polygon is convex
        is_convex = len(hull.vertices) == len(x)

        # Compute the convex hull and extract the volume / area
        return hull.volume, is_convex

    # Handle error when all points is on a line
    except QhullError:
        return 0, True


def sum_contributions(
    groups: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum all contributions from value over each group.

    Args:
        groups: The groups to get the summed values for. Must be a one
            dimensional vector.
        values: The values for each item. Must be a one dimensional vector.

    Returns:
        A tuple of groups, group values, and counts.
    """

    # Get the unique groups and the corresponding indices
    unique_groups, idx, counts = np.unique(
        groups, return_inverse=True, return_counts=True
    )

    # Sum each unique group
    group_values = np.bincount(idx, weights=values)

    return unique_groups, group_values, counts


def get_supercurrent(
    psi: np.ndarray, gradient: csr_matrix, edges: np.ndarray
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
