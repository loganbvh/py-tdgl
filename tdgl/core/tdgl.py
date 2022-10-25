import numpy as np
from scipy.sparse import csr_matrix

from .mesh.mesh import Mesh


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


def get_observable_on_site(
    observable_on_edge: np.ndarray, mesh: Mesh, vector: bool = True
) -> np.ndarray:
    """Compute the observable on site.

    Args:
        observable_on_edge: Observable on the edges.
        mesh: The corresponding mesh.

    Returns:
        The observable vector at each site.
    """
    # Normalize the edge direction
    directions = mesh.edge_mesh.directions
    normalized_directions = (
        directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    )
    if vector:
        flux_x = observable_on_edge * normalized_directions[:, 0]
        flux_y = observable_on_edge * normalized_directions[:, 1]
    else:
        flux_x = flux_y = observable_on_edge
    # Sum x and y components for every edge connecting to the vertex
    vertices = np.concatenate([mesh.edge_mesh.edges[:, 0], mesh.edge_mesh.edges[:, 1]])
    x_values = np.concatenate([flux_x, flux_x])
    y_values = np.concatenate([flux_y, flux_y])

    counts = np.bincount(vertices)
    x_group_values = np.bincount(vertices, weights=x_values) / counts
    y_group_values = np.bincount(vertices, weights=y_values) / counts

    vector_val = np.array([x_group_values, y_group_values]).T / 2
    if vector:
        return vector_val
    return vector_val[:, 0]
