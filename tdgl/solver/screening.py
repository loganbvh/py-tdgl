import numba
import numpy as np


@numba.njit(fastmath=True, parallel=True)
def get_A_induced_numba(
    J_site: np.ndarray,
    areas: np.ndarray,
    sites: np.ndarray,
    edge_centers: np.ndarray,
) -> np.ndarray:
    """Calculates the induced vector potential on the mesh edges.

    Args:
        J_site: The current density on the sites, shape ``(n, )``
        areas: The mesh site areas, shape ``(n, )``
        sites: The mesh site coordinates, shape ``(n, 2)``
        edge_centers: The coordinates of the edge centers, shape ``(m, 2)``

    Returns:
        The induced vector potential on the mesh edges.
    """
    assert J_site.ndim == 2
    assert J_site.shape[1] == 2
    assert sites.shape == J_site.shape
    assert edge_centers.ndim == 2
    assert edge_centers.shape[1] == 2
    out = np.empty((edge_centers.shape[0], sites.shape[1]), dtype=J_site.dtype)
    for i in numba.prange(edge_centers.shape[0]):
        for k in range(J_site.shape[1]):
            tmp = 0.0
            for j in range(J_site.shape[0]):
                dr = np.sqrt(
                    (edge_centers[i, 0] - sites[j, 0]) ** 2
                    + (edge_centers[i, 1] - sites[j, 1]) ** 2
                )
                tmp += J_site[j, k] * areas[j] / dr
            out[i, k] = tmp
    return out
