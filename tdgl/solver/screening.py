import numba
import numpy as np

try:
    import cupy  # type: ignore
    import cupyx  # type: ignore
except ImportError:
    cupy = None
    cupyx = None


@numba.njit(fastmath=True, parallel=True)
def get_A_induced_numba(
    J_site: np.ndarray,
    site_areas: np.ndarray,
    sites: np.ndarray,
    edge_centers: np.ndarray,
    A_induced: np.ndarray,
) -> None:
    """Calculates the induced vector potential on the mesh edges.

    Args:
        J_site: The current density on the sites, shape ``(n, )``
        site_areas: The mesh site areas, shape ``(n, )``
        sites: The mesh site coordinates, shape ``(n, 2)``
        edge_centers: The coordinates of the edge centers, shape ``(m, 2)``
        A_induced: The output array, shape ``(m, 2)``
    """
    assert J_site.ndim == 2
    assert J_site.shape[1] == 2
    assert sites.shape == J_site.shape
    assert edge_centers.ndim == 2
    assert edge_centers.shape[1] == 2
    for i in numba.prange(edge_centers.shape[0]):
        for k in range(J_site.shape[1]):
            tmp = 0.0
            for j in range(J_site.shape[0]):
                dx = edge_centers[i, 0] - sites[j, 0]
                dy = edge_centers[i, 1] - sites[j, 1]
                dr = np.sqrt(dx * dx + dy * dy)
                tmp += J_site[j, k] * site_areas[j] / dr
            A_induced[i, k] = tmp


get_A_induced_cupy = None

if cupy is not None:

    @cupyx.jit.rawkernel()
    def get_A_induced_cupy(
        J_site: cupy.ndarray,
        site_areas: cupy.ndarray,
        sites: cupy.ndarray,
        edge_centers: cupy.ndarray,
        A_induced: cupy.ndarray,
    ) -> None:
        """Calculates the induced vector potential on the mesh edges.

        Args:
            J_site: The current density on the sites, shape ``(n, )``
            site_areas: The mesh site areas, shape ``(n, )``
            sites: The mesh site coordinates, shape ``(n, 2)``
            edge_centers: The coordinates of the edge centers, shape ``(m, 2)``
            A_induced: The induced vector potential on the mesh edges,
                i.e. the output array, shape ``(m, 2)``.
        """
        i, k = cupyx.jit.grid(2)
        if i < edge_centers.shape[0] and k < J_site.shape[1]:
            tmp = 0.0
            for j in cupyx.jit.range(sites.shape[0]):
                dx = edge_centers[i, 0] - sites[j, 0]
                dy = edge_centers[i, 1] - sites[j, 1]
                dr = cupy.sqrt(dx * dx + dy * dy)
                tmp += J_site[j, k] * site_areas[j] / dr
            A_induced[i, k] = tmp
