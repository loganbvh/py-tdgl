import numba
import numpy as np


@numba.njit(fastmath=True, parallel=True)
def pairwise_difference(xA: np.ndarray, xB: np.ndarray):
    """Pairwise different between two 1D arrays.

    This is equivalent to ``numpy.subtract.outer(XA, XB)``.

    Args:
        xA: A shape (n,) array
        xB: A shep (m,) array

    Returns:
        A shape (n, m) array of pairwise differences.
    """
    out = np.empty((xA.shape[0], xB.shape[0]), dtype=xA.dtype)
    for i in numba.prange(xA.shape[0]):
        for j in range(xB.shape[0]):
            out[i, j] = xA[i] - xB[j]
    return out


@numba.njit(fastmath=True, parallel=True)
def sqeuclidean_distance_2d(XA: np.ndarray, XB: np.ndarray):
    """Squared Euclidean pointwise distance between two 2D arrays."""
    out = np.empty((XA.shape[0], XB.shape[0]), dtype=XA.dtype)
    for i in numba.prange(XA.shape[0]):
        for j in range(XB.shape[0]):
            out[i, j] = (XA[i, 0] - XB[j, 0]) ** 2 + (XA[i, 1] - XB[j, 1]) ** 2
    return out


@numba.njit(fastmath=True, parallel=True)
def sqeuclidean_distance_3d(XA: np.ndarray, XB: np.ndarray):
    """Squared Euclidean pointwise distance between two 3D arrays."""
    out = np.empty((XA.shape[0], XB.shape[0]), dtype=XA.dtype)
    for i in numba.prange(XA.shape[0]):
        for j in range(XB.shape[0]):
            out[i, j] = (
                (XA[i, 0] - XB[j, 0]) ** 2
                + (XA[i, 1] - XB[j, 1]) ** 2
                + (XA[i, 2] - XB[j, 2]) ** 2
            )
    return out


@numba.njit(fastmath=True, parallel=True)
def euclidean_distance_2d(XA: np.ndarray, XB: np.ndarray):
    """Euclidean pointwise distance between two 2D arrays."""
    out = np.empty((XA.shape[0], XB.shape[0]), dtype=XA.dtype)
    for i in numba.prange(XA.shape[0]):
        for j in range(XB.shape[0]):
            out[i, j] = np.sqrt((XA[i, 0] - XB[j, 0]) ** 2 + (XA[i, 1] - XB[j, 1]) ** 2)
    return out


@numba.njit(fastmath=True, parallel=True)
def euclidean_distance_3d(XA: np.ndarray, XB: np.ndarray):
    """Euclidean pointwise distance between two 3D arrays."""
    out = np.empty((XA.shape[0], XB.shape[0]), dtype=XA.dtype)
    for i in numba.prange(XA.shape[0]):
        for j in range(XB.shape[0]):
            out[i, j] = np.sqrt(
                (XA[i, 0] - XB[j, 0]) ** 2
                + (XA[i, 1] - XB[j, 1]) ** 2
                + (XA[i, 2] - XB[j, 2]) ** 2
            )
    return out


def cdist(XA: np.ndarray, XB: np.ndarray, metric: str = "euclidean"):
    """Pointwise distance between observations in 2D or 3D space.

    This function provides a subset of the functionality of
    ``scipy.spatial.distance.cdist``. The output array has the same dtype
    as the first input, XA.

    Args:
        XA: An (mA, n) array of observations, where n is 2 or 3.
        XB: An (mB, n) array of observations, where n is 2 or 3.
        metric: Either 'euclidean' or 'sqeuclidean'.

    Returns:
        An (mA, mB) matrix of pointwise distances, where ``out[i, j] = dist(XA[i], XB[j])``.
    """
    metrics = ("euclidean", "sqeuclidean")
    if metric not in metrics:
        raise ValueError(f"Metric must be one of {metrics!r}, got {metric!r}.")
    if XA.shape[1] != XB.shape[1]:
        raise ValueError(
            f"XA.shape[1] ({XA.shape[1]}) must be equal to XB.shape[1] ({XB.shape[1]})."
        )
    if XA.shape[1] == 2:
        if metric == "euclidean":
            return euclidean_distance_2d(XA, XB)
        return sqeuclidean_distance_2d(XA, XB)
    elif XA.shape[1] == 3:
        if metric == "euclidean":
            return euclidean_distance_3d(XA, XB)
        return sqeuclidean_distance_3d(XA, XB)
    raise ValueError(f"Excpected shape (n, 2) arrays, got {XA.shape} and {XB.shape}.")
