from typing import Tuple, Dict, Any, Sequence, Optional, NamedTuple

import h5py
import numpy as np

from ..mesh.mesh import Mesh
from ..enums import Observable
from ..tdgl import get_observable_on_site
from ..util import sum_contributions
from ..matrices import build_gradient


class TDGLData(NamedTuple):
    step: int
    psi: np.ndarray
    mu: np.ndarray
    a: np.ndarray
    supercurrent: np.ndarray
    normal_current: np.ndarray


def get_data_range(h5file: h5py.File) -> Tuple[int, int]:
    keys = np.asarray(list(int(key) for key in h5file["data"]))

    minimum = np.min(keys)
    maximum = np.max(keys)

    return minimum, maximum


def has_voltage_data(h5file: h5py.File) -> bool:
    return "voltage" in h5file["data"]["1"] and "current" in h5file["data"]["1"]


def load_tdgl_data(
    h5file: h5py.File, step: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    step = str(step)

    def get(key, default=None):
        if key in ["step"]:
            return int(step)
        if key in h5file["data"][step]:
            return np.asarray(h5file["data"][step][key])
        return default

    # keys = ["step", "psi", "mu", "a", "supercurrent", "normal_current"]
    return TDGLData(*map(get, TDGLData._fields))


def load_state_data(h5file: h5py.File, step: int) -> Dict[str, Any]:
    return dict(h5file["data"][str(step)].attrs)


def get_edge_observable_data(
    observable: np.ndarray, mesh: Mesh
) -> Tuple[np.ndarray, np.ndarray, Sequence[float]]:
    directions = get_observable_on_site(observable, mesh)
    norm = np.linalg.norm(directions, axis=1)
    directions /= np.maximum(norm, 1e-12)[:, None]
    return norm, directions, [np.min(norm), np.max(norm)]


def get_alpha(h5file: h5py.File) -> Optional[np.ndarray]:
    if "disorder" in h5file:
        return np.asarray(h5file["disorder"]["alpha"])
    return None


def get_plot_data(
    h5file: h5py.File, mesh: Mesh, observable: Observable, frame: int
) -> Tuple[np.ndarray, np.ndarray, Sequence[float]]:
    """
    Get data to plot.
    :param h5file: The data file.
    :param mesh: The mesh used in the simulation.
    :param observable: The observable to return.
    :param frame: The current frame.
    :return: A tuple of the values for the color plot, the directions for the
    quiver plot and the limits for the
    color plot.
    """

    # Get the tdgl fields
    _, psi, mu, a, supercurrent, normal_current = load_tdgl_data(h5file, frame)

    if observable is Observable.COMPLEX_FIELD and psi is not None:
        return np.abs(psi), np.zeros((len(mesh.x), 2)), [0, 1]

    elif observable is Observable.PHASE and psi is not None:
        return np.angle(psi), np.zeros((len(mesh.x), 2)), [-np.pi, np.pi]

    elif observable is Observable.SUPERCURRENT and supercurrent is not None:
        return get_edge_observable_data(supercurrent, mesh)

    elif observable is Observable.NORMAL_CURRENT and normal_current is not None:
        return get_edge_observable_data(normal_current, mesh)

    elif observable is Observable.SCALAR_POTENTIAL and mu is not None:
        return mu, np.zeros((len(mesh.x), 2)), [np.min(mu), np.max(mu)]

    elif observable is Observable.VECTOR_POTENTIAL and a is not None:
        return get_edge_observable_data(
            (a * mesh.edge_mesh.directions).sum(axis=1), mesh
        )

    elif observable is Observable.ALPHA:
        alpha = get_alpha(h5file)

        if alpha is None:
            alpha = np.ones_like(mu)

        return alpha, np.zeros((len(mesh.x), 2)), [np.min(alpha), np.max(alpha)]

    elif observable is Observable.VORTICITY and supercurrent is not None:
        j_sc_site = get_observable_on_site(supercurrent, mesh)
        j_nm_site = get_observable_on_site(normal_current, mesh)
        j_site = j_sc_site + j_nm_site
        gradient = build_gradient(mesh)
        normalized_directions = (
            mesh.edge_mesh.directions
            / np.linalg.norm(mesh.edge_mesh.directions, axis=1)[:, np.newaxis]
        )
        grad_jx = gradient @ j_site[:, 0]
        grad_jy = gradient @ j_site[:, 1]
        djy_dx = grad_jy * normalized_directions[:, 0]
        djx_dy = grad_jx * normalized_directions[:, 1]
        vorticity_on_edges = djy_dx - djx_dy
        vorticity = get_observable_on_site(vorticity_on_edges, mesh, vector=False)
        vmax = max(np.abs(np.max(vorticity)), np.abs(np.min(vorticity)))
        return vorticity, np.zeros((len(mesh.x), 2)), [-vmax, vmax]

    return np.zeros_like(mesh.x), np.zeros((len(mesh.x), 2)), [0, 0]


def get_state_string(h5file: h5py.File, frame: int, max_frame: int) -> str:
    state = load_state_data(h5file, frame)

    state_string = f"Frame {frame} of {max_frame}"
    i = 1
    for key, value in state.items():
        state_string += ", "
        if i % 3 == 0:
            state_string += "\n"
        if type(value) is np.float64:
            state_string += f"{key}: {value:.2e}"
        else:
            state_string += f"{key}: {value}"

        i += 1

    return state_string[:-1]


def find_voltage_points(mesh: Mesh, h5file: h5py.File, frame: int) -> np.ndarray:
    # Get psi on the boundary
    psi_boundary = np.asarray(h5file["data"]["0"]["psi"])[mesh.boundary_indices]

    # Select boundary points where the complex field is small on the first frame
    metal_boundary = mesh.boundary_indices[np.where(np.abs(psi_boundary) < 1e-7)[0]]

    # Get the scalar potential on the boundary
    scalar_metal_boundary = np.asarray(h5file["data"][str(frame)]["mu"])[metal_boundary]

    # Find the max and the min
    minimum = np.argmin(scalar_metal_boundary)
    maximum = np.argmax(scalar_metal_boundary)

    # Return the indices
    return metal_boundary[[minimum, maximum]]


def get_mean_voltage(input_path: str) -> Tuple[np.ndarray, np.ndarray]:

    # Open the file
    with h5py.File(input_path, "r") as h5file:

        min_frame, max_frame = get_data_range(h5file)

        current_arr = []
        voltage_arr = []

        # Check if the old or the new approach should be used
        if not has_voltage_data(h5file):

            # Compute mean voltage from flow in the state
            current = h5file["data"][str(min_frame)].attrs["current"]
            old_flow = h5file["data"][str(min_frame)].attrs["flow"]
            old_time = h5file["data"][str(min_frame)].attrs["time"]
            flow = old_flow
            time = old_time

            for i, frame in enumerate(range(min_frame + 1, max_frame + 1)):
                tmp_current = h5file["data"][str(frame)].attrs["current"]
                tmp_flow = h5file["data"][str(frame)].attrs["flow"]
                tmp_time = h5file["data"][str(frame)].attrs["time"]

                if tmp_current > current:
                    current_arr.append(current)
                    voltage_arr.append((flow - old_flow) / (time - old_time))
                    current = tmp_current
                    old_time = tmp_time
                    old_flow = tmp_flow

                time = tmp_time
                flow = tmp_flow

            # Add last point
            current_arr.append(current)
            voltage_arr.append((flow - old_flow) / (time - old_time))

        else:

            # Compute the mean voltage from the voltage
            for i in range(1, max_frame + 1):
                current_arr = np.concatenate(
                    [current_arr, h5file["data"][str(i)]["current"]]
                )

                voltage_arr = np.concatenate(
                    [voltage_arr, h5file["data"][str(i)]["voltage"]]
                )

            current_arr, voltage_arr, counts = sum_contributions(
                current_arr, voltage_arr
            )
            voltage_arr /= counts

    return np.asarray(current_arr), np.asarray(voltage_arr)


def get_magnetic_field(input_path: str, frame: int) -> float:
    # Open the file
    with h5py.File(input_path, "r") as h5file:
        return h5file["data"][str(frame)].attrs["magnetic field"]
