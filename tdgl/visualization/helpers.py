from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ..enums import Observable
from ..finite_volume.matrices import build_gradient
from ..finite_volume.mesh import Mesh
from ..finite_volume.util import sum_contributions


class TDGLData(NamedTuple):
    step: int
    psi: np.ndarray
    mu: np.ndarray
    applied_vector_potential: np.ndarray
    induced_vector_potential: np.ndarray
    supercurrent: np.ndarray
    normal_current: np.ndarray


@dataclass
class DynamicsData:
    dt: np.ndarray
    time: np.ndarray = field(init=False)
    current: np.ndarray
    voltage: np.ndarray

    def __post_init__(self):
        self.time = np.cumsum(self.dt)

    def mean_voltage(self, indices: Union[Sequence[int], slice, None] = None) -> float:
        if indices is None:
            indices = slice(None)
        return np.average(self.voltage[indices], weights=self.dt[indices])

    def plot(
        self,
        t_min: float = -np.inf,
        t_max: float = +np.inf,
        ax: Union[plt.Axes, None] = None,
        grid: bool = True,
        mean: bool = True,
        labels: Union[bool, None] = True,
        legend: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()
        ts = self.time
        vs = self.voltage
        (indices,) = np.where((ts > t_min) & (ts < t_max))
        ax.plot(ts[indices], vs[indices])
        if mean:
            ax.axhline(
                self.mean_voltage(indices),
                label="Mean voltage",
                color="k",
                ls="--",
            )
        if labels:
            ax.set_xlabel("Time, $t/\\tau$")
            ax.set_ylabel("Voltage, $V/V_0$")
        if legend:
            ax.legend(loc=0)
        if grid is not None:
            ax.grid(grid)
        return ax


def get_data_range(h5file: h5py.File) -> Tuple[int, int]:
    keys = np.asarray([int(key) for key in h5file["data"]])
    return np.min(keys), np.max(keys)


def has_voltage_data(h5file: h5py.File) -> bool:
    return "voltage" in h5file["data"]["1"] and "total_current" in h5file["data"]["1"]


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

    return TDGLData(*map(get, TDGLData._fields))


def load_running_state_data(
    h5file: h5py.File, step_min: Optional[int] = None, step_max: Optional[int] = None
):
    dts = []
    currents = []
    voltages = []
    if step_min is None:
        step_min, step_max = get_data_range(h5file)
    for i in range(step_min, step_max + 1):
        grp = h5file[f"data/{i}"]
        if "dt" in grp:
            dts.append(np.asarray(grp["dt"]))
            currents.append(np.asarray(grp["total_current"]))
            voltages.append(np.asarray(grp["voltage"]))
    dt = np.concatenate(dts)
    mask = dt > 0
    dt = dt[mask]
    current = np.concatenate(currents)[mask]
    voltage = np.concatenate(voltages)[mask]
    return DynamicsData(dt, current, voltage)


def load_state_data(h5file: h5py.File, step: int) -> Dict[str, Any]:
    return dict(h5file["data"][str(step)].attrs)


def get_edge_observable_data(
    observable: np.ndarray, mesh: Mesh
) -> Tuple[np.ndarray, np.ndarray, Sequence[float]]:
    directions = mesh.get_observable_on_site(observable)
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
    tdgl_data = load_tdgl_data(h5file, frame)
    _, psi, mu, a_applied, a_induced, supercurrent, normal_current = tdgl_data

    if observable is Observable.COMPLEX_FIELD and psi is not None:
        return np.abs(psi), np.zeros((len(mesh.x), 2)), [0, 1]

    elif observable is Observable.PHASE and psi is not None:
        return np.angle(psi) / np.pi, np.zeros((len(mesh.x), 2)), [-1, 1]

    elif observable is Observable.SUPERCURRENT and supercurrent is not None:
        return get_edge_observable_data(supercurrent, mesh)

    elif observable is Observable.NORMAL_CURRENT and normal_current is not None:
        return get_edge_observable_data(normal_current, mesh)

    elif observable is Observable.SCALAR_POTENTIAL and mu is not None:
        mu = mu - np.nanmin(mu)
        return mu, np.zeros((len(mesh.x), 2)), [np.min(mu), np.max(mu)]

    elif observable is Observable.APPLIED_VECTOR_POTENTIAL and a_applied is not None:
        return get_edge_observable_data(
            (a_applied * mesh.edge_mesh.directions).sum(axis=1), mesh
        )

    elif observable is Observable.INDUCED_VECTOR_POTENTIAL and a_induced is not None:
        return get_edge_observable_data(
            (a_induced * mesh.edge_mesh.directions).sum(axis=1), mesh
        )

    elif (
        observable is Observable.TOTAL_VECTOR_POTENTIAL
        and a_applied is not None
        and a_induced is not None
    ):
        return get_edge_observable_data(
            ((a_applied + a_induced) * mesh.edge_mesh.directions).sum(axis=1), mesh
        )

    elif observable is Observable.ALPHA:
        alpha = get_alpha(h5file)

        if alpha is None:
            alpha = np.ones_like(mu)

        return alpha, np.zeros((len(mesh.x), 2)), [np.min(alpha), np.max(alpha)]

    elif (
        observable is Observable.VORTICITY
        and supercurrent is not None
        and normal_current is not None
    ):
        j_sc_site = mesh.get_observable_on_site(supercurrent)
        j_nm_site = mesh.get_observable_on_site(normal_current)
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
        vorticity = mesh.get_observable_on_site(vorticity_on_edges, vector=False)
        vmax = max(np.abs(np.max(vorticity)), np.abs(np.min(vorticity)))
        return vorticity, np.zeros((len(mesh.x), 2)), [-vmax, vmax]

    return np.zeros_like(mesh.x), np.zeros((len(mesh.x), 2)), [0, 0]


def get_state_string(h5file: h5py.File, frame: int, max_frame: int) -> str:
    state = load_state_data(h5file, frame)

    state_string = f"Frame {frame} of {max_frame}"
    i = 1
    for key, value in state.items():
        if key == "timestamp":
            continue
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
    with h5py.File(input_path, "r", libver="latest") as h5file:

        min_frame, max_frame = get_data_range(h5file)

        current_arr = []
        voltage_arr = []

        # Check if the old or the new approach should be used
        if not has_voltage_data(h5file):

            # Compute mean voltage from flow in the state
            current = h5file["data"][str(min_frame)].attrs["total_current"]
            old_flow = h5file["data"][str(min_frame)].attrs["flow"]
            old_time = h5file["data"][str(min_frame)].attrs["time"]
            flow = old_flow
            time = old_time

            for i, frame in enumerate(range(min_frame + 1, max_frame + 1)):
                tmp_current = h5file["data"][str(frame)].attrs["total_current"]
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
                    [current_arr, h5file["data"][str(i)]["total_current"]]
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
    with h5py.File(input_path, "r", libver="latest") as h5file:
        return h5file["data"][str(frame)].attrs["magnetic field"]


def auto_grid(
    num_plots: int,
    max_cols: int = 3,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Creates a grid of at least ``num_plots`` subplots
    with at most ``max_cols`` columns.
    Additional keyword arguments are passed to plt.subplots().
    Args:
        num_plots: Total number of plots that will be populated.
        max_cols: Maximum number of columns in the grid.
    Returns:
        matplotlib figure and axes
    """
    ncols = min(max_cols, num_plots)
    nrows = int(np.ceil(num_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    if not isinstance(axes, (list, np.ndarray)):
        axes = np.array([axes])
    return fig, axes
