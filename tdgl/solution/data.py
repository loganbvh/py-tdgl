import dataclasses
from typing import Any, Dict, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ..finite_volume.mesh import Mesh


def get_data_range(h5file: h5py.File) -> Tuple[int, int]:
    """Returns the minimum and maximum solve steps in the file."""
    keys = np.asarray([int(key) for key in h5file["data"]])
    return np.min(keys), np.max(keys)


def load_state_data(h5file: h5py.File, step: int) -> Dict[str, Any]:
    """Returns a dict of state data for the given solve step."""
    return dict(h5file["data"][str(step)].attrs)


def array_safe_equals(a: Any, b: Any) -> bool:
    """Check if a and b are equal, even if they are numpy arrays."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and np.allclose(a, b)
    try:
        return a == b
    except TypeError:
        return NotImplemented


def dataclass_equals(dc1: Any, dc2: Any) -> bool:
    """Check if two dataclasses that may hold numpy arrays are equal."""
    if dc1 is dc2:
        return True
    if dc1.__class__ is not dc2.__class__:
        return NotImplemented
    t1 = dataclasses.astuple(dc1)
    t2 = dataclasses.astuple(dc2)
    return all(array_safe_equals(a1, a2) for a1, a2 in zip(t1, t2))


def get_edge_observable_data(
    observable_on_edges: np.ndarray, mesh: Mesh
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Returns the value of a vector observable living on the edges of the mesh,
    evaluated at sites in the mesh.

    Args:
        observable_on_edges: An array of observable values on the mesh edges.
        mesh: The :class:`tdgl.finite_volume.mesh.Mesh` instance.

    Returns:
        The magnitude and directions of the observable on the sites, and a tuple
        of the (min, max) of the data.
    """
    directions = mesh.get_observable_on_site(observable_on_edges)
    norm = np.linalg.norm(directions, axis=1)
    directions /= np.maximum(norm, 1e-12)[:, np.newaxis]
    return norm, directions, (np.min(norm), np.max(norm))


@dataclasses.dataclass(eq=False)
class TDGLData:
    """A container for raw data from the TDGL solver at a single solve step.

    Args:
        step: The solver iteration.
        psi: The complex order parameter at each site in the mesh.
        mu: The scalar potential at each site in the mesh.
        applied_vector_potential: The applied vector potential at each edge in the mesh.
        induced_vector_potential: The induced vector potential at each edge in the mesh.
        supercurrent: The supercurrent density at each edge in the mesh.
        normal_current: The normal density at each edge in the mesh.
        state: The solver state for the current iteration.
    """

    step: int
    psi: np.ndarray
    mu: np.ndarray
    applied_vector_potential: np.ndarray
    induced_vector_potential: np.ndarray
    supercurrent: np.ndarray
    normal_current: np.ndarray
    state: Dict[str, Any]

    @staticmethod
    def from_hdf5(h5file: h5py.File, step: int) -> "TDGLData":
        """Load a :class:`TDGLData` instance from an output :class:`h5py.File`.

        Args:
            h5file: An open HDF5 output file.
            step: The solver iteration for which to load data.

        Returns:
            A :class:`TDGLData` instance containing data for the requested solve step.
        """
        step = str(step)

        def get(key, default=None):
            if key in ["step"]:
                return int(step)
            if key in ["state"]:
                return load_state_data(h5file, step)
            if key in h5file["data"][step]:
                return np.asarray(h5file["data"][step][key])
            return default

        return TDGLData(
            **{field.name: get(field.name) for field in dataclasses.fields(TDGLData)}
        )

    def to_hdf5(self, h5group: h5py.Group) -> None:
        """Save a :class:`TDGLData` instance to an :class:`h5py.Group`.

        Args:
            h5group: An open :class:`h5py.Group` in which to save the data.
        """
        group = h5group.create_group(str(self.step))
        for key, value in dataclasses.asdict(self).items():
            if key in ["step"]:
                continue
            if key in ["state"]:
                group.attrs.update(value)
            else:
                group[key] = value

    def __eq__(self, other: Any) -> bool:
        return dataclass_equals(self, other)


@dataclasses.dataclass(eq=False)
class DynamicsData:
    """A container for the measured dynamics of a TDGL solution,
    measured at each time step in the simulation.

    Args:
        dt: The solver time step, :math:`\\Delta t^{n}`.
        time: The solver time, a derived attribute which is equal to the cumulative
            sum of the time step.
        voltage: The difference in scalar potential :math:`\\mu` between the model's
            voltage points, :math:`V/V_0`.
        phase_difference: The difference in the phase of the order parameter
            :math:`\\arg(\\psi)` between the model's voltage points.
    """

    dt: np.ndarray
    time: np.ndarray = dataclasses.field(init=False)
    voltage: np.ndarray
    phase_difference: np.ndarray

    def __post_init__(self):
        self.time = np.cumsum(self.dt)

    def time_slice(self, tmin: float = -np.inf, tmax: float = np.inf) -> np.ndarray:
        """Returns the integer indices corresponding to the specified time window.

        Args:
            tmin: The minimum of the time window.
            tmax: The maximum of the time window.

        Returns:
            An array of indices for the time window.
        """
        ts = self.time
        (indices,) = np.where((ts >= tmin) & (ts <= tmax))
        return indices

    def mean_voltage(self, tmin: float = -np.inf, tmax: float = np.inf) -> float:
        """Returns the time-averaged voltage :math:`\\langle V/V_0\\rangle`
        over the specified time interval.

        .. math::

            \\langle V/V_0 \\rangle =
            \\frac{\\sum_n V^{n}/V_0\\cdot\\Delta t^{n}}{\\sum_n\\Delta t^{n}}

        Args:
            tmin: The minimum of the time window over which to average.
            tmax: The maximum of the time window over which to average.

        Returns:
            The time-averaged voltage over the specified time window.
        """
        indices = self.time_slice(tmin, tmax)
        return np.average(self.voltage[indices], weights=self.dt[indices])

    def plot(
        self,
        tmin: float = -np.inf,
        tmax: float = +np.inf,
        grid: bool = True,
        mean_voltage: bool = True,
        labels: bool = True,
        legend: bool = False,
    ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """Plot the voltage and phase difference over the specified time window.

        Args:
            tmin: The minimum of the time window to plot.
            tmax: The maximum of the time window to plot.
            grid: Whether to add grid lines to the plots.
            mean_voltage: Whether to plot a horizontal line at the mean voltage.
            labels: Whether to include axis labels.
            legend: Whether to include a legend.

        Returns:
            matplotlib figure and axes.
        """
        fig, axes = plt.subplots(2, 1, sharex=True)
        ax, bx = axes
        ax.grid(grid)
        bx.grid(grid)
        ts = self.time
        vs = self.voltage
        phases = np.unwrap(self.phase_difference) / np.pi
        indices = self.time_slice(tmin, tmax)
        ax.plot(ts[indices], vs[indices])
        if mean_voltage:
            ax.axhline(
                self.mean_voltage(tmin, tmax),
                label="Mean voltage",
                color="k",
                ls="--",
            )
        bx.plot(ts[indices], phases[indices])
        if labels:
            ax.set_ylabel("Voltage\n$\\Delta\\mu/V_0$")
            bx.set_xlabel("Time, $t/\\tau_0$")
            bx.set_ylabel("Phase difference\n$\\Delta\\theta/\\pi$")
        if legend:
            ax.legend(loc=0)
        return fig, axes

    def plot_dt(
        self,
        tmin: float = -np.inf,
        tmax: float = +np.inf,
        grid: bool = True,
        labels: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the time step :math:`\\Delta t^{n}` vs. time.

        Args:
            tmin: The minimum of the time window to plot.
            tmax: The maximum of the time window to plot.
            grid: Whether to add grid lines to the plots.
            labels: Whether to include axis labels.

        Returns:
            matplotlib figure and axes.
        """
        fig, ax = plt.subplots()
        ax.grid(grid)
        ts = self.time
        indices = self.time_slice(tmin, tmax)
        ax.plot(ts[indices], self.dt[indices])
        if labels:
            ax.set_xlabel("Time, $t/\\tau_0$")
            ax.set_ylabel("Time step, $\\Delta t/\\tau_0$")
        return fig, ax

    @staticmethod
    def from_hdf5(
        h5file: h5py.File,
        step_min: Union[int, None] = None,
        step_max: Union[int, None] = None,
    ) -> "DynamicsData":
        """Load a :class:`DynamicsData` instance from an output :class:`h5py.File`.

        Args:
            h5file: An open HDF5 output file.
            step_min: The minimum solve step to load.
            step_max: The maximum solve step to load.

        Returns:
            A new :class:`DynamicsData` instance.
        """
        dts = []
        voltages = []
        phases = []
        if step_min is None:
            step_min, step_max = get_data_range(h5file)
        for i in range(step_min, step_max + 1):
            grp = h5file[f"data/{i}"]
            if "dt" in grp:
                dts.append(np.asarray(grp["dt"]))
                voltages.append(np.asarray(grp["voltage"]))
                phases.append(np.asarray(grp["phase_difference"]))
        dt = np.concatenate(dts)
        mask = dt > 0
        dt = dt[mask]
        voltage = np.concatenate(voltages)[mask]
        phase = np.concatenate(phases)[mask]
        return DynamicsData(dt, voltage, phase)

    def to_hdf5(self, h5group: h5py.Group, subgroup: str) -> None:
        """Save a :class:`DynamicsData` instance to an :class:`h5py.Group`.

        Args:
            h5group: An open :class:`h5py.Group` in which to save the data.
            subgroup: The name of the subgroup into which to save the data.
        """
        grp = h5group.require_group(str(subgroup))
        for key in ("dt", "voltage", "phase_difference"):
            grp[key] = getattr(self, key)

    def __eq__(self, other: Any) -> bool:
        return dataclass_equals(self, other)
