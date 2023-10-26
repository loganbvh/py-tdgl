import dataclasses
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from tqdm import tqdm

from ..finite_volume.mesh import Mesh
from ..geometry import path_vectors


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


def get_edge_quantity_data(
    quantity_on_edges: np.ndarray, mesh: Mesh
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Returns the value of a vector quantity living on the edges of the mesh,
    evaluated at sites in the mesh.

    Args:
        quantity_on_edges: An array of quantity values on the mesh edges.
        mesh: The :class:`tdgl.finite_volume.mesh.Mesh` instance.

    Returns:
        The magnitude and directions of the quantity on the sites, and a tuple
        of the (min, max) of the data.
    """
    directions = mesh.get_quantity_on_site(quantity_on_edges)
    norm = np.linalg.norm(directions, axis=1)
    directions /= np.maximum(norm, 1e-12)[:, np.newaxis]
    return norm, directions, (np.min(norm), np.max(norm))


@dataclasses.dataclass(eq=False)
class TDGLData:
    """A container for raw data from the TDGL solver at a single solve step.

    Args:
        step: The solver iteration.
        epsilon: The disorder parameter. :math:`\\epsilon<1` weakens the
            order parameter.
        psi: The complex order parameter at each site in the mesh.
        mu: The scalar potential at each site in the mesh.
        applied_vector_potential: The applied vector potential at each edge in the mesh.
        induced_vector_potential: The induced vector potential at each edge in the mesh.
        supercurrent: The supercurrent density at each edge in the mesh.
        normal_current: The normal density at each edge in the mesh.
        state: The solver state for the current iteration.
    """

    step: int
    epsilon: np.ndarray
    psi: np.ndarray
    mu: np.ndarray
    applied_vector_potential: np.ndarray
    induced_vector_potential: np.ndarray
    supercurrent: np.ndarray
    normal_current: np.ndarray
    state: Dict[str, Any]

    @staticmethod
    def from_hdf5(h5file: Union[h5py.File, h5py.Group], step: int) -> "TDGLData":
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
            if key in h5file:
                dset = h5file[key]
                dset.refresh()
                return np.asarray(dset)
            if key in h5file["data"][step]:
                dset = h5file["data"][step][key]
                dset.refresh()
                return np.array(dset)
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
        mu: The electric potential, :math:`\\mu`.
        theta: The phase of the order parameter, :math:`\\theta=\\arg\\psi`
        screening_iterations: The number of screening iterations performed at each
            time step.
    """

    dt: np.ndarray
    time: np.ndarray = dataclasses.field(init=False)
    mu: Union[np.ndarray, None] = None
    theta: Union[np.ndarray, None] = None
    screening_iterations: Union[np.ndarray, None] = None

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

    def closest_time(self, time: float) -> int:
        """Returns the index of the time step closest to ``time``.

        Args:
            time: The time for which to find the index.

        Returns:
            The index of the time step closest to ``time``.
        """
        return np.argmin(np.abs(self.time - time))

    def voltage(self, i: int = 0, j: int = 1) -> np.ndarray:
        """Returns the voltage, i.e., the electric potential difference
        between probe points ``i`` and ``j``, as a function of time.

        Args:
            i: Index for the first probe point.
            j: Index for the second probe point.

        Returns:
            The voltage :math:`V_{ij}(t)=\\mu_i(t)-\\mu_j(t)`
        """
        if self.mu is None:
            raise ValueError("No voltage data available.")
        if self.mu.shape[0] == 1:
            raise ValueError("The solution has only one probe point.")
        return self.mu[i] - self.mu[j]

    def phase_difference(self, i: int = 0, j: int = 1) -> np.ndarray:
        """Returns the phase difference between probe points ``i`` and ``j``
        as a function of time.

        Args:
            i: Index for the first probe point.
            j: Index for the second probe point.

        Returns:
            The phase difference :math:`\\Delta\\theta_{ij}(t)=\\theta_i(t)-\\theta_j(t)`,
            where :math:`\\theta=\\arg\\psi`.
        """
        if self.theta is None:
            raise ValueError("No phase data available.")
        if self.theta.shape[0] == 1:
            raise ValueError("The solution has only one probe point.")
        return self.theta[i] - self.theta[j]

    def mean_voltage(
        self, i: int = 0, j: int = 1, tmin: float = -np.inf, tmax: float = np.inf
    ) -> float:
        """Returns the time-averaged voltage :math:`\\langle \\Delta\\mu \\rangle`
        over the specified time interval.

        .. math::

            \\langle V_{i,j} \\rangle =
            \\frac{\\sum_n V_{i,j}^{n}\\cdot\\Delta t^{n}}{\\sum_n\\Delta t^{n}}

        Args:
            i: Index for the first probe point.
            j: Index for the second probe point.
            tmin: The minimum of the time window over which to average.
            tmax: The maximum of the time window over which to average.

        Returns:
            The time-averaged voltage over the specified time window.
        """
        if self.mu is None:
            raise ValueError("No voltage data available.")
        indices = self.time_slice(tmin, tmax)
        return np.average(self.voltage(i, j)[indices], weights=self.dt[indices])

    def resample(self, num_points: Union[int, None] = None) -> "DynamicsData":
        """Re-sample the dynamics to a uniform grid using linear interpolation.

        Args:
            num_points: The number of points to interpolate to.

        Returns:
            A new :class:`DynamicsData` instance with the re-sampled data.
        """
        time = self.time
        if num_points is None:
            num_points = len(time)
        ts = np.linspace(time.min(), time.max(), num_points)
        mu = theta = None
        if self.mu is not None:
            mu = np.array([np.interp(ts, time, val) for val in self.mu])
        if self.theta is not None:
            theta = np.array([np.interp(ts, time, val) for val in self.theta])
        return DynamicsData(dt=(ts[1] - ts[0]) * np.ones_like(ts), mu=mu, theta=theta)

    def plot(
        self,
        i: int = 0,
        j: int = 1,
        tmin: float = -np.inf,
        tmax: float = +np.inf,
        grid: bool = True,
        mean_voltage: bool = True,
        labels: bool = True,
        legend: bool = False,
    ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """Plot the voltage and phase difference over the specified time window.

        Args:
            i: Index for the first probe point.
            j: Index for the second probe point.
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
        vs = self.voltage(i, j)
        phases = np.unwrap(self.phase_difference(i, j)) / np.pi
        indices = self.time_slice(tmin, tmax)
        ax.plot(ts[indices], vs[indices])
        if mean_voltage:
            ax.axhline(
                self.mean_voltage(i=i, j=j, tmin=tmin, tmax=tmax),
                label="Mean voltage",
                color="k",
                ls="--",
            )
        bx.plot(ts[indices], phases[indices])
        if labels:
            ax.set_ylabel(f"Voltage\n$\\Delta\\mu_{{{i},{j}}}$ [$V_0$]")
            bx.set_xlabel("Time, $t$ [$\\tau_0$]")
            bx.set_ylabel(f"Phase difference\n$\\Delta\\theta_{{{i},{j}}}/\\pi$")
        if legend:
            ax.legend(loc=0)
        return fig, axes

    def plot_dt(
        self,
        tmin: float = -np.inf,
        tmax: float = +np.inf,
        grid: bool = True,
        labels: bool = True,
        **histogram_kwargs,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """Plots the time step :math:`\\Delta t^{n}` vs. time and a histogram of
        :math:`\\Delta t^{n}`.

        Args:
            tmin: The minimum of the time window to plot.
            tmax: The maximum of the time window to plot.
            grid: Whether to add grid lines to the plots.
            labels: Whether to include axis labels.
            histogram_kwargs: Passed to plt.Axes.hist().

        Returns:
            matplotlib figure and two axes.
        """
        fig, (ax, bx) = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[2, 1]))
        ax.sharey(bx)
        ax.grid(grid)
        bx.grid(grid)
        ts = self.time
        indices = self.time_slice(tmin, tmax)
        ax.plot(ts[indices], self.dt[indices])
        histogram_kwargs = histogram_kwargs.copy()
        histogram_kwargs.setdefault("bins", 101)
        histogram_kwargs.setdefault("density", True)
        histogram_kwargs["orientation"] = "horizontal"
        bx.hist(self.dt[indices], **histogram_kwargs)
        if labels:
            ax.set_xlabel("Time, $t$ [$\\tau_0$]")
            ax.set_ylabel("Time step, $\\Delta t$ [$\\tau_0$]")
            if histogram_kwargs.get("density", False):
                bx.set_xlabel("Density")
            else:
                bx.set_xlabel("Counts per bin")
        fig.tight_layout()
        return fig, (ax, bx)

    @staticmethod
    def from_hdf5(
        h5file: Union[h5py.File, h5py.Group],
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
        iterations = None
        if "theta" in h5file:
            # Load from DynamicsData.to_hdf5()
            dt = np.array(h5file["dt"])
            mu = screening_iterations = None
            theta = np.array(h5file["theta"])
            if "mu" in h5file:
                mu = np.array(h5file["mu"])
            if "screening_iterations" in h5file:
                iterations = np.array(h5file["screening_iterations"])
        else:
            dts = []
            mus = []
            thetas = []
            screening_iterations = []
            if step_min is None:
                step_min, step_max = get_data_range(h5file)
            for i in range(step_min, step_max + 1):
                grp = h5file[f"data/{i}"]
                if "running_state" not in grp:
                    continue
                grp = grp["running_state"]
                dts.append(np.array(grp["dt"]))
                if "mu" in grp:
                    mus.append(np.array(grp["mu"]))
                if "theta" in grp:
                    thetas.append(np.array(grp["theta"]))
                if "screening_iterations" in grp:
                    screening_iterations.append(np.array(grp["screening_iterations"]))
            dt = np.concatenate(dts)
            mask = dt > 0
            dt = dt[mask]
            mu = theta = iterations = None
            if mus:
                mu = np.concatenate(mus, axis=1)[..., mask]
            if thetas:
                theta = np.concatenate(thetas, axis=1)[..., mask]
            if screening_iterations:
                iterations = np.concatenate(screening_iterations)[mask]
        return DynamicsData(
            dt,
            mu=mu,
            theta=theta,
            screening_iterations=iterations,
        )

    def to_hdf5(self, h5group: h5py.Group) -> None:
        """Save a :class:`DynamicsData` instance to an :class:`h5py.Group`.

        Args:
            h5group: An open :class:`h5py.Group` in which to save the data.
        """
        h5group["dt"] = self.dt
        if self.mu is not None:
            h5group["mu"] = self.mu
        if self.theta is not None:
            h5group["theta"] = self.theta
        if self.screening_iterations is not None:
            h5group["screening_iterations"] = self.screening_iterations

    @staticmethod
    def from_solution(
        solution_path: str,
        probe_points: Optional[Sequence[Tuple[float, float]]] = None,
        progress_bar: bool = False,
    ) -> "DynamicsData":
        """Load :class:`DynamicsData` from the saved time steps of a :class:`tdgl.Solution`.

        Args:
            solution_path: Path to the :class:`tdgl.Solution`
            probe_points: The probe coordinates for which to extract dynamics.
                If ``None``, defaults to ``solution.device.probe_points``.
            progress_bar: Show a progress bar while loading data.

        Returns:
            A new :class:`DynamicsData` instance
        """
        from .solution import Solution

        solution = Solution.from_hdf5(solution_path)
        device = solution.device
        mesh = device.mesh
        if probe_points is None:
            probe_points = device.probe_points
        if probe_points is None:
            raise ValueError("No probe points were provided.")
        probe_points = np.asarray(probe_points).squeeze()
        if probe_points.ndim != 2 or probe_points.shape[1] != 2:
            raise ValueError(
                f"Probe points must have shape (n, 2), got {probe_points.shape}."
            )
        if not device.contains_points(probe_points).all():
            raise ValueError("All probe points must lie within the film.")

        xi = device.coherence_length.magnitude
        probe_point_indices = [mesh.closest_site(xy) for xy in probe_points / xi]
        step_min, step_max = solution.data_range

        num_probes = len(probe_points)
        num_steps = step_max - step_min + 1
        times = np.zeros(num_steps)
        mus = np.zeros((num_probes, num_steps))
        thetas = np.zeros((num_probes, num_steps))

        with h5py.File(solution_path, "r") as h5file:
            for i in tqdm(
                range(step_min, step_max + 1),
                desc="Time steps",
                disable=(not progress_bar),
            ):
                grp = h5file[f"data/{i}"]
                times[i] = float(grp.attrs["time"])
                mus[:, i] = np.array(grp["mu"])[probe_point_indices]
                thetas[:, i] = np.angle(np.array(grp["psi"]))[probe_point_indices]

        return DynamicsData(dt=np.diff(times), mu=mus, theta=thetas)

    def __eq__(self, other: Any) -> bool:
        return dataclass_equals(self, other)


def get_current_through_paths(
    solution_path: str,
    paths: Union[np.ndarray, List[np.ndarray]],
    dataset: Optional[str] = None,
    interp_method: Literal["linear", "cubic"] = "linear",
    units: Optional[str] = None,
    with_units: bool = True,
    progress_bar: bool = True,
) -> Tuple[np.ndarray, Union[np.ndarray, List[np.ndarray]]]:
    """Calculates the current through one or more paths for each saved time step.

    Args:
        solution_path: Path to the solution HDF5 file.
        paths: A list of ``(n, 2)`` arrays of ``(x, y)`` coordinates defining
            the paths. A single ``(n, 2)`` array is also allowed.
        dataset: ``None``, ``"supercurrent"``, or ``"normal_current"``.
            ``None`` indicates the total current.
        interp_method: Interpolation method: either "linear" or "cubic".
        units: The current units to return.
        with_units: Whether to return a :class:`pint.Quantity` with units attached.
        progress_bar: Whether to display a progress bar.

    Returns:
        ``(times, currents)``, where ``currents`` is a list of arrays of the
        time-dependent current through each path. If ``paths`` is given as a
        single array, ``currents`` will be returned as a single array.
    """
    from .solution import Solution

    solution = Solution.from_hdf5(solution_path)
    device = solution.device
    tri = device.triangulation
    ureg = device.ureg

    valid_methods = ("linear", "cubic")
    if interp_method not in valid_methods:
        raise ValueError(
            f"Interpolation method must be one of {valid_methods} (got {interp_method})."
        )
    interp_type = {
        "linear": mtri.LinearTriInterpolator,
        "cubic": mtri.CubicTriInterpolator,
    }[interp_method]

    valid_datasets = ("supercurrent", "normal_current", None)
    if dataset not in valid_datasets:
        raise ValueError(
            f"Dataset name must be one of {valid_datasets} (got {dataset})."
        )

    if units is None:
        units = solution.current_units

    length_units = ureg(device.length_units)

    if isinstance(paths, np.ndarray):
        paths = [paths]
    paths = [np.asarray(p) for p in paths]
    edge_positions = []
    edge_lengths = []
    unit_normals = []
    in_device = []
    for path in paths:
        edge_positions.append((path[:-1] + path[1:]) / 2)
        lengths, normals = path_vectors(path)
        edge_lengths.append(lengths)
        unit_normals.append(normals)
        in_device.append(device.contains_points(edge_positions[-1]))

    step_min, step_max = solution.data_range
    times = solution.times
    raw_currents = [np.zeros_like(times) for _ in paths]
    with h5py.File(solution_path, "r") as h5file:
        for i in tqdm(
            range(step_min, step_max + 1), desc="Time steps", disable=(not progress_bar)
        ):
            grp = h5file[f"data/{i}"]
            if dataset is None:
                K = np.array(grp["normal_current"]) + np.array(grp["supercurrent"])
            else:
                K = np.array(grp[dataset])
            K = device.mesh.get_quantity_on_site(K)
            Kx_interp = interp_type(tri, K[:, 0])
            Ky_interp = interp_type(tri, K[:, 1])
            for j, (path, lengths, normals, ix) in enumerate(
                zip(paths, edge_lengths, unit_normals, in_device)
            ):
                Kx_path = Kx_interp(path[:, 0], path[:, 1]).data
                Ky_path = Ky_interp(path[:, 0], path[:, 1]).data
                K_path = np.array([Kx_path, Ky_path]).T
                K_path[~np.isfinite(K_path).all(axis=1)] = 0
                # Evaluate the sheet current at the edge centers
                K_edge = (K_path[:-1] + K_path[1:]) / 2
                K_dot_n = (K_edge * normals).sum(axis=1)
                raw_currents[j][i] = np.trapz((K_dot_n * lengths)[ix])

    currents = []
    for current in raw_currents:
        J = current * (device.K0 * length_units).to(units)
        if not with_units:
            J = J.magnitude
        currents.append(J)

    if len(currents) == 1:
        currents = currents[0]

    return times, currents
