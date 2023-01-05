import dataclasses
import logging
import operator
import os
import pickle
import shutil
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union

import cloudpickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pint
from scipy import interpolate
from scipy.spatial import distance

from ..about import version_dict
from ..device.device import Device
from ..device.polygon import Polygon
from ..em import biot_savart_2d, convert_field
from ..finite_volume.operators import build_gradient
from ..fluxoid import Fluxoid
from ..geometry import path_vectors
from ..parameter import Parameter
from ..solver.runner import SolverOptions
from .data import DynamicsData, TDGLData, get_data_range, get_edge_quantity_data

logger = logging.getLogger(__name__)


class BiotSavartField(NamedTuple):
    """The magnetic field due to a current distribution, with the field due to the
    supercurrent and normal current labeled separately.

    Args:
        supercurrent: An array of fields due to the supercurrent.
        normal_current: An array of fields due to the normal current.
    """

    supercurrent: np.ndarray
    normal_current: np.ndarray


class BoundaryPhases(NamedTuple):
    """A container for the phase of the order parameter along a polygon boundary.

    Args:
        indices: The mesh vertex indices of the boundary.
        phase: The phase of the order parameter at each vertex on the boundary.
    """

    indices: np.ndarray
    phases: np.ndarray


class Solution:
    """A container for the results of a TDGL simulation.

    Args:
        device: The :class:`tdgl.Device` that was solved
        options: A :class:`tdgl.SolverOptions` instance.
        path: Path to the HDF5 file containing the raw output data.
        applied_vector_potential: The ``Parameter`` defining the applied vector potential.
        terminal_currents: A dict of ``{terminal_name: current}`` or a callable with signature
            ``func(time) -> {terminal_name: current}``, where ``current`` is a float
            in units of ``current_units``.
        disorder_epsilon: The disorder parameter :math:`\\epsilon`. If
            :math:`\\epsilon(\\mathbf{r}) < 1` weakens the order parameter at position
            :math:`\\mathbf{r}`, which can be used to model inhomogeneity.
        total_seconds: The total wall time in seconds.
    """

    def __init__(
        self,
        *,
        device: Device,
        options: SolverOptions,
        path: os.PathLike,
        applied_vector_potential: Parameter,
        terminal_currents: Union[Dict[str, float], Callable],
        disorder_epsilon: Union[float, Callable],
        total_seconds: float,
        _solve_step: int = -1,
    ):
        self.device = device.copy()
        self.device.mesh = device.mesh
        self.options = options
        self.path = path
        self.applied_vector_potential = applied_vector_potential
        self.terminal_currents = terminal_currents
        self.disorder_epsilon = disorder_epsilon

        self.data_range: Union[Tuple[int, int], None] = None
        """A tuple of ``(min_step, max_step)``."""
        self.supercurrent_density: Union[np.ndarray, None] = None
        """Sheet supercurrent density, :math:`\\mathbf{K}_s`"""
        self.normal_current_density: Union[np.ndarray, None] = None
        """Sheet normal density, :math:`\\mathbf{K}_n`"""
        self._vorticity: Union[np.ndarray, None] = None

        # Make field_units and current_units "read-only" attributes.
        # The should never be changed after instantiation.
        self._field_units = str(self.options.field_units)
        self._current_units = str(self.options.current_units)
        self._time_created = datetime.now()
        self.total_seconds = total_seconds

        self.tdgl_data: Union[TDGLData, None] = None
        """A container for the raw TDGL data (in dimensionless units)."""
        self.dynamics: Union[DynamicsData, None] = None
        """A container for the time dynamics of the solution (in dimensionless units)."""
        self._solve_step = _solve_step
        self.load_tdgl_data(self._solve_step)
        self._version_info = version_dict()

    @property
    def saved_on_disk(self) -> bool:
        """Returns ``True`` if the underlying HDF5 file exists on disk."""
        return os.path.exists(self.path)

    @property
    def solve_step(self) -> int:
        """The solver iteration corresponding to the current
        :class:`tdgl.solution.data.TDGLData`.

        Setting ``solve_step`` automatically loads data for the specitied step.
        """
        return self._solve_step

    @solve_step.setter
    def solve_step(self, step: int) -> None:
        self.load_tdgl_data(solve_step=step)

    @property
    def times(self) -> Union[np.ndarray, None]:
        """The time associated with each solve step."""
        if self.dynamics is None:
            return None
        times = self.dynamics.time
        step = self.options.save_every
        saved_times = times[::step]
        if saved_times[-1] == times[-1]:
            return saved_times.copy()
        # Append the final time step in the simulation, which is always saved.
        return np.concatenate([saved_times, times[-1:]])

    def closest_solve_step(self, time: float) -> int:
        """Returns the index of the saved step closest in time to ``time``.

        Args:
            time: The time for which to find the closest index.

        Returns:
            The index of the saved solve step whose time is closest to ``time``
        """
        return np.argmin(np.abs(self.times - time))

    def load_tdgl_data(
        self, solve_step: int = -1, h5file: Union[h5py.File, None] = None
    ) -> None:
        """Loads the TDGL results from file for a given solve step.

        Args:
            solve_step: The step index for which to load data.
                Defaults to -1, i.e. the final step.
        """
        if h5file is None:
            read_context = h5py.File(self.path, "r", libver="latest")
        else:
            read_context = nullcontext(h5file)
        with read_context as f:
            self.data_range = step_min, step_max = get_data_range(f)
            if solve_step == 0:
                step = step_min
            elif solve_step < 0:
                step = step_max + 1 + solve_step
            else:
                step = solve_step
            self.tdgl_data = TDGLData.from_hdf5(f, step)
            self.dynamics = DynamicsData.from_hdf5(f, *self.data_range)
        mesh = self.device.mesh
        self._solve_step = step
        supercurrent, sc_direc, _ = get_edge_quantity_data(
            self.tdgl_data.supercurrent, mesh
        )
        normal_current, nc_direc, _ = get_edge_quantity_data(
            self.tdgl_data.normal_current, mesh
        )
        K0 = self.device.K0.to(f"{self.current_units} / {self.device.length_units}")
        # Current density, evaluated on the mesh edges.
        self.supercurrent_density = K0 * supercurrent[:, np.newaxis] * sc_direc
        self.normal_current_density = K0 * normal_current[:, np.newaxis] * nc_direc
        self._vorticity = None

    def _compute_vorticity(self) -> None:
        device = self.device
        mesh = device.mesh
        # Calculate the vorticity, evaluated on mesh sites.
        # The vorticity is the curl of the current density.
        j_sc_site = mesh.get_quantity_on_site(self.tdgl_data.supercurrent)
        j_nm_site = mesh.get_quantity_on_site(self.tdgl_data.normal_current)
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
        vorticity = mesh.get_quantity_on_site(vorticity_on_edges, vector=False)
        scale = (device.K0 / device.coherence_length).to(
            f"{self.current_units} / {self.device.length_units}**2"
        )
        self._vorticity = vorticity * scale

    @property
    def vorticity(self) -> Union[np.ndarray, None]:
        """The current vorticity,
        :math:`\\omega=(\\nabla\\times\\mathbf{K})\\cdot\\hat{\\mathbf{z}}`
        """
        if self.supercurrent_density is None:
            return None
        if self._vorticity is None:
            self._compute_vorticity()
        return self._vorticity

    @property
    def current_density(self) -> np.ndarray:
        """The total sheet current density,
        :math:`\\mathbf{K}=\\mathbf{K}_s+\\mathbf{K}_n`.
        """
        if self.supercurrent_density is None:
            return None
        return self.supercurrent_density + self.normal_current_density

    @property
    def field_units(self) -> str:
        """The units in which magnetic fields are specified."""
        return self._field_units

    @property
    def current_units(self) -> str:
        """The units in which currents are specified."""
        return self._current_units

    @property
    def time_created(self) -> datetime:
        """The time at which the solution was originally created."""
        return self._time_created

    @property
    def version_info(self) -> Dict[str, str]:
        """A dictionary of dependency versions."""
        return self._version_info

    def grid_current_density(
        self,
        *,
        dataset: Union[str, None] = None,
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
        method: str = "linear",
        units: Union[str, None] = None,
        with_units: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolates the sheet current density to a rectangular grid.

        Keyword arguments are passed to :func:`scipy.interpolate.griddata`.

        .. seealso::

            :meth:`tdgl.Solution.interp_current_density`

        Args:
            dataset: The dataset to interpolate. One of ``"supercurrent"``,
                ``"normal_current"``, or ``None``. If ``None``, then the total
                sheet current density is used.
            grid_shape: Shape of the desired rectangular grid. If a single integer
                N is given, then the grid will be square, shape = (N, N).
            method: Interpolation method to use (see :func:`scipy.interpolate.griddata`).
            units: The desired units for the current density. Defaults to
                ``self.current_units / self.device.length_units``.
            with_units: Whether to return a :class:`pint.Quantity` array
                with units attached.

        Returns:
            x grid, y grid, interpolated current density
        """
        if dataset is None:
            J = self.current_density
        elif dataset == "supercurrent":
            J = self.supercurrent_density
        elif dataset == "normal_current":
            J = self.normal_current_density
        else:
            raise ValueError(f"Unexpected dataset: {dataset}.")

        units = units or f"{self.current_units} / {self.device.length_units}"
        J = J.to(units)
        if isinstance(grid_shape, int):
            grid_shape = (grid_shape, grid_shape)
        points = self.device.points
        x = points[:, 0]
        y = points[:, 1]
        xgrid, ygrid = np.meshgrid(
            np.linspace(x.min(), x.max(), grid_shape[1]),
            np.linspace(y.min(), y.max(), grid_shape[0]),
        )
        Jx = interpolate.griddata(
            points, J[:, 0].magnitude, (xgrid, ygrid), method=method, **kwargs
        ).ravel()
        Jy = interpolate.griddata(
            points, J[:, 1].magnitude, (xgrid, ygrid), method=method, **kwargs
        ).ravel()
        xy = np.array([xgrid.ravel(), ygrid.ravel()]).T
        hole_mask = np.logical_or.reduce(
            [hole.contains_points(xy) for hole in self.device.holes]
        )
        Jx[hole_mask] = 0
        Jy[hole_mask] = 0
        Jgrid = np.array([Jx.reshape(grid_shape), Jy.reshape(grid_shape)])
        if with_units:
            length_units = self.device.ureg(self.device.length_units)
            xgrid = xgrid * length_units
            ygrid = ygrid * length_units
            Jgrid = (Jgrid * J.units).to(units)
        return xgrid, ygrid, Jgrid

    def interp_current_density(
        self,
        positions: np.ndarray,
        *,
        dataset: Union[str, None] = None,
        method: str = "linear",
        units: Union[str, None] = None,
        with_units: bool = False,
    ) -> np.ndarray:
        """Interpolates the sheet current density at unstructured coordinates.

        .. seealso::

            :meth:`tdgl.Solution.grid_current_density`

        Args:
            positions: Shape ``(m, 2)`` array of x, y coordinates at which to evaluate
                the current density.
            dataset: The dataset to interpolate. One of ``"supercurrent"``,
                ``"normal_current"``, or ``None``. If ``None``, then the total
                sheet current density is used.
            method: Interpolation method to use, ``"nearest"``, ``"linear"``,
                or ``"cubic"``.
            units: The desired units for the current density. Defaults to
                ``self.current_units / self.device.length_units``.
            with_units: Whether to return a :class:`pint.Quantity` array
                with units attached.

        Returns:
            The interpolated current density as an array of floats
            or a :class:`pint.Quantity` array.
        """
        valid_methods = ("nearest", "linear", "cubic")
        if method not in valid_methods:
            raise ValueError(
                f"Interpolation method must be one of {valid_methods} (got {method})."
            )
        if method == "nearest":
            interpolator = interpolate.NearestNDInterpolator
            interp_kwargs = dict()
        elif method == "linear":
            interpolator = interpolate.LinearNDInterpolator
            interp_kwargs = dict(fill_value=0)
        else:  # "cubic"
            interpolator = interpolate.CloughTocher2DInterpolator
            interp_kwargs = dict(fill_value=0)

        if dataset is None:
            J = self.current_density
        elif dataset == "supercurrent":
            J = self.supercurrent_density
        elif dataset == "normal_current":
            J = self.normal_current_density
        else:
            raise ValueError(f"Unexpected dataset: {dataset}.")

        if units is None:
            units = f"{self.current_units} / {self.device.length_units}"
        positions = np.atleast_2d(positions)
        xy = self.device.points
        J_interp = interpolator(xy, J.to(units).magnitude, **interp_kwargs)
        J = J_interp(positions)
        J[~np.isfinite(J)] = 0
        J[~self.device.contains_points(positions)] = 0
        if with_units:
            J = J * self.device.ureg(units)
        return J

    def interp_order_parameter(
        self,
        positions: np.ndarray,
        method: str = "linear",
    ) -> np.ndarray:
        """Interpolates the order parameter at unstructured coordinates.

        Args:
            positions: Shape ``(m, 2)`` array of x, y coordinates at which to evaluate
                the order parameter.
            method: Interpolation method to use, ``"nearest"``, ``"linear"``,
                or ``"cubic"``.

        Returns:
            The interpolated order parameter.
        """
        valid_methods = ("nearest", "linear", "cubic")
        if method not in valid_methods:
            raise ValueError(
                f"Interpolation method must be one of {valid_methods} (got {method})."
            )
        if method == "nearest":
            interpolator = interpolate.NearestNDInterpolator
            interp_kwargs = dict()
        elif method == "linear":
            interpolator = interpolate.LinearNDInterpolator
            interp_kwargs = dict(fill_value=1)
        else:  # "cubic"
            interpolator = interpolate.CloughTocher2DInterpolator
            interp_kwargs = dict(fill_value=1)
        positions = np.atleast_2d(positions)
        xy = self.device.points
        psi = self.tdgl_data.psi
        psi_interp = interpolator(xy, psi, **interp_kwargs)
        return psi_interp(positions)

    def polygon_fluxoid(
        self,
        polygon_points: Union[np.ndarray, Polygon],
        interp_method: str = "linear",
        units: str = "Phi_0",
        with_units: bool = True,
    ) -> Fluxoid:
        """Computes the :class:`tdgl.Fluxoid` (flux + supercurrent) for
        a given polygonal region.

        The fluxoid for a closed region :math:`S` with boundary :math:`\\partial S`
        is defined as:

        .. math::

            \\begin{split}
            \\Phi^f_S &= \\Phi^f_{S,\\text{ flux}} + \\Phi^f_{S,\\text{ supercurrent}}
            \\\\&=\\int_S \\mu_0 H_z(\\mathbf{r})\\,\\mathrm{d}^2r +
                \\oint_{\\partial S}
                \\mu_0\\Lambda(\\mathbf{r})\\mathbf{K}_s(\\mathbf{r})\\cdot\\mathrm{d}\\mathbf{r}
            \\\\&=\\oint_{\\partial S} \\mathbf{A}(\\mathbf{r})\\cdot\\mathrm{d}\\mathbf{r} +
                \\oint_{\\partial S}
                \\mu_0\\Lambda(\\mathbf{r})\\mathbf{K}_s(\\mathbf{r})\\cdot\\mathrm{d}\\mathbf{r}
            \\end{split}

        .. seealso::

            :class:`tdgl.Fluxoid`, :func:`tdgl.make_fluxoid_polygons`

        Args:
            polygon_points: A shape ``(n, 2)`` array of ``(x, y)`` coordinates of
                polygon vertices defining the closed region :math:`S`.
            interp_method: Interpolation method to use, ``"nearest"``, ``"linear"``,
                or ``"cubic"``.
            units: The desired units for the fluxoid.
            with_units: Whether to return values as :class:`pint.Quantity` instances
                with units attached.

        Returns:
            The polygon's :class:`Fluxoid`.
        """
        device = self.device
        ureg = device.ureg
        if units is None:
            units = f"{self.field_units} * {self.device.length_units} ** 2"
        polygon = Polygon(points=polygon_points)
        points = polygon.points
        if not device.film.contains_points(points).all():
            raise ValueError(
                "The polygon must lie completely within the superconducting film."
            )
        # Evaluate the supercurrent density at the polygon coordinates.
        J_units = f"{self.current_units} / {device.length_units}"
        J_poly = self.interp_current_density(
            points,
            dataset="supercurrent",
            method=interp_method,
            units=J_units,
            with_units=True,
        )
        zs = device.layer.z0 * np.ones(len(points))
        dl = np.diff(points, axis=0, prepend=points[:1]) * ureg(device.length_units)
        A_units = f"{self.field_units} * {device.length_units}"
        A_poly = self.vector_potential_at_position(
            points,
            zs=zs,
            units=A_units,
            with_units=True,
            return_sum=True,
        )[:, :2]
        # Compute the flux part of the fluxoid:
        # \oint_{\\partial poly} \vec{A}\cdot\mathrm{d}\vec{r}
        int_A = np.trapz((A_poly * dl).sum(axis=1))
        flux_part = int_A.to(units)
        # Compute the supercurrent part of the fluxoid:
        # \oint_{poly}\Lambda\vec{J}\cdot\mathrm{d}\vec{r}
        Lambda = device.layer.Lambda
        psi_poly = self.interp_order_parameter(points, method=interp_method)
        ns = np.abs(psi_poly) ** 2
        Lambda = Lambda / ns * ureg(device.length_units)
        int_J = np.trapz((Lambda[:, np.newaxis] * J_poly * dl).sum(axis=1))
        supercurrent_part = (ureg("mu_0") * int_J).to(units)
        if not with_units:
            flux_part = flux_part.magnitude
            supercurrent_part = supercurrent_part.magnitude
        return Fluxoid(flux_part, supercurrent_part)

    def hole_fluxoid(
        self,
        hole_name: str,
        points: Union[np.ndarray, None] = None,
        interp_method: str = "linear",
        units: str = "Phi_0",
        with_units: bool = True,
    ) -> Fluxoid:
        """Calculcates the fluxoid for a polygon enclosing the specified hole.

        .. seealso::

            :meth:`tdgl.Solution.polygon_fluxoid`, :meth:`tdgl.Solution.boundary_phases`

        Args:
            hole_name: The name of the hole for which to calculate the fluxoid.
            points: The vertices of the polygon enclosing the hole. If None is given,
                a polygon is generated using
                :func:`tdgl.make_fluxoid_polygons`.
            interp_method: Interpolation method to use, ``"nearest"``, ``"linear"``,
                or ``"cubic"``.
            units: The desired units for the fluxoid.
            with_units: Whether to return values as :class:`pint.Quantity` instances
                with units attached.

        Returns:
            The hole's :class:`tdgl.Fluxoid`.
        """
        if points is None:
            from ..fluxoid import make_fluxoid_polygons

            points = make_fluxoid_polygons(self.device, holes=hole_name)[hole_name]
        hole = {hole.name: hole for hole in self.device.holes}[hole_name]
        if not Polygon(points=points).contains_points(hole.points).all():
            raise ValueError(
                f"Hole {hole_name} is not completely enclosed by the given polygon."
            )
        return self.polygon_fluxoid(
            points,
            interp_method=interp_method,
            units=units,
            with_units=with_units,
        )

    def boundary_phases(
        self, delta: bool = False
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Returns a dict of ``{polygon_name: (boundary_indices, boundary_phases)}``.

        ``(boundary_phases[-1] - boundary_phases[0]) / (2 * np.pi)`` gives the winding
        number for the polygon, i.e., the fluxoid in units of ``Phi_0``.

        .. seealso::

            :meth:`tdgl.Solution.hole_fluxoid`

        Args:
            delta: If True, ``boundary_phases[0]`` will be subtracted for each polygon.

        Returns:
            ``{polygon_name: (boundary_indices, boundary_phases)}``
        """
        device = self.device
        boundary_indices = device.boundary_sites()
        psi = self.tdgl_data.psi
        theta = np.angle(psi)
        phases = {}
        for name, indices in boundary_indices.items():
            phase = np.unwrap(theta[indices])
            if delta:
                phase -= phase[0]
            phases[name] = BoundaryPhases(indices, phase)
        return phases

    def current_through_path(
        self,
        path_coords: np.ndarray,
        dataset: Union[str, None] = None,
        method: str = "linear",
        units: Union[str, None] = None,
        with_units: bool = True,
    ) -> Union[float, pint.Quantity]:
        """Calculates the total current crossing a given path.

        Args:
            path_coords: An ``(n, 2)`` array of ``(x, y)`` coordinates defining
                the path.
            dataset: ``None``, ``"supercurrent"``, or ``"normal_current"``.
                ``None`` indicates the total current.
            method: Interpolation method: either "linear" or "cubic".
            units: The current units to return.
            with_units: Whether to return a :class:`pint.Quantity` with units attached.

        Returns:
            The total current crossing the path as either a float or a
            :class:`pint.Quantity`.
        """
        device = self.device
        if units is None:
            units = self.current_units
        J = self.interp_current_density(
            path_coords,
            dataset=dataset,
            method=method,
            with_units=True,
        )
        # The center of each edge in the path
        edge_positions = (path_coords[:-1] + path_coords[1:]) / 2
        # Evaluate the supercurrent at the edge centers
        J_edge = (J[:-1] + J[1:]) / 2
        edge_lengths, unit_normals = path_vectors(path_coords)
        edge_lengths = edge_lengths * device.ureg(device.length_units)
        J_dot_n = (J_edge * unit_normals).sum(axis=1)
        # Exclude points that are not inside the device.
        in_device = self.device.contains_points(edge_positions)
        total_current = np.trapz((J_dot_n * edge_lengths)[in_device]).to(units)
        if not with_units:
            total_current = total_current.magnitude
        return total_current

    def field_at_position(
        self,
        positions: np.ndarray,
        *,
        zs: Optional[Union[float, np.ndarray]] = None,
        vector: bool = False,
        units: Optional[str] = None,
        with_units: bool = True,
        return_sum: bool = True,
    ) -> Union[BiotSavartField, np.ndarray]:
        """Calculates the field due to currents in the device at any point(s) in space.

        .. seealso::

            :class:`tdgl.BiotSavartField`

        Args:
            positions: Shape (m, 2) array of (x, y) coordinates, or (m, 3) array
                of (x, y, z) coordinates at which to calculate the magnetic field.
                A single sequence like [x, y] or [x, y, z] is also allowed.
            zs: z coordinates at which to calculate the field. If positions has shape
                (m, 3), then this argument is not allowed. If zs is a scalar, then
                the fields are calculated in a plane parallel to the x-y plane.
                If zs is any array, then it must be same length as positions.
            vector: Whether to return the full vector magnetic field
                or just the z component.
            units: Units to which to convert the fields (can be either magnetic field H
                or magnetic flux density B = mu0 * H). If not given, then the fields
                are returned in units of ``self.field_units``.
            with_units: Whether to return the fields as ``pint.Quantity``
                with units attached.
            return_sum: If ``False``, this method will return a :class:`tdgl.BiotSavartField`
                instance, where the field from the supercurrent and normal current
                are identified separately.

        Returns:
            An np.ndarray if ``return_sum`` is ``True``, otherwise an instance of
            :class:`tdgl.BiotSavartField`. If ``with_units`` is ``True``, then the
            array(s) will be of type :class:`pint.Quantity`. The array(s) will have
            shape ``(m, )`` if vector is False, or shape ``(m, 3)`` if ``vector`` is True.
        """
        device = self.device
        ureg = device.ureg
        points = device.points
        units = units or self.field_units
        # In case something like a list [x, y] or [x, y, z] is given
        positions = np.atleast_2d(positions)
        # If positions includes z coordinates, peel those off here
        if positions.shape[1] == 3:
            if zs is not None:
                raise ValueError(
                    "If positions has shape (m, 3) then zs cannot be specified."
                )
            zs = positions[:, 2]
            positions = positions[:, :2]
        elif isinstance(zs, (int, float, np.generic)):
            # constant zs
            zs = zs * np.ones(len(positions))
        zs = zs.squeeze()
        if not isinstance(zs, np.ndarray):
            raise ValueError(f"Expected zs to be an ndarray, but got {type(zs)}.")
        weights = device.mesh.areas * device.coherence_length.magnitude**2
        # Compute the fields at the specified positions from the currents in each layer
        layer = self.device.layer
        if np.all((zs - layer.z0) == 0):
            if device.film.contains_points(positions).any():
                raise ValueError("Cannot interpolate fields within a film.")
        fields = []
        for name in ("supercurrent_density", "normal_current_density"):
            J = (
                getattr(self, name)
                .to(f"{self.current_units} / {device.length_units}")
                .magnitude
            )
            H = biot_savart_2d(
                positions[:, 0],
                positions[:, 1],
                zs,
                positions=points,
                areas=weights,
                current_densities=J,
                z0=layer.z0,
                length_units=device.length_units,
                current_units=self.current_units,
                vector=vector,
            )
            field = convert_field(
                H,
                units,
                old_units="tesla",
                ureg=ureg,
                with_units=with_units,
            )
            fields.append(field)
        fields = BiotSavartField(*fields)
        if return_sum:
            return sum(fields)
        return fields

    def vector_potential_at_position(
        self,
        positions: np.ndarray,
        *,
        zs: Union[float, np.ndarray, None] = None,
        units: Union[str, None] = None,
        with_units: bool = True,
        return_sum: bool = True,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Calculates the vector potential due to currents in the device at any
        point(s) in space, plus the applied vector potential.

        The vector potential :math:`\\mathbf{A}` at position :math:`\\mathbf{r}`
        due to sheet current density :math:`\\mathbf{K}(\\mathbf{r}')` flowing in a film
        with lateral geometry :math:`S` is:

        .. math::

            \\mathbf{A}(\\mathbf{r}) = \\frac{\\mu_0}{4\\pi}
            \\int_S\\frac{\\mathbf{K}(\\mathbf{r}')}{|\\mathbf{r}-\\mathbf{r}'|}\\mathrm{d}^2r'.

        Args:
            positions: Shape (m, 2) array of (x, y) coordinates, or (m, 3) array
                of (x, y, z) coordinates at which to calculate the vector potential.
                A single list like [x, y] or [x, y, z] is also allowed.
            zs: z coordinates at which to calculate the potential. If positions has shape
                (m, 3), then this argument is not allowed. If zs is a scalar, then
                the fields are calculated in a plane parallel to the x-y plane.
                If zs is any array, then it must be same length as positions.
            units: Units to which to convert the vector potential.
            with_units: Whether to return the vector potential as a ``pint.Quantity``
                with units attached.
            return_sum: Whether to return the total potential or a dict with keys
                ``("applied", "supercurrent", "normal_current")``.

        Returns:
            An np.ndarray if ``return_sum`` is ``True``, otherwise a dict of
            ``{source: potential_from_source}``. If ``with_units`` is ``True``, then
            the array(s) will be of type :class:`pint.Quantity`.
            ``potential_from_source`` will have shape ``(m, 3)``.
        """
        device = self.device
        ureg = device.ureg
        points = device.points
        areas = device.mesh.areas * device.coherence_length.magnitude**2
        units = units or f"{self.field_units} * {device.length_units}"
        # In case something like a list [x, y] or [x, y, z] is given
        positions = np.atleast_2d(positions)
        # If positions includes z coordinates, peel those off here
        if positions.shape[1] == 3:
            if zs is not None:
                raise ValueError(
                    "If positions has shape (m, 3) then zs cannot be specified."
                )
            zs = positions[:, 2]
            positions = positions[:, :2]
        elif isinstance(zs, (int, float, np.generic)):
            # constant zs
            zs = zs * np.ones(len(positions))
        if not isinstance(zs, np.ndarray):
            raise ValueError(f"Expected zs to be an ndarray, but got {type(zs)}.")
        if zs.ndim == 1:
            # We need zs to be shape (m, 1)
            zs = zs[:, np.newaxis]
        rho2 = distance.cdist(positions, points, metric="sqeuclidean")
        layer = device.layer
        vector_potentials = {}
        applied = self.applied_vector_potential(
            positions[:, 0],
            positions[:, 1],
            zs.squeeze(),
        )
        if applied.shape[1] == 2:
            applied = A = np.concatenate(
                [applied, np.zeros_like(applied[:, :1])], axis=1
            )
        applied = (applied * ureg(f"{self.field_units} * {device.length_units}")).to(
            units
        )
        if not with_units:
            applied = applied.magnitude
        vector_potentials["applied"] = applied
        dz = zs - layer.z0
        # rho has units of [length] and
        # shape = (postitions.shape[0], device.points.shape[0], 1)
        rho = np.sqrt(rho2 + dz**2)[:, :, np.newaxis]
        J_units = f"{self.current_units} / {device.length_units}"
        for name in ("supercurrent_density", "normal_current_density"):
            # J has units of [current / length], shape = (device.points.shape[0], 2)
            J = getattr(self, name).to(J_units).magnitude
            Axy = np.einsum("ijk, j -> ik", J / rho, areas)
            # z-component is zero because currents are parallel to the x-y plane.
            A = np.concatenate([Axy, np.zeros_like(Axy[:, :1])], axis=1)
            A = A * ureg(self.current_units)
            A = (ureg("mu_0") / (4 * np.pi) * A).to(units)
            if not with_units:
                A = A.magnitude
            vector_potentials[name] = A
        if return_sum:
            return sum(vector_potentials.values())
        return vector_potentials

    def _save_to_hdf5_file(
        self,
        h5file: Union[h5py.File, str],
        save_tdgl_data: bool = False,
        save_mesh: bool = True,
    ) -> None:
        def serialize_func(func, name, h5group):
            try:
                h5group.attrs[name] = func
            except TypeError:
                # Unsupported dtype - just pickle it.
                h5group[f"{name}.pickle"] = np.void(cloudpickle.dumps(func))

        if isinstance(h5file, str):
            mode = "x" if save_tdgl_data else "r+"
            save_context = h5py.File(h5file, mode, libver="latest")
        else:
            save_context = nullcontext(h5file)

        with save_context as f:
            if "mesh" in f:
                del f["mesh"]
            data_grp = f.require_group("data")
            if save_tdgl_data:
                self.tdgl_data.to_hdf5(data_grp)
                self.dynamics.to_hdf5(data_grp.require_group(str(self.tdgl_data.step)))
            if "solution" in f:
                del f["solution"]
            group = f.create_group("solution")
            options_grp = group.create_group("options")
            for k, v in dataclasses.asdict(self.options).items():
                if v is not None:
                    options_grp.attrs[k] = v
            group.attrs["time_created"] = self.time_created.isoformat()
            group.attrs["current_units"] = self.current_units
            group.attrs["field_units"] = self.field_units
            serialize_func(
                self.applied_vector_potential,
                "applied_vector_potential",
                group,
            )
            serialize_func(
                self.terminal_currents,
                "terminal_currents",
                group,
            )
            serialize_func(
                self.disorder_epsilon,
                "disorder_epsilon",
                group,
            )
            group.attrs["total_seconds"] = self.total_seconds
            self.device.to_hdf5(group.create_group("device"), save_mesh=save_mesh)

    def to_hdf5(self, h5path: Union[str, None] = None, save_mesh: bool = True) -> None:
        """Save the Solution to the existing output HDF5 file or to a new HDF5 file.

        Args:
            h5path: Path to an HDF5 file. If ``None`` is given, the
                :class:`tdgl.Solution` will be saved to the existing HDF5 output file
                located at ``self.path``.
            save_mesh: Whether to save the Device's mesh.
        """
        if self.saved_on_disk:
            if h5path is None:
                self._save_to_hdf5_file(self.path, save_mesh=save_mesh)
            else:
                shutil.copy(self.path, h5path)
                self._save_to_hdf5_file(h5path, save_mesh=save_mesh)
            return

        if h5path is None:
            raise ValueError(
                "The solution HDF5 file does not exist, "
                "and a new HDF5 file was not given."
            )
        self._save_to_hdf5_file(h5path, save_tdgl_data=True, save_mesh=save_mesh)

    @staticmethod
    def from_hdf5(path: os.PathLike, solve_step: int = -1) -> "Solution":
        """Loads a :class:`tdgl.Solution` from file.

        Args:
            path: Path to the HDF5 file containing a serialized :class:`tdgl.Solution`.
            solve_step: The solve step to load.

        Returns:
            The loaded Solution instance.
        """

        def deserialize_func(name, h5group):
            if name in h5group.attrs:
                return h5group.attrs[name]
            if f"{name}.pickle" in h5group:
                return pickle.loads(np.void(grp[f"{name}.pickle"]).tobytes())
            raise IOError(f"Unable to load {name}.")

        with h5py.File(path, "r", libver="latest") as f:
            grp = f["solution"]
            options_kwargs = dict()
            for k, v in grp["options"].attrs.items():
                options_kwargs[k] = v
            options = SolverOptions(**options_kwargs)
            time_created = datetime.fromisoformat(grp.attrs["time_created"])
            vector_potential = deserialize_func("applied_vector_potential", grp)
            terminal_currents = deserialize_func("terminal_currents", grp)
            disorder_epsilon = deserialize_func("disorder_epsilon", grp)
            total_seconds = grp.attrs["total_seconds"]
            device = Device.from_hdf5(grp["device"])

        solution = Solution(
            device=device,
            path=path,
            options=options,
            applied_vector_potential=vector_potential,
            terminal_currents=terminal_currents,
            disorder_epsilon=disorder_epsilon,
            total_seconds=total_seconds,
            _solve_step=solve_step,
        )
        solution._time_created = time_created
        return solution

    def delete_hdf5(self) -> None:
        """Delete the HDF5 file accompanying the :class:`tdgl.Solution`."""
        if self.saved_on_disk:
            os.remove(self.path)

    def equals(
        self,
        other: Any,
        require_same_timestamp: bool = False,
    ) -> bool:
        """Checks whether two solutions are equal.

        Args:
            other: The :class:`tdgl.Solution` to compare for equality.
            require_same_timestamp: If True, two solutions are only considered
                equal if they have the exact same time_created.

        Returns:
            A boolean indicating whether the two solutions are equal
        """
        # First check things that are "easy" to check
        if other is self:
            return True
        if not isinstance(other, Solution):
            return False

        def compare_callables(first, second):
            if isinstance(first, Parameter):
                return first == second
            if callable(first):
                if not callable(second):
                    return False
                get_code = operator.attrgetter("co_code", "co_consts")
                if get_code(first.__code__) != get_code(second.__code__):
                    return False
            elif first != second:
                return False
            return True

        if not (
            (self.device == other.device)
            and (self.options == other.options)
            and (self.solve_step == other.solve_step)
            and compare_callables(
                self.applied_vector_potential, other.applied_vector_potential
            )
            and compare_callables(self.terminal_currents, other.terminal_currents)
            and compare_callables(self.disorder_epsilon, other.disorder_epsilon)
            and (self.tdgl_data == other.tdgl_data)
            and (self.dynamics == other.dynamics)
        ):
            return False
        if require_same_timestamp and (self.time_created != other.time_created):
            return False
        return True

    def __eq__(self, other) -> bool:
        return self.equals(other, require_same_timestamp=True)

    def plot_currents(self, **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """An alias for :func:`tdgl.plot_currents`."""
        from .plot_solution import plot_currents

        return plot_currents(self, **kwargs)

    def plot_order_parameter(self, **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """An alias for :func:`tdgl.plot_order_parameter`."""
        from .plot_solution import plot_order_parameter

        return plot_order_parameter(self, **kwargs)

    def plot_field_at_positions(
        self, positions: np.ndarray, **kwargs
    ) -> Tuple[plt.Figure, np.ndarray]:
        """An alias for :func:`tdgl.plot_field_at_positions`."""
        from .plot_solution import plot_field_at_positions

        return plot_field_at_positions(self, positions, **kwargs)

    def plot_vorticity(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """An alias for :func:`tdgl.plot_vorticity`."""
        from .plot_solution import plot_vorticity

        return plot_vorticity(self, **kwargs)

    def plot_scalar_potential(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """An alias for :func:`tdgl.plot_scalar_potential`."""
        from .plot_solution import plot_scalar_potential

        return plot_scalar_potential(self, **kwargs)
