import dataclasses
import logging
import os
from datetime import datetime
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import dill
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pint
from scipy import interpolate
from scipy.spatial import distance

from .device.components import Polygon

# from .about import version_dict
from .device.device import Device
from .em import biot_savart_2d, convert_field
from .finite_volume.matrices import build_gradient
from .parameter import Parameter
from .solver.runner import SolverOptions
from .visualization.helpers import (
    TDGLData,
    get_data_range,
    get_edge_observable_data,
    load_state_data,
    load_tdgl_data,
)

logger = logging.getLogger(__name__)


class Fluxoid(NamedTuple):
    """The fluxoid for a closed region :math:`S` with boundary :math:`\\partial S`
    is defined as:

    .. math::

        \\Phi^f_S = \\underbrace{
            \\int_S \\mu_0 H_z(\\vec{r})\\,\\mathrm{d}^2r
        }_{\\text{flux part}}
        + \\underbrace{
            \\oint_{\\partial S}
            \\mu_0\\Lambda(\\vec{r})\\vec{J}(\\vec{r})\\cdot\\mathrm{d}\\vec{r}
        }_{\\text{supercurrent part}}

    Args:
        flux_part: :math:`\\int_S \\mu_0 H_z(\\vec{r})\\,\\mathrm{d}^2r`.
        supercurrent_part: :math:`\\oint_{\\partial S}\\mu_0\\Lambda(\\vec{r})\\vec{J}(\\vec{r})\\cdot\\mathrm{d}\\vec{r}`.
    """

    flux_part: Union[float, pint.Quantity]
    supercurrent_part: Union[float, pint.Quantity]


class BiotSavartField(NamedTuple):
    supercurrent: np.ndarray
    normal_current: np.ndarray


class Solution:
    """A container for the calculated stream functions and fields,
    with some convenient data processing methods.

    Args:
        device: The ``Device`` that was solved
        streams: A dict of ``{layer_name: stream_function}``
        current_densities: A dict of ``{layer_name: current_density}``
        fields: A dict of ``{layer_name: total_field}``
        screening_fields: A dict of ``{layer_name: screening_field}``
        applied_field: The function defining the applied field
        field_units: Units of the applied field
        current_units: Units used for current quantities.
        circulating_currents: A dict of ``{hole_name: circulating_current}``.
        terminal_currents: A dict of ``{terminal_name: terminal_current}``.
        vortices: A list of ``Vortex`` objects located in the ``Device``.
        solver: The solver method that generated the solution.
    """

    def __init__(
        self,
        *,
        device: Device,
        filename: os.PathLike,
        options: SolverOptions,
        applied_vector_potential: Parameter,
        source_drain_current: Union[float, str, pint.Quantity],
        field_units: str,
        current_units: str,
        total_seconds: float,
        solver: str = "tdgl.solve",
    ):
        self.device = device.copy()
        self.device.mesh = device.mesh
        self.options = options
        self.path = filename
        self.applied_vector_potential = applied_vector_potential
        self.source_drain_current = float(source_drain_current)

        self.supercurrent_density: Optional[np.ndarray] = None
        self.normal_current_density: Optional[np.ndarray] = None
        self.vorticity: Optional[np.ndarray] = None

        # Make field_units and current_units "read-only" attributes.
        # The should never be changed after instantiation.
        self._field_units = field_units
        self._current_units = current_units
        self._solver = solver
        self._time_created = datetime.now()
        self.total_seconds = total_seconds

        self.tdgl_data: Optional[TDGLData] = None
        self.state: Optional[dict[str, Any]] = None
        self._solve_step: int = -1
        self.load_tdgl_data(self._solve_step)

        # self._version_info = version_dict()

    @property
    def solve_step(self) -> int:
        return self._solve_step

    def load_tdgl_data(self, solve_step: int = -1) -> None:
        """Loads the TDGL results from file for a given solve step.

        Args:
            solve_step: The step index for which to load data.
                Defaults to -1, i.e. the final step.
        """
        with h5py.File(self.path, "r", libver="latest") as f:
            step_min, step_max = get_data_range(f)
            if solve_step == 0:
                step = step_min
            elif solve_step < 0:
                step = step_max + 1 + solve_step
            else:
                step = solve_step
            self.tdgl_data = load_tdgl_data(f, step)
            self.state = load_state_data(f, step)
        mesh = self.device.mesh
        device = self.device
        self._solve_step = step
        supercurrent, sc_direc, _ = get_edge_observable_data(
            self.tdgl_data.supercurrent, mesh
        )
        normal_current, nc_direc, _ = get_edge_observable_data(
            self.tdgl_data.normal_current, mesh
        )
        K0 = self.device.K0.to(f"{self.current_units} / {self.device.length_units}")
        self.supercurrent_density = K0 * supercurrent[:, np.newaxis] * sc_direc
        self.normal_current_density = K0 * normal_current[:, np.newaxis] * nc_direc

        j_sc_site = mesh.get_observable_on_site(self.tdgl_data.supercurrent)
        j_nm_site = mesh.get_observable_on_site(self.tdgl_data.normal_current)
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
        scale = (
            device.K0 / (device.coherence_length * device.ureg(device.length_units))
        ).to(f"{self.current_units} / {self.device.length_units}**2")
        self.vorticity = vorticity * scale

    @property
    def current_density(self) -> pint.Quantity:
        if self.supercurrent_density is None:
            return None
        return self.supercurrent_density + self.normal_current_density

    @solve_step.setter
    def solve_step(self, step: int) -> None:
        self.load_tdgl_data(solve_step=step)

    @property
    def field_units(self) -> str:
        """The units in which magnetic fields are specified."""
        return self._field_units

    @property
    def current_units(self) -> str:
        """The units in which currents are specified."""
        return self._current_units

    @property
    def solver(self) -> str:
        """The solver method that generated the solution."""
        return self._solver

    @property
    def time_created(self) -> datetime:
        """The time at which the solution was originally created."""
        return self._time_created

    # @property
    # def version_info(self) -> Dict[str, str]:
    #     """A dictionary of dependency versions."""
    #     return self._version_info

    def grid_current_density(
        self,
        *,
        dataset: Optional[str] = None,
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
        method: str = "linear",
        units: Optional[str] = None,
        with_units: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the current density ``J = [dg/dy, -dg/dx]`` on a rectangular grid.

        Keyword arguments are passed to scipy.interpolate.griddata().

        Args:
            grid_shape: Shape of the desired rectangular grid. If a single integer
                N is given, then the grid will be square, shape = (N, N).
            method: Interpolation method to use (see scipy.interpolate.griddata).
            units: The desired units for the current density. Defaults to
                ``self.current_units / self.device.length_units``.
            with_units: Whether to return arrays of pint.Quantities with units attached.

        Returns:
            x grid, y grid, nterpolated current density
        """
        if dataset is None:
            J = self.current_density
        elif dataset in ["supercurrent"]:
            J = self.supercurrent_density
        elif dataset in ["normal_current"]:
            J = self.normal_current_density
        else:
            raise ValueError(f"Unexpected dataset: {dataset}.")
        units = units or f"{self.current_units} / {self.device.length_units}"
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
        Jgrid = (
            J.units * np.array([Jx.reshape(grid_shape), Jy.reshape(grid_shape)])
        ).to(units)
        if with_units:
            length_units = self.device.ureg(self.device.length_units)
            xgrid = xgrid * length_units
            ygrid = ygrid * length_units
        if not with_units:
            Jgrid = Jgrid.magnitude
        return xgrid, ygrid, Jgrid

    def interp_current_density(
        self,
        positions: np.ndarray,
        *,
        dataset: Optional[str] = None,
        method: str = "linear",
        units: Optional[str] = None,
        with_units: bool = False,
    ):
        """Computes the current density ``J = [dg/dy, -dg/dx]``
        at unstructured coordinates via interpolation.

        Args:
            positions: Shape ``(m, 2)`` array of x, y coordinates at which to evaluate
                the current density.
            layers: Name(s) of the layer(s) for which to interpolate current density.
            method: Interpolation method to use, 'nearest', 'linear', or 'cubic'.
            units: The desired units for the current density. Defaults to
                ``self.current_units / self.device.length_units``.
            with_units: Whether to return arrays of pint.Quantities with units attached.

        Returns:
            A dict of interpolated current density for each layer.
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
        elif dataset in ["supercurrent"]:
            J = self.supercurrent_density
        elif dataset in ["normal_current"]:
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
        if with_units:
            J = J * self.device.ureg(units)
        return J

    def interp_order_parameter(
        self,
        positions: np.ndarray,
        *,
        method: str = "linear",
    ):
        """Computes the current density ``J = [dg/dy, -dg/dx]``
        at unstructured coordinates via interpolation.

        Args:
            positions: Shape ``(m, 2)`` array of x, y coordinates at which to evaluate
                the current density.
            method: Interpolation method to use (see scipy.interpolate.griddata).
            units: The desired units for the current density. Defaults to
                ``self.current_units / self.device.length_units``.
            with_units: Whether to return arrays of pint.Quantities with units attached.

        Returns:
            A dict of interpolated current density for each layer.
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
        units: Optional[str] = "Phi_0",
        with_units: bool = True,
    ) -> Dict[str, Fluxoid]:
        """Computes the :class:`Fluxoid` (flux + supercurrent) for
        a given polygonal region.

        The fluxoid for a closed region :math:`S` with boundary :math:`\\partial S`
        is defined as:

        .. math::

            \\Phi^f_S = \\underbrace{
                \\int_S \\mu_0 H_z(\\vec{r})\\,\\mathrm{d}^2r
            }_{\\text{flux part}}
            + \\underbrace{
                \\oint_{\\partial S}
                \\mu_0\\Lambda(\\vec{r})\\vec{J}(\\vec{r})\\cdot\\mathrm{d}\\vec{r}
            }_{\\text{supercurrent part}}

        Args:
            polygon_points: A shape ``(n, 2)`` array of ``(x, y)`` coordinates of
                polygon vertices defining the closed region :math:`S`.
            layers: Name(s) of the layer(s) for which to compute the fluxoid.
            grid_shape: Shape of the desired rectangular grid to use for interpolation.
                If a single integer N is given, then the grid will be square,
                shape = (N, N).
            interp_method: Interpolation method to use.
            units: The desired units for the current density.
                Defaults to :math:`\\Phi_0`.
            with_units: Whether to return values as pint.Quantities with units attached.

        Returns:
            A dict of ``{layer_name: fluxoid}`` for each specified layer, where
            ``fluxoid`` is an instance of :class:`Fluxoid`.
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
        zs = device.layer.z0 * np.ones(points.shape[0])
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
        # ns = np.ones(dl.shape[0])
        Lambda = Lambda / ns * ureg(device.length_units)
        int_J = np.trapz((Lambda[:, np.newaxis] * J_poly * dl).sum(axis=1))
        supercurrent_part = (ureg("mu_0") * int_J).to(units)
        if not with_units:
            flux_part = flux_part.magnitude
            supercurrent_part = supercurrent_part.magnitude
        fluxoid = Fluxoid(flux_part, supercurrent_part)
        return fluxoid

    def hole_fluxoid(
        self,
        hole_name: str,
        points: Optional[np.ndarray] = None,
        interp_method: str = "linear",
        units: Optional[str] = "Phi_0",
        with_units: bool = True,
    ) -> Fluxoid:
        """Calculcates the fluxoid for a polygon enclosing the specified hole.

        Args:
            hole_name: The name of the hole for which to calculate the fluxoid.
            points: The vertices of the polygon enclosing the hole. If None is given,
                a polygon is generated using
                :func:`tdgl.fluxoid.make_fluxoid_polygons`.
            interp_method: Interpolation method to use.
            units: The desired units for the current density.
                Defaults to :math:`\\Phi_0`.
            with_units: Whether to return values as pint.Quantities with units attached.

        Returns:
            The hole's Fluxoid.
        """
        if points is None:
            from .fluxoid import make_fluxoid_polygons

            points = make_fluxoid_polygons(self.device, holes=hole_name)[hole_name]
        hole = self.device.holes[hole_name]
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
            return_sum: Whether to return the sum of the fields from all layers in
                the device, or a dict of ``{layer_name: field_from_layer}``.

        Returns:
            An np.ndarray if return_sum is True, otherwise a dict of
            ``{layer_name: field_from_layer}``. If with_units is True, then the
            array(s) will contain pint.Quantities. ``field_from_layer`` will have
            shape ``(m, )`` if vector is False, or shape ``(m, 3)`` if ``vector`` is True.
        """
        device = self.device
        dtype = device.solve_dtype
        ureg = device.ureg
        points = device.points.astype(dtype, copy=False)
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
            zs = zs * np.ones(positions.shape[0], dtype=dtype)
        zs = zs.squeeze()
        if not isinstance(zs, np.ndarray):
            raise ValueError(f"Expected zs to be an ndarray, but got {type(zs)}.")
        weights = device.mesh.areas * device.coherence_length**2
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
        zs: Optional[Union[float, np.ndarray]] = None,
        units: Optional[str] = None,
        with_units: bool = True,
        return_sum: bool = True,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Calculates the vector potential due to currents in the device at any
        point(s) in space, plus the applied vector potential.

        The vector potential :math:`\\vec{A}` at position :math:`\\vec{r}`
        due to sheet current density :math:`\\vec{J}(\\vec{r}')` flowing in a film
        with lateral geometry :math:`S` is:

        .. math::

            \\vec{A}(\\vec{r}) = \\frac{\\mu_0}{4\\pi}
            \\int_S\\frac{\\vec{J}(\\vec{r}')}{|\\vec{r}-\\vec{r}'|}\\mathrm{d}^2r'.

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
                ('applied', 'supercurrent', 'normal_current').

        Returns:
            An np.ndarray if return_sum is True, otherwise a dict of
            ``{source: potential_from_source}``. If with_units is True, then the
            array(s) will contain pint.Quantities. ``potential_from_source`` will have
            shape ``(m, 3)``.
        """
        device = self.device
        dtype = device.solve_dtype
        ureg = device.ureg
        points = device.points.astype(dtype, copy=False)
        areas = device.mesh.areas * device.coherence_length**2
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
            zs = zs * np.ones(positions.shape[0], dtype=dtype)
        if not isinstance(zs, np.ndarray):
            raise ValueError(f"Expected zs to be an ndarray, but got {type(zs)}.")
        if zs.ndim == 1:
            # We need zs to be shape (m, 1)
            zs = zs[:, np.newaxis]
        rho2 = distance.cdist(positions, points, metric="sqeuclidean").astype(
            dtype, copy=False
        )
        layer = device.layer
        vector_potentials = {}
        applied = self.applied_vector_potential(
            positions[:, 0],
            positions[:, 1],
            zs.squeeze(),
        ) * ureg(f"{self.field_units} * {device.length_units}")
        applied.ito(units)
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
            Axy = np.einsum("ijk, j -> ik", J / rho, areas, dtype=dtype)
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

    def to_hdf5(self, save_mesh: bool = True) -> None:
        """Save the Solution to the existing output HDF5 file.

        Args:
            save_mesh: Whether to save the Device's mesh.
        """
        with h5py.File(self.path, "r+", libver="latest") as f:
            if "mesh" in f:
                del f["mesh"]
            data_grp = f.require_group("data")
            for k, v in dataclasses.asdict(self.options).items():
                if v is not None:
                    data_grp.attrs[k] = v
            group = f.create_group("solution")
            group.attrs["time_created"] = self.time_created.isoformat()
            group.attrs["current_units"] = self.current_units
            group.attrs["field_units"] = self.field_units
            try:
                # See: https://docs.h5py.org/en/2.8.0/strings.html
                group.attrs["applied_vector_potential"] = np.void(
                    dill.dumps(self.applied_vector_potential)
                )
            except RuntimeError as e:
                dirname = os.path.dirname(self.path)
                fname = os.path.basename(self.path).replace(".h5", "")
                dill_path = os.path.join(
                    dirname,
                    f"applied_vector_potential-{fname}.dill",
                )
                logger.warning(
                    f"Unable to serialize the applied vector potential to HDF5: {e}. "
                    f"Saving the applied vector potential to {dill_path!r} instead."
                )
                with open(dill_path, "wb") as f:
                    dill.dump(self.applied_vector_potential, f)
            group.attrs["source_drain_current"] = self.source_drain_current
            group.attrs["total_seconds"] = self.total_seconds
            self.device.to_hdf5(group.create_group("device"), save_mesh=save_mesh)

    @classmethod
    def from_hdf5(cls, path: os.PathLike) -> "Solution":
        """Loads a Solution from file.

        Args:
            path: Path to the HDF5 file containing a serialized Solution.

        Returns:
            The loaded Solution instance
        """
        with h5py.File(path, "r", libver="latest") as f:
            fname = os.path.basename(path).replace(".h5", "")
            dill_path = f"applied_vector_potential-{fname}.dill"
            data_grp = f["data"]
            options_kwargs = dict()
            for k, v in data_grp.attrs.items():
                options_kwargs[k] = v
            options = SolverOptions(**options_kwargs)
            grp = f["solution"]
            time_created = datetime.fromisoformat(grp.attrs["time_created"])
            current_units = grp.attrs["current_units"]
            field_units = grp.attrs["field_units"]
            if "applied_vector_potential" in grp.attrs:
                # See: https://docs.h5py.org/en/2.8.0/strings.html
                vector_potential = dill.loads(
                    grp.attrs["applied_vector_potential"].tostring()
                )
            elif dill_path in os.listdir(os.path.dirname(path)):
                with open(dill_path, "rb") as f:
                    vector_potential = dill.load(f)
            else:
                raise IOError(f"Unable to load applied vector potential from {path!r}.")
            current = grp.attrs["source_drain_current"]
            total_seconds = grp.attrs["total_seconds"]
            device = Device.from_hdf5(grp["device"])

        solution = Solution(
            device=device,
            filename=path,
            options=options,
            applied_vector_potential=vector_potential,
            source_drain_current=current,
            current_units=current_units,
            field_units=field_units,
            total_seconds=total_seconds,
        )

        # Set "read-only" attributes
        solution._time_created = time_created

        return solution

    def equals(
        self,
        other: Any,
        require_same_timestamp: bool = False,
    ) -> bool:
        """Checks whether two solutions are equal.

        Args:
            other: The Solution to compare for equality.
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

        if not (
            (self.device == other.device)
            and (self.field_units == other.field_units)
            and (self.current_units == other.current_units)
            and (self.source_drain_current == other.source_drain_current)
            and (self.path == other.path)
            and (self.solve_step == other.solve_step)
            and (self.applied_vector_potential == other.applied_vector_potential)
        ):
            return False
        if require_same_timestamp and (self.time_created != other.time_created):
            return False
        return True

    def __eq__(self, other) -> bool:
        return self.equals(other, require_same_timestamp=True)

    def plot_currents(self, **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for :func:`tdgl.visualization.plot_currents`."""
        from .visualization.visualization import plot_currents

        return plot_currents(self, **kwargs)

    def plot_order_parameter(self, **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for :func:`tdgl.visualization.plot_order_parameter`."""
        from .visualization.visualization import plot_order_parameter

        return plot_order_parameter(self, **kwargs)

    def plot_field_at_positions(
        self, points: np.ndarray, **kwargs
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for :func:`tdgl.visualization.plot_field_at_positions`."""
        from .visualization.visualization import plot_field_at_positions

        return plot_field_at_positions(self, points, **kwargs)

    def plot_vorticity(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alias for :func:`tdgl.visualization.plot_vorticity`."""
        from .visualization.visualization import plot_vorticity

        return plot_vorticity(self, **kwargs)

    def plot_scalar_potential(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alis for :func:`tdgl.visualization.plot_scalar_potential`."""
        from .visualization.visualization import plot_scalar_potential

        return plot_scalar_potential(self, **kwargs)
