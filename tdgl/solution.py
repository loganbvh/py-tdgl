import os
import json
import logging
import zipfile
from datetime import datetime
from typing import (
    Optional,
    Union,
    Dict,
    Tuple,
    Any,
    NamedTuple,
)

import dill
import pint
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial import distance

# from .about import version_dict
from .device.device import Device
from .device.components import Polygon
from .fem import mass_matrix
from .parameter import Parameter
from .em import biot_savart_2d, convert_field
from ._core.mesh.mesh import Mesh
from ._core.tdgl import get_observable_on_site
from ._core.matrices import build_gradient
from ._core.visualization.helpers import (
    get_data_range,
    load_tdgl_data,
    load_state_data,
    get_edge_observable_data,
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
        applied_vector_potential: Parameter,
        source_drain_current: Union[float, str, pint.Quantity],
        field_units: str,
        current_units: str,
        total_seconds: float,
        solver: str = "tdgl.solve",
    ):
        self.device = device.copy()
        self.path = filename
        self.applied_vector_potential = applied_vector_potential
        self.source_drain_current = source_drain_current

        self.supercurrent_density = None
        self.normal_current_density = None
        self.vorticity = None

        # Make field_units and current_units "read-only" attributes.
        # The should never be changed after instantiation.
        self._field_units = field_units
        self._current_units = current_units
        self._solver = solver
        self._time_created = datetime.now()
        self.total_seconds = total_seconds

        self.tdgl_data = None
        self.state = None
        self._solve_step = None
        with h5py.File(self.path, "r") as f:
            self.device.mesh = Mesh.load_from_hdf5(f["mesh"])
        self.load_tdgl_data()

        # self.current_densities = current_densities
        # self.fields = fields
        # self.applied_field = applied_field
        # self.screening_fields = screening_fields
        # self.circulating_currents = circulating_currents or {}
        # self.terminal_currents = terminal_currents or {}

        # self._version_info = version_dict()

    def load_tdgl_data(self, solve_step: int = -1):
        with h5py.File(self.path, "r") as f:
            step_min, step_max = get_data_range(f)
            if solve_step == 0:
                step = step_min
            elif solve_step == -1:
                step = step_max
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
        J0 = self.device.J0.to(f"{self.current_units} / {self.device.length_units}")
        self.supercurrent_density = J0 * supercurrent[:, np.newaxis] * sc_direc
        self.normal_current_density = J0 * normal_current[:, np.newaxis] * nc_direc

        j_sc_site = get_observable_on_site(self.tdgl_data.supercurrent, mesh)
        j_nm_site = get_observable_on_site(self.tdgl_data.normal_current, mesh)
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
        scale = (
            device.J0 / (device.coherence_length * device.ureg(device.length_units))
        ).to(f"{self.current_units} / {self.device.length_units}**2")
        self.vorticity = vorticity * scale

    @property
    def current_density(self) -> pint.Quantity:
        if self.supercurrent_density is None:
            return None
        return self.supercurrent_density + self.normal_current_density

    @property
    def solve_step(self) -> int:
        return self._solve_step

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
            [hole.contains_points(xy) for hole in self.device.holes.values()]
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
        if not any(
            film.contains_points(points).all() for film in device.films.values()
        ):
            raise ValueError(
                "The polygon must lie completely within a superconducting film."
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
        # A_poly = self.applied_vector_potential(points[:, 0], points[:, 1], zs)[:, :2]
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
        # ns = 1
        Lambda = Lambda / ns * ureg(device.length_units)
        int_J = np.trapz((Lambda[:, np.newaxis] * J_poly * dl).sum(axis=1))
        supercurrent_part = (ureg("mu_0") * int_J).to(units)
        if not with_units:
            flux_part = flux_part.magnitude
            supercurrent_part = supercurrent_part.magnitude
        fluxoid = Fluxoid(flux_part, supercurrent_part)
        return fluxoid

    # def hole_fluxoid(
    #     self,
    #     hole_name: str,
    #     points: Optional[np.ndarray] = None,
    #     grid_shape: Union[int, Tuple[int, int]] = (200, 200),
    #     interp_method: str = "linear",
    #     units: Optional[str] = "Phi_0",
    #     with_units: bool = True,
    # ) -> Fluxoid:
    #     """Calculcates the fluxoid for a polygon enclosing the specified hole.

    #     Args:
    #         hole_name: The name of the hole for which to calculate the fluxoid.
    #         points: The vertices of the polygon enclosing the hole. If None is given,
    #             a polygon is generated using
    #             :func:`supercreen.fluxoid.make_fluxoid_polygons`.
    #         grid_shape: Shape of the desired rectangular grid to use for interpolation.
    #             If a single integer N is given, then the grid will be square,
    #             shape = (N, N).
    #         interp_method: Interpolation method to use.
    #         units: The desired units for the current density.
    #             Defaults to :math:`\\Phi_0`.
    #         with_units: Whether to return values as pint.Quantities with units attached.

    #     Returns:
    #         The hole's Fluxoid.
    #     """
    #     if points is None:
    #         from .fluxoid import make_fluxoid_polygons

    #         points = make_fluxoid_polygons(self.device, holes=hole_name)[hole_name]
    #     hole = self.device.holes[hole_name]
    #     if not in_polygon(points, hole.points).all():
    #         raise ValueError(
    #             f"Hole {hole_name} is not completely enclosed by the given polygon."
    #         )
    #     fluxoids = self.polygon_fluxoid(
    #         points,
    #         hole.layer,
    #         grid_shape=grid_shape,
    #         interp_method=interp_method,
    #         units=units,
    #         with_units=with_units,
    #     )
    #     return fluxoids[hole.layer]

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
        triangles = device.triangles
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
        weights = mass_matrix(points, triangles)
        # Compute the fields at the specified positions from the currents in each layer
        layer = self.device.layer
        if np.all((zs - layer.z0) == 0):
            for film in device.films.values():
                if film.layer == layer.name and film.contains_points(positions).any():
                    raise ValueError("Cannot interpolate fields within a layer.")
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
        triangles = device.triangles
        areas = mass_matrix(points, triangles)
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

    def to_file(
        self,
        directory: str,
        save_mesh: bool = True,
        compressed: bool = True,
        to_zip: bool = False,
    ) -> None:
        """Saves a Solution to disk.

        Args:
            directory: The name of the directory in which to save the solution
                (must either be empty or not yet exist).
            save_mesh: Whether to save the device mesh.
            compressed: Whether to use numpy.savez_compressed rather than numpy.savez.
            to_zip: Whether to save the Solution to a zip file.
        """
        if to_zip:
            from .io import zip_solution

            zip_solution(self, directory)
            return

        if os.path.isdir(directory) and len(os.listdir(directory)):
            raise IOError(f"Directory '{directory}' already exists and is not empty.")
        os.makedirs(directory, exist_ok=True)

        # Save device
        device_path = "device"
        self.device.to_file(os.path.join(directory, device_path), save_mesh=save_mesh)

        # Save arrays
        array_paths = []
        save_npz = np.savez_compressed if compressed else np.savez
        for layer in self.device.layers:
            path = f"{layer}_arrays.npz"
            save_npz(
                os.path.join(directory, path),
                streams=self.streams[layer],
                current_densities=self.current_densities[layer],
                fields=self.fields[layer],
                screening_fields=self.screening_fields[layer],
            )
            array_paths.append(path)

        # Save applied field function
        applied_field_path = "applied_field.dill"
        with open(os.path.join(directory, applied_field_path), "wb") as f:
            dill.dump(self.applied_field, f)

        # Handle circulating current formatting
        circ_currents = {}
        for name, val in self.circulating_currents.items():
            if isinstance(val, pint.Quantity):
                val = str(val)
            circ_currents[name] = val

        metadata = {
            "device": device_path,
            "arrays": array_paths,
            "applied_field": applied_field_path,
            "circulating_currents": circ_currents,
            "vortices": self.vortices,
            "field_units": self.field_units,
            "current_units": self.current_units,
            "solver": self.solver,
            "time_created": self.time_created.isoformat(),
            "version_info": self.version_info,
        }

        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def from_file(cls, directory: str, compute_matrices: bool = False) -> "Solution":
        """Loads a Solution from file.

        Args:
            directory: The directory from which to load the solution.
            compute_matrices: Whether to compute the field-independent
                matrices for the device if the mesh already exists.

        Returns:
            The loaded Solution instance
        """
        if directory.endswith(".zip") or zipfile.is_zipfile(directory):
            from .io import unzip_solution

            solution = unzip_solution(directory)
            if compute_matrices:
                solution.device.compute_matrices()
            return solution

        with open(os.path.join(directory, "metadata.json"), "r") as f:
            info = json.load(f)

        # Load device
        device_path = os.path.join(directory, info.pop("device"))
        device = Device.from_file(device_path, compute_matrices=compute_matrices)

        # Load arrays
        streams = {}
        current_densities = {}
        fields = {}
        screening_fields = {}
        array_paths = info.pop("arrays")
        for path in array_paths:
            layer = path.replace("_arrays.npz", "")
            with np.load(os.path.join(directory, path)) as arrays:
                streams[layer] = arrays["streams"]
                current_densities[layer] = arrays["current_densities"]
                fields[layer] = arrays["fields"]
                screening_fields[layer] = arrays["screening_fields"]

        # Load applied field function
        with open(os.path.join(directory, info.pop("applied_field")), "rb") as f:
            applied_field = dill.load(f)

        time_created = datetime.fromisoformat(info.pop("time_created"))
        version_info = info.pop("version_info", None)

        solution = cls(
            device=device,
            streams=streams,
            current_densities=current_densities,
            fields=fields,
            screening_fields=screening_fields,
            applied_field=applied_field,
            **info,
        )
        # Set "read-only" attributes
        solution._time_created = time_created
        solution._version_info = version_info

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
            and (self.circulating_currents == other.circulating_currents)
            and (
                getattr(self, "terminal_currents", None)
                == getattr(other, "terminal_currents", None)
            )
            and (self.applied_field == other.applied_field)
            and (self.vortices == other.vortices)
        ):
            return False
        if require_same_timestamp and (self.time_created != other.time_created):
            return False
        # Then check the arrays, which will take longer
        for name, array in self.streams.items():
            if not np.allclose(array, other.streams[name]):
                return False
        for name, array in self.current_densities.items():
            if not np.allclose(array, other.current_densities[name]):
                return False
        for name, array in self.fields.items():
            if not np.allclose(array, other.fields[name]):
                return False
        return True

    def __eq__(self, other) -> bool:
        return self.equals(other, require_same_timestamp=True)

    def plot_currents(self, **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for :func:`tdgl.visualization.plot_currents`."""
        from .visualization import plot_currents

        return plot_currents(self, **kwargs)

    def plot_order_parameter(self, **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for :func:`tdgl.visualization.plot_order_parameter`."""
        from .visualization import plot_order_parameter

        return plot_order_parameter(self, **kwargs)

    def plot_field_at_positions(
        self, points: np.ndarray, **kwargs
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for :func:`tdgl.visualization.plot_field_at_positions`."""
        from .visualization import plot_field_at_positions

        return plot_field_at_positions(self, points, **kwargs)

    def plot_vorticity(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alias for :func:`tdgl.visualization.plot_vorticity`."""
        from .visualization import plot_vorticity

        return plot_vorticity(self, **kwargs)
