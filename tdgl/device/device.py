import os
import json
import logging
from contextlib import contextmanager
from typing import Optional, Sequence, Union, List, Tuple, Dict, Any
import warnings

import h5py
import pint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from . import mesh
from .. import fem
from ..units import ureg
from .components import Layer, Polygon
from .._core.mesh.mesh import Mesh

logger = logging.getLogger(__name__)


class Device:
    """An object representing a device composed of multiple layers of
    thin film superconductor.

    Args:
        name: Name of the device.
        layers: ``Layers`` making up the device.
        films: ``Polygons`` representing regions of superconductor.
        holes: ``Holes`` representing holes in superconducting films.
        abstract_regions: ``Polygons`` representing abstract regions in a device.
            Abstract regions will be meshed, and one can calculate the flux through them.
        length_units: Distance units for the coordinate system.
        solve_dtype: The float data type to use when solving the device.
    """

    POLYGONS = (
        "films",
        "holes",
        "abstract_regions",
        "terminals",
    )

    ureg = ureg

    def __init__(
        self,
        name: str,
        *,
        layer: Layer,
        film: Polygon,
        source_terminal: Optional[Polygon] = None,
        drain_terminal: Optional[Polygon] = None,
        holes: Optional[Union[List[Polygon], Dict[str, Polygon]]] = None,
        abstract_regions: Optional[Union[List[Polygon], Dict[str, Polygon]]] = None,
        voltage_points: Sequence[float] = None,
        length_units: str = "um",
        solve_dtype: Union[str, np.dtype] = "float64",
    ):
        self.name = name

        self.source_terminal = source_terminal
        self.drain_terminal = drain_terminal
        if voltage_points is not None:
            voltage_points = np.asarray(voltage_points).squeeze()
            if voltage_points.shape != (2, 2):
                raise ValueError(
                    f"Voltage points must have shape (2, 2), "
                    f"got {voltage_points.shape}."
                )
        self.voltage_points = voltage_points

        self.layer = layer
        self._films_list = []
        self._holes_list = []
        self._abstract_regions_list = []
        self.layers_list = [layer]

        films = [film]

        if isinstance(films, dict):
            self.films = films
        else:
            self.films = {film.name: film for film in films}
        if holes is None:
            holes = []
        if isinstance(holes, dict):
            self.holes = holes
        else:
            self.holes = {hole.name: hole for hole in holes}
        if abstract_regions is None:
            abstract_regions = []
        if isinstance(abstract_regions, dict):
            self.abstract_regions = abstract_regions
        else:
            self.abstract_regions = {region.name: region for region in abstract_regions}
        for polygons, label in [(self._films_list, "film"), (self._holes_list, "hole")]:
            for polygon in polygons:
                if not polygon.is_valid:
                    raise ValueError(f"The following {label} is not valid: {polygon}.")
                if polygon.layer not in [self.layer.name]:
                    raise ValueError(
                        f"The following {label} is assigned to a layer that doesn not "
                        f"exist in the device: {polygon}."
                    )
        if self.source_terminal is not None:
            for terminal in [self.source_terminal, self.drain_terminal]:
                terminal.mesh = False
                terminal.layer = film.layer
            if self.source_terminal.name is None:
                self.source_terminal.name = "source"
            if self.drain_terminal.name is None:
                self.drain_terminal.name = "drain"

        if len(self.polygons) < (
            len(self._holes_list)
            + len(self._films_list)
            + len(self._abstract_regions_list)
        ):
            raise ValueError("All Polygons in a Device must have a unique name.")
        # Make units a "read-only" attribute.
        # It should never be changed after instantiation.
        self._length_units = length_units
        self.solve_dtype = solve_dtype

        self.mesh = None

    @property
    def coherence_length(self) -> float:
        return self.layer._coherence_length

    @coherence_length.setter
    def coherence_length(self, value: float) -> None:
        old_value = self.layer._coherence_length
        logger.debug(
            f"Updating {self.layer.name} coherence length from "
            f"{old_value:.3f} to {value:.3f} {self.length_units}."
        )
        if self.mesh is None:
            self.layer._coherence_length = value
            return
        logger.debug(
            "Rebuilding the dimensionless mesh with "
            f"coherence length = {value:.3f} {self.length_units}."
        )
        # Get points in {length_units}.
        points = self.points
        triangles = self.triangles
        self.layer._coherence_length = value
        if self.source_terminal is None:
            if self.drain_terminal is not None:
                raise ValueError(
                    "If source_terminal is None, drain_terminal must also be None."
                )
            input_edge = None
            output_edge = None
        else:
            if self.drain_terminal is None:
                raise ValueError(
                    "If source_terminal is not None, drain_terminal must also be"
                    " not None."
                )
            input_edge = self.source_terminal.contains_points(points, index=True)
            output_edge = self.drain_terminal.contains_points(points, index=True)
        # Make dimensionless mesh with the new coherence length.
        if self.voltage_points is None:
            voltage_points = None
        else:
            voltage_points = [
                np.argmin(np.linalg.norm(points - xy, axis=1))
                for xy in self.voltage_points
            ]
            voltage_points = np.array(voltage_points)
        self.mesh = Mesh.from_triangulation(
            points[:, 0] / self.coherence_length,
            points[:, 1] / self.coherence_length,
            triangles,
            input_edge=input_edge,
            output_edge=output_edge,
            voltage_points=voltage_points,
        )

    @property
    def kappa(self) -> float:
        return self.layer.london_lambda / self.coherence_length

    @property
    def J0(self) -> pint.Quantity:
        """Sheet current density scale (dimensions of current / length)."""
        length_units = ureg(self.length_units)
        xi = self.coherence_length * length_units
        # e = ureg("elementary_charge")
        # Phi_0 = ureg("Phi_0")
        mu_0 = ureg("mu_0")
        Lambda = self.layer.Lambda * length_units
        return (
            4
            * (
                xi
                * self.Bc2
                / (mu_0 * Lambda)
                # ureg("hbar") / (2 * mu_0 * e * xi * lambda_**2 / d)
                # Phi_0 / (2 * np.pi * mu_0 * xi * lambda_**2 / d)
                # np.pi * Phi_0 / (2 * np.pi * mu_0 * xi * lambda_**2 / d)
            ).to_base_units()
        )

    @property
    def Bc2(self) -> pint.Quantity:
        """Upper critical field."""
        xi_ = self.coherence_length * ureg(self.length_units)
        return (ureg("Phi_0") / (2 * np.pi * xi_**2)).to_base_units()

    @property
    def film(self) -> Polygon:
        return self._films_list[0]

    @property
    def terminals(self) -> Dict[str, Polygon]:
        if self.source_terminal is None:
            return {}
        return {
            self.source_terminal.name: self.source_terminal,
            self.drain_terminal.name: self.drain_terminal,
        }

    @property
    def points(self) -> Optional[np.ndarray]:
        if self.mesh is None:
            return None
        return self.coherence_length * np.array([self.mesh.x, self.mesh.y]).T

    @property
    def triangles(self) -> Optional[np.ndarray]:
        if self.mesh is None:
            return None
        return self.mesh.elements

    @property
    def length_units(self) -> str:
        """Length units used for the device geometry."""
        return self._length_units

    @property
    def solve_dtype(self) -> np.dtype:
        """Numpy dtype to use for floating point numbers."""
        return self._solve_dtype

    @solve_dtype.setter
    def solve_dtype(self, dtype) -> None:
        try:
            _ = np.finfo(dtype)
        except ValueError as e:
            raise ValueError(f"Invalid float dtype: {dtype}") from e
        self._solve_dtype = np.dtype(dtype)

    @staticmethod
    def _validate_polygons(polygons: List[Polygon], label: str) -> List[Polygon]:
        for polygon in polygons:
            if not polygon.is_valid:
                raise ValueError(f"The following {label} is not valid: {polygon}.")
        return polygons

    @property
    def films(self) -> Dict[str, Polygon]:
        """Dict of ``{film_name: film_polygon}``"""
        return {film.name: film for film in self._films_list}

    @films.setter
    def films(self, films_dict: Dict[str, Polygon]) -> None:
        """Dict of ``{film_name: film_polygon}``"""
        if not (
            isinstance(films_dict, dict)
            and all(isinstance(obj, Polygon) for obj in films_dict.values())
        ):
            raise TypeError("Films must be a dict of {film_name: Polygon}.")
        for name, polygon in films_dict.items():
            polygon.name = name
        self._films_list = list(self._validate_polygons(films_dict.values(), "film"))

    @property
    def holes(self) -> Dict[str, Polygon]:
        """Dict of ``{hole_name: hole_polygon}``"""
        return {hole.name: hole for hole in self._holes_list}

    @holes.setter
    def holes(self, holes_dict: Dict[str, Polygon]) -> None:
        """Dict of ``{hole_name: hole_polygon}``"""
        if not (
            isinstance(holes_dict, dict)
            and all(isinstance(obj, Polygon) for obj in holes_dict.values())
        ):
            raise TypeError("Holes must be a dict of {hole_name: Polygon}.")
        for name, polygon in holes_dict.items():
            polygon.name = name
        self._holes_list = list(self._validate_polygons(holes_dict.values(), "hole"))

    @property
    def abstract_regions(self) -> Dict[str, Polygon]:
        """Dict of ``{region_name: region_polygon}``"""
        return {region.name: region for region in self._abstract_regions_list}

    @abstract_regions.setter
    def abstract_regions(self, regions_dict: Dict[str, Polygon]) -> None:
        """Dict of ``{region_name: region_polygon}``"""
        if not (
            isinstance(regions_dict, dict)
            and all(isinstance(obj, Polygon) for obj in regions_dict.values())
        ):
            raise TypeError(
                "Abstract regions must be a dict of {region_name: Polygon}."
            )
        for name, polygon in regions_dict.items():
            polygon.name = name
        self._abstract_regions_list = list(
            self._validate_polygons(regions_dict.values(), "abstract region")
        )

    @property
    def polygons(self) -> Dict[str, Polygon]:
        """A dict of ``{name: polygon}`` for all Polygons in the device."""
        polygons = {}
        for attr_name in self.POLYGONS:
            polygons.update(getattr(self, attr_name))
        return polygons

    @property
    def poly_points(self) -> np.ndarray:
        """Shape (n, 2) array of (x, y) coordinates of all Polygons in the Device."""
        points = np.concatenate(
            [poly.points for poly in self.polygons.values() if poly.mesh]
        )
        # Remove duplicate points to avoid meshing issues.
        # If you don't do this and there are duplicate points,
        # meshpy.triangle will segfault.
        _, ix = np.unique(points, return_index=True, axis=0)
        points = points[np.sort(ix)]
        return points

    @property
    def vertex_distances(self) -> Optional[np.ndarray]:
        """An array of the mesh vertex-to-vertex distances."""
        if self.mesh is None:
            return None
        return self.mesh.edge_mesh.edge_lengths * self.coherence_length

    @property
    def triangle_areas(self) -> Optional[np.ndarray]:
        """An array of the mesh triangle areas."""
        if self.mesh is None:
            return None
        return fem.triangle_areas(self.points, self.triangles)

    def copy(self, with_arrays: bool = True, copy_arrays: bool = False) -> "Device":
        """Copy this Device to create a new one.
        Args:
            with_arrays: Whether to set the large arrays on the new Device.
            copy_arrays: Whether to create copies of the large arrays, or just
                return references to the existing arrays.
        Returns:
            A new Device instance, copied from self
        """
        holes = [hole.copy() for hole in self.holes.values()]
        abstract_regions = [region.copy() for region in self.abstract_regions.values()]
        if self.source_terminal is None:
            source = drain = None
        else:
            source = self.source_terminal.copy()
            drain = self.drain_terminal.copy()

        device = Device(
            self.name,
            layer=self.layer.copy(),
            film=self.film.copy(),
            holes=holes,
            abstract_regions=abstract_regions,
            source_terminal=source,
            drain_terminal=drain,
            length_units=self.length_units,
        )
        return device

    def _warn_if_mesh_exist(self, method: str) -> None:
        if self.mesh is None:
            return
        message = (
            f"Calling device.{method} on a device whose mesh already exists returns "
            f"a new device with no mesh. Call new_device.make_mesh() to generate the mesh "
            f"for the new device."
        )
        logger.warning(message)

    def scale(
        self, xfact: float = 1, yfact: float = 1, origin: Tuple[float, float] = (0, 0)
    ) -> "Device":
        """Returns a new device with polygons scaled horizontally and/or vertically.
        Negative ``xfact`` (``yfact``) can be used to reflect the device horizontally
        (vertically) about the ``origin``.
        Args:
            xfact: Factor by which to scale the device horizontally.
            yfact: Factor by which to scale the device vertically.
            origin: (x, y) coorindates of the origin.
        Returns:
            The scaled :class:`superscreen.device.device.Device`.
        """
        if not (
            isinstance(origin, tuple)
            and len(origin) == 2
            and all(isinstance(val, (int, float)) for val in origin)
        ):
            raise TypeError("Origin must be a tuple of floats (x, y).")
        self._warn_if_mesh_exist("scale()")
        device = self.copy(with_arrays=False)
        for polygon in device.polygons.values():
            polygon.scale(xfact=xfact, yfact=yfact, origin=origin, inplace=True)
        return device

    def rotate(self, degrees: float, origin: Tuple[float, float] = (0, 0)) -> "Device":
        """Returns a new device with polygons rotated a given amount
        counterclockwise about specified origin.
        Args:
            degrees: The amount by which to rotate the polygons.
            origin: (x, y) coorindates of the origin.
        Returns:
            The rotated :class:`superscreen.device.device.Device`.
        """
        if not (
            isinstance(origin, tuple)
            and len(origin) == 2
            and all(isinstance(val, (int, float)) for val in origin)
        ):
            raise TypeError("Origin must be a tuple of floats (x, y).")
        self._warn_if_mesh_exist("rotate()")
        device = self.copy(with_arrays=False)
        for polygon in device.polygons.values():
            polygon.rotate(degrees, origin=origin, inplace=True)
        return device

    def mirror_layer(self, about_z: float = 0.0) -> "Device":
        """Returns a new device with its layers mirrored about the plane
        ``z = about_z``.
        Args:
            about_z: The z-position of the plane (parallel to the x-y plane)
                about which to mirror the layers.
        Returns:
            The mirrored :class:`superscreen.device.device.Device`.
        """
        device = self.copy(with_arrays=True, copy_arrays=True)
        device.layer.z0 = about_z - device.layer.z0
        return device

    def translate(
        self,
        dx: float = 0,
        dy: float = 0,
        dz: float = 0,
        inplace: bool = False,
    ) -> "Device":
        """Translates the device polygons, layers, and mesh in space by a given amount.
        Args:
            dx: Distance by which to translate along the x-axis.
            dy: Distance by which to translate along the y-axis.
            dz: Distance by which to translate layers along the z-axis.
            inplace: If True, modifies the device (``self``) in-place and returns None,
                otherwise, creates a new device, translates it, and returns it.
        Returns:
            The translated device.
        """
        if inplace:
            device = self
        else:
            self._warn_if_mesh_exist("translate(..., inplace=False)")
            device = self.copy(with_arrays=True, copy_arrays=True)
        for polygon in device.polygons.values():
            polygon.translate(dx, dy, inplace=True)
        if device.points is not None:
            device.points += np.array([[dx, dy]])
        if dz:
            device.layer.z0 += dz
        return device

    @contextmanager
    def translation(self, dx: float, dy: float, dz: float = 0) -> None:
        """A context manager that temporarily translates a device in-place,
        then returns it to its original position.
        Args:
            dx: Distance by which to translate polygons along the x-axis.
            dy: Distance by which to translate polygons along the y-axis.
            dz: Distance by which to translate layers along the z-axis.
        """
        try:
            self.translate(dx, dy, dz=dz, inplace=True)
            yield
        finally:
            self.translate(-dx, -dy, dz=-dz, inplace=True)

    def make_mesh(
        self,
        # bounding_polygon: Optional[str] = None,
        compute_matrices: bool = True,
        convex_hull: bool = False,
        # weight_method: str = "half_cotangent",
        min_points: Optional[int] = None,
        optimesh_steps: Optional[int] = None,
        optimesh_method: str = "cvt-block-diagonal",
        optimesh_tolerance: float = 1e-3,
        optimesh_verbose: bool = False,
        **meshpy_kwargs,
    ) -> None:
        """Generates and optimizes the triangular mesh.
        Args:
            compute_matrices: Whether to compute the field-independent matrices
                (weights, Q, Laplace operator) needed for Brandt simulations.
            convex_hull: If True, mesh the entire convex hull of the device's polygons.
            weight_method: Weight methods for computing the Laplace operator:
                one of "uniform", "half_cotangent", or "inv_euclidian".
            min_points: Minimum number of vertices in the mesh. If None, then
                the number of vertices will be determined by meshpy_kwargs and the
                number of vertices in the underlying polygons.
            optimesh_steps: Maximum number of optimesh steps. If None, then no
                optimization is done.
            optimesh_method: Name of the optimization method to use.
            optimesh_tolerance: Optimesh quality tolerance.
            optimesh_verbose: Whether to use verbose mode in optimesh.
            **meshpy_kwargs: Passed to meshpy.triangle.build().
        """
        logger.info("Generating mesh...")
        # poly_points = self.poly_points
        boundary = self.polygons[self.film.name].points
        points, triangles = mesh.generate_mesh(
            boundary,
            hole_coords=[hole.points for hole in self.holes.values()],
            min_points=min_points,
            convex_hull=convex_hull,
            # boundary=boundary,
            **meshpy_kwargs,
        )
        if optimesh_steps:
            logger.info(f"Optimizing mesh with {points.shape[0]} vertices.")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    points, triangles = mesh.optimize_mesh(
                        points,
                        triangles,
                        optimesh_steps,
                        method=optimesh_method,
                        tolerance=optimesh_tolerance,
                        verbose=optimesh_verbose,
                    )
            except np.linalg.LinAlgError as e:
                err = (
                    "LinAlgError encountered in optimesh. Try reducing min_points "
                    "or increasing the number of points in the device's important polygons."
                )
                raise RuntimeError(err) from e
        logger.info(
            f"Finished generating mesh with {points.shape[0]} points and "
            f"{triangles.shape[0]} triangles."
        )
        if self.source_terminal is None:
            if self.drain_terminal is not None:
                raise ValueError(
                    "If source_terminal is None, drain_terminal must also be None."
                )
            input_edge = None
            output_edge = None
        else:
            if self.drain_terminal is None:
                raise ValueError(
                    "If source_terminal is not None, drain_terminal must also be"
                    " not None."
                )
            input_edge = self.source_terminal.contains_points(points, index=True)
            output_edge = self.drain_terminal.contains_points(points, index=True)

        if self.voltage_points is None:
            voltage_points = None
        else:
            voltage_points = [
                np.argmin(np.linalg.norm(points - xy, axis=1))
                for xy in self.voltage_points
            ]
            voltage_points = np.array(voltage_points)

        self.mesh = Mesh.from_triangulation(
            points[:, 0] / self.coherence_length,
            points[:, 1] / self.coherence_length,
            triangles,
            input_edge=input_edge,
            output_edge=output_edge,
            voltage_points=voltage_points,
        )

    def boundary_vertices(self) -> np.ndarray:
        """An array of boundary vertex indices, ordered counterclockwise.

        Returns:
            An array of indices for vertices that are on the device boundary,
            ordered counterclockwise.
        """
        if self.points is None:
            return None
        points = self.points
        indices = mesh.boundary_vertices(points, self.triangles)
        indices_list = indices.tolist()
        # Ensure that the indices wrap around outside of any terminals.
        boundary = self.points[indices]
        for term in [self.source_terminal, self.drain_terminal]:
            boundary = points[indices]
            term_ix = indices[term.contains_points(boundary)]
            discont = np.diff(term_ix) != 1
            if np.any(discont):
                i_discont = indices_list.index(term_ix[np.where(discont)[0][0]])
                indices = np.roll(indices, -(i_discont + 2))
                break
        return indices

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        legend: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        mesh: bool = False,
        mesh_kwargs: Dict[str, Any] = dict(color="k", lw=0.5),
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot all of the device's polygons.

        Args:
            ax: matplotlib axis on which to plot. If None, a new figure is created.
            subplots: If True, plots each layer on a different subplot.
            legend: Whether to add a legend.
            figsize: matplotlib figsize, only used if ax is None.
            mesh: If True, plot the mesh.
            mesh_kwargs: Keyword arguments passed to ``ax.triplot()``
                if ``mesh`` is True.
            kwargs: Passed to ``ax.plot()`` for the polygon boundaries.

        Returns:
            Matplotlib Figure and Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        points = self.points
        if mesh:
            if self.triangles is None:
                raise RuntimeError(
                    "Mesh does not exist. Run device.make_mesh() to generate the mesh."
                )
            x = points[:, 0]
            y = points[:, 1]
            tri = self.triangles
            ax.triplot(x, y, tri, **mesh_kwargs)
        for polygon in self.polygons.values():
            ax = polygon.plot(ax=ax, **kwargs)
        if self.mesh.voltage_points is not None:
            ax.plot(*points[self.mesh.voltage_points].T, "ko", label="Voltage points")
        if legend:
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
        units = self.ureg(self.length_units).units
        ax.set_xlabel(f"$x$ $[{units:~L}]$")
        ax.set_ylabel(f"$y$ $[{units:~L}]$")
        ax.set_aspect("equal")
        return fig, ax

    def patches(self) -> Dict[str, PathPatch]:
        """Returns a dict of ``{film_name: PathPatch}``
        for visualizing the device.
        """
        abstract_regions = self.abstract_regions
        holes = self.holes
        patches = dict()
        for polygon in self.polygons.values():
            if polygon.name in holes:
                continue
            coords = polygon.points.tolist()
            codes = [Path.LINETO for _ in coords]
            codes[0] = Path.MOVETO
            codes[-1] = Path.CLOSEPOLY
            poly = polygon.polygon
            for hole in holes.values():
                if polygon.name not in abstract_regions and poly.contains(hole.polygon):
                    hole_coords = hole.points.tolist()[::-1]
                    hole_codes = [Path.LINETO for _ in hole_coords]
                    hole_codes[0] = Path.MOVETO
                    hole_codes[-1] = Path.CLOSEPOLY
                    coords.extend(hole_coords)
                    codes.extend(hole_codes)
            patches[polygon.name] = PathPatch(Path(coords, codes))
        return patches

    def draw(
        self,
        ax: Optional[plt.Axes] = None,
        legend: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        alpha: float = 0.5,
        exclude: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """Draws all polygons in the device as matplotlib patches.
        Args:
            ax: matplotlib axis on which to plot. If None, a new figure is created.
            legend: Whether to add a legend.
            figsize: matplotlib figsize, only used if ax is None.
            alpha: The alpha (opacity) value for the patches (0 <= alpha <= 1).
            exclude: A polygon name or list of polygon names to exclude
                from the figure.
        Returns:
            Matplotlib Figre and Axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        exclude = exclude or []
        if isinstance(exclude, str):
            exclude = [exclude]
        patches = self.patches()
        units = self.ureg(self.length_units).units
        x, y = self.poly_points.T
        margin = 0.1
        dx = np.ptp(x)
        dy = np.ptp(y)
        x0 = x.min() + dx / 2
        y0 = y.min() + dy / 2
        dx *= 1 + margin
        dy *= 1 + margin
        labels = []
        handles = []
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlim(x0 - dx / 2, x0 + dx / 2)
        ax.set_ylim(y0 - dy / 2, y0 + dy / 2)
        ax.set_xlabel(f"$x$ $[{units:~L}]$")
        ax.set_ylabel(f"$y$ $[{units:~L}]$")
        for i, (name, patch) in enumerate(patches.items()):
            if name in exclude:
                continue
            patch.set_alpha(alpha)
            patch.set_color(f"C{i % 10}")
            ax.add_artist(patch)
            labels.append(name)
            handles.append(patch)
        if self.mesh.voltage_points is not None:
            (line,) = ax.plot(*self.points[self.mesh.voltage_points].T, "ko")
            handles.append(line)
            labels.append("Voltage points")
        if legend:
            ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left")
        return fig, ax

    def to_file(
        self, directory: str, save_mesh: bool = True, compressed: bool = True
    ) -> None:
        """Serializes the Device to disk.
        Args:
            directory: The name of the directory in which to save the Device
                (must either be empty or not yet exist).
            save_mesh: Whether to save the full mesh to file.
            compressed: Whether to use numpy.savez_compressed rather than numpy.savez
                when saving the mesh.
        """
        from ..io import NumpyJSONEncoder

        if os.path.isdir(directory) and len(os.listdir(directory)):
            raise IOError(f"Directory '{directory}' already exists and is not empty.")
        os.makedirs(directory, exist_ok=True)

        # Serialize films, holes, and abstract_regions to JSON
        polygons = {"device_name": self.name, "length_units": self.length_units}
        for poly_type in ["films", "holes", "abstract_regions"]:
            polygons[poly_type] = {}
            for name, poly in getattr(self, poly_type).items():
                polygons[poly_type][name] = {
                    "layer": poly.layer,
                    "points": poly.points,
                }
        with open(os.path.join(directory, "polygons.json"), "w") as f:
            json.dump(polygons, f, indent=4, cls=NumpyJSONEncoder)

        # Serialize layers to JSON.
        layers = {
            "device_name": self.name,
            "length_units": self.length_units,
            "solve_dtype": str(self.solve_dtype),
        }
        for layer in [self.layer]:
            name = layer.name
            layers[name] = {"z0": layer.z0, "thickness": layer.thickness}
            layers[name]["coherence_length"] = layer.coherence_length
            layers[name]["london_lambda"] = layer.london_lambda

        with open(os.path.join(directory, "layers.json"), "w") as f:
            json.dump(layers, f, indent=4, cls=NumpyJSONEncoder)

        if save_mesh and self.mesh is not None:
            # Serialize mesh, if it exists.
            with h5py.File(os.path.join(directory, "mesh.h5"), "w") as f:
                self.mesh.save_to_hdf5(f)

    @classmethod
    def from_file(cls, directory: str, compute_matrices: bool = False) -> "Device":
        """Creates a new Device from one serialized to disk.

        Args:
            directory: The directory from which to load the device.
            compute_matrices: Whether to compute the field-independent
                matrices for the device if the mesh already exists.

        Returns:
            The loaded Device instance
        """
        from ..io import json_numpy_obj_hook

        # Load all polygons (films, holes, abstract_regions)
        with open(os.path.join(directory, "polygons.json"), "r") as f:
            polygons_json = json.load(f, object_hook=json_numpy_obj_hook)

        device_name = polygons_json.pop("device_name")
        length_units = polygons_json.pop("length_units")
        films = {
            name: Polygon(name, **kwargs)
            for name, kwargs in polygons_json["films"].items()
        }
        holes = {
            name: Polygon(name, **kwargs)
            for name, kwargs in polygons_json["holes"].items()
        }
        abstract_regions = {
            name: Polygon(name, **kwargs)
            for name, kwargs in polygons_json["abstract_regions"].items()
        }

        # Load all layers
        with open(os.path.join(directory, "layers.json"), "r") as f:
            layers_json = json.load(f, object_hook=json_numpy_obj_hook)

        device_name = layers_json.pop("device_name")
        length_units = layers_json.pop("length_units")
        solve_dtype = layers_json.pop("solve_dtype", "float64")
        for name, layer_dict in layers_json.items():
            layers_json[name] = layer_dict

        layers = {name: Layer(name, **kwargs) for name, kwargs in layers_json.items()}

        device = cls(
            device_name,
            layer=list(layers.values())[0],
            film=list(films.values())[0],
            holes=holes,
            abstract_regions=abstract_regions,
            length_units=length_units,
            solve_dtype=solve_dtype,
        )

        # Load the mesh if it exists
        if "mesh.h5" in os.listdir(directory):
            with h5py.File(os.path.join(directory, "mesh.h5"), "r") as f:
                device.mesh = Mesh.load_from_hdf5(f)

        return device

    def __repr__(self) -> str:
        # Normal tab "\t" renders a bit too big in jupyter if you ask me.
        indent = 4
        t = " " * indent
        nt = "\n" + t

        # def format_dict(d):
        #     if not d:
        #         return None
        #     items = [f'{t}"{key}": {value}' for key, value in d.items()]
        #     return "{" + nt + (", " + nt).join(items) + "," + nt + "}"

        # args = [
        #     f'"{self.name}"',
        #     f"layers={format_dict(self.layers)}",
        #     f"films={format_dict(self.films)}",
        #     f"holes={format_dict(self.holes)}",
        #     f"abstract_regions={format_dict(self.abstract_regions)}",
        #     f'length_units="{self.length_units}"',
        # ]

        def format_list(L):
            if not L:
                return None
            items = [f"{t}{value}" for value in L]
            return "[" + nt + (", " + nt).join(items) + "," + nt + "]"

        args = [
            f'"{self.name}"',
            f"layer={self.layer}",
            f"films={format_list(self._films_list)}",
            f"holes={format_list(self._holes_list)}",
            f"abstract_regions={format_list(self._abstract_regions_list)}",
            f'length_units="{self.length_units}"',
        ]

        return f"{self.__class__.__name__}(" + nt + (", " + nt).join(args) + ",\n)"

    def __eq__(self, other) -> bool:
        if other is self:
            return True

        if not isinstance(other, Device):
            return False

        return (
            self.name == other.name
            and self.layer == other.layer
            and self.films == other.films
            and self.holes == other.holes
            and self.abstract_regions == other.abstract_regions
            and self.length_units == other.length_units
        )
