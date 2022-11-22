import logging
import os
import warnings
from contextlib import contextmanager, nullcontext
from operator import attrgetter, itemgetter
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pint
from IPython.display import HTML
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely import affinity
from shapely.geometry import Point

from ..em import ureg
from ..finite_volume.mesh import Mesh
from . import mesh
from .layer import Layer
from .polygon import Polygon

logger = logging.getLogger(__name__)


class TerminalInfo(NamedTuple):
    """A containter for information about a single current terminal.

    Args:
        name: The terminal's name.
        site_indices: An array of indices for mesh sites making up the terminal.
        edge_indices: An array of indices for mesh edges making up the terminal.
        boundary_edge_indices: An array of indices of mesh boundary edges making
            up the terminal.
        length: The length of the terminal in physical units.
    """

    name: str
    site_indices: Sequence[int]
    edge_indices: Sequence[int]
    boundary_edge_indices: Sequence[int]
    length: float


class Device:
    """An object representing a device composed of multiple layers of
    thin film superconductor.

    Args:
        name: Name of the device.
        layer: The ``Layer`` making up the device.
        film: The ``Polygon`` representing the superconducting film.
        holes: ``Polygons`` representing holes in the superconducting film.
        abstract_regions: ``Polygons`` representing abstract regions in a device.
            Abstract regions are areas that can be meshed, but need not correspond
            to any physical structres in the device.
        terminals: A sequence of ``Polygosn`` representing the current terminals.
            Any points that are on the boundary of the mesh and lie inside a
            terminal will have current source/sink boundary conditions.
        voltage_points: A shape ``(2, 2)`` sequence of floats, with each row
            representing the ``(x, y)`` position of a voltage probe.
        length_units: Distance units for the coordinate system.
    """

    ureg = ureg

    def __init__(
        self,
        name: str,
        *,
        layer: Layer,
        film: Polygon,
        holes: Union[List[Polygon], None] = None,
        abstract_regions: Union[List[Polygon], None] = None,
        terminals: Union[List[Polygon], None] = None,
        voltage_points: Sequence[float] = None,
        length_units: str = "um",
    ):
        self.name = name
        self.layer = layer
        self.film = film
        self.holes = holes or []
        self.abstract_regions = abstract_regions or []
        self.terminals = tuple(terminals or [])
        if voltage_points is not None:
            voltage_points = np.asarray(voltage_points).squeeze()
            if voltage_points.shape != (2, 2):
                raise ValueError(
                    f"Voltage points must have shape (2, 2), "
                    f"got {voltage_points.shape}."
                )
        self.voltage_points = voltage_points

        names = set()
        for terminal in self.terminals:
            terminal.mesh = False
            if terminal.name is None or terminal.name in names:
                raise ValueError("All terminals must have a unique name")
            names.add(terminal.name)

        # Make units a "read-only" attribute.
        # It should never be changed after instantiation.
        self._length_units = length_units

        self.mesh = None
        self.mesh_info = None

    @property
    def length_units(self) -> str:
        """Length units used for the device geometry."""
        return self._length_units

    @property
    def coherence_length(self) -> float:
        """Ginzburg-Landau coherence length in ``length_units``."""
        return self.layer._coherence_length

    @coherence_length.setter
    def coherence_length(self, value: float) -> None:
        old_value = self.layer._coherence_length
        logger.debug(
            f"Updating coherence length from "
            f"{old_value:.3f} to {value:.3f} {self.length_units}."
        )
        if self.mesh is None:
            self.layer._coherence_length = value
            return
        logger.debug(
            f"Rebuilding the dimensionless mesh with "
            f"coherence length = {value:.3f} {self.length_units}."
        )
        # Get points in {length_units}.
        points = self.points
        triangles = self.triangles
        self.layer._coherence_length = value
        self._create_dimensionless_mesh(points, triangles)

    @property
    def kappa(self) -> float:
        """The Ginzburg-Landau parameter, :math:`\\kappa=\\lambda/\\xi`."""
        return self.layer.london_lambda / self.coherence_length

    @property
    def Bc2(self) -> pint.Quantity:
        """Upper critical field, :math:`B_{c2}=\\Phi_0/(2\\pi\\xi^2)`."""
        xi_ = self.coherence_length * ureg(self.length_units)
        return (ureg("Phi_0") / (2 * np.pi * xi_**2)).to_base_units()

    @property
    def A0(self) -> pint.Quantity:
        """Scale for the magnetic vector potential, :math:`A_0=\\xi B_{c2}`."""
        return (
            self.Bc2 * self.coherence_length * self.ureg(self.length_units)
        ).to_base_units()

    @property
    def K0(self) -> pint.Quantity:
        """Sheet current density scale (dimensions of current / length),
        :math:`K_0=4\\xi B_{c2}/(\\mu_0\\Lambda)`.
        """
        length_units = ureg(self.length_units)
        xi = self.coherence_length * length_units
        Lambda = self.layer.Lambda * length_units
        mu_0 = ureg("mu_0")
        K0 = 4 * xi * self.Bc2 / (mu_0 * Lambda)
        return K0.to_base_units()

    def terminal_info(self) -> Tuple[TerminalInfo, ...]:
        """Returns a tuple of ``TerminalInfo`` objects,
        one for each current terminal in the device.
        """
        xi = self.layer.coherence_length
        mesh = self.mesh
        sites = self.points
        edge_positions = xi * np.array([mesh.edge_mesh.x, mesh.edge_mesh.y]).T
        ix_boundary = mesh.edge_mesh.boundary_edge_indices
        edge_lengths = self.edge_lengths[ix_boundary]
        boundary_edge_positions = edge_positions[ix_boundary]
        info = []
        for terminal in self.terminals:
            # Index into self.points
            sites_index = np.intersect1d(
                terminal.contains_points(sites, index=True), mesh.boundary_indices
            )
            # Index into self.edges
            edges_index = np.intersect1d(
                terminal.contains_points(edge_positions, index=True), ix_boundary
            )
            # Index into self.edges[mesh.edge_mesh.boundary_edge_indices]
            boundary_edges_index = terminal.contains_points(
                boundary_edge_positions, index=True
            )
            length = edge_lengths[boundary_edges_index].sum()
            info.append(
                TerminalInfo(
                    terminal.name,
                    sites_index,
                    edges_index,
                    boundary_edges_index,
                    length,
                )
            )
        return tuple(sorted(info, key=attrgetter("length")))

    @property
    def polygons(self) -> Tuple[Polygon, ...]:
        """Tuple of all ``Polygons`` in the ``Device``."""
        return (
            self.terminals
            + (self.film,)
            + tuple(self.holes)
            + tuple(self.abstract_regions)
        )

    @property
    def points(self) -> Union[np.ndarray, None]:
        """The mesh vertex coordinates in ``length_units``
        (shape ``(n, 2)``, type ``float``).
        """
        if self.mesh is None:
            return None
        return self.coherence_length * self.mesh.sites

    @property
    def triangles(self) -> Union[np.ndarray, None]:
        """Mesh triangle indices (shape ``(m, 3)``, type ``int``)."""
        if self.mesh is None:
            return None
        return self.mesh.elements

    @property
    def edges(self) -> Union[np.ndarray, None]:
        """Mesh edge indices (shape ``(p, 2)``, type ``int``)."""
        if self.mesh is None:
            return None
        return self.mesh.edge_mesh.edges

    @property
    def edge_lengths(self) -> Union[np.ndarray, None]:
        """An array of the mesh vertex-to-vertex distances."""
        if self.mesh is None:
            return None
        return self.mesh.edge_mesh.edge_lengths * self.coherence_length

    @property
    def areas(self) -> Union[np.ndarray, None]:
        """An array of the mesh triangle areas."""
        if self.mesh is None:
            return None
        return self.mesh.areas * self.coherence_length**2

    @property
    def voltage_point_indices(self) -> Union[Tuple[int, int], None]:
        """A tuple of the mesh site indices for the voltage points."""
        if self.mesh is None or self.voltage_points is None:
            return None
        xi = self.coherence_length
        return tuple(self.mesh.closest_site(xy) for xy in self.voltage_points / xi)

    def contains_points(
        self,
        points: np.ndarray,
        index: bool = False,
        radius: float = 0,
    ) -> np.ndarray:
        """Determines whether ``points`` lie within the Device.

        Args:
            points: Shape ``(n, 2)`` array of x, y coordinates.
            index: If True, then return the indices of the points in ``points``
                that lie within the polygon. Otherwise, returns a shape ``(n, )``
                boolean array.
            radius: An additional margin on ``self.path``.
                See :meth:`matplotlib.path.Path.contains_points`.

        Returns:
            If index is True, returns the indices of the points in ``points``
            that lie within the polygon. Otherwise, returns a shape ``(n, )``
            boolean array indicating whether each point lies within the polygon.
        """
        mask = self.film.contains_points(points, radius=radius)
        mask = mask & ~np.logical_or.reduce(
            [hole.contains_points(points, radius=-radius) for hole in self.holes]
        )
        if index:
            return np.where(mask)[0]
        return mask

    @property
    def poly_points(self) -> np.ndarray:
        """A shape (n, 2) array of (x, y) coordinates of all Polygons in the Device."""
        points = np.concatenate(
            [self.film.points]
            + [poly.points for poly in self.abstract_regions if poly.mesh],
            axis=0,
        )
        # Remove duplicate points to avoid meshing issues.
        # If you don't do this and there are duplicate points,
        # meshpy.triangle will segfault.
        points = mesh.ensure_unique(points)
        return points

    def copy(self) -> "Device":
        """Copy this Device to create a new one.

        Note that the new Device is returned without a mesh.

        Returns:
            A new Device instance, copied from self.
        """
        holes = [hole.copy() for hole in self.holes]
        abstract_regions = [region.copy() for region in self.abstract_regions]
        terminals = [term.copy() for term in self.terminals]
        if self.voltage_points is None:
            voltage_points = None
        else:
            voltage_points = self.voltage_points.copy()

        device = Device(
            self.name,
            layer=self.layer.copy(),
            film=self.film.copy(),
            holes=holes,
            abstract_regions=abstract_regions,
            terminals=terminals,
            voltage_points=voltage_points,
            length_units=self.length_units,
        )
        return device

    def _warn_if_mesh_exist(self, method: str) -> None:
        if self.mesh is not None:
            message = (
                f"Calling device.{method} on a device whose mesh already exists "
                f"returns a new device with no mesh. Call new_device.make_mesh() "
                f"to generate the mesh for the new device."
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
            The scaled ``Device``.
        """
        if not (
            isinstance(origin, tuple)
            and len(origin) == 2
            and all(isinstance(val, (int, float)) for val in origin)
        ):
            raise TypeError("Origin must be a tuple of floats (x, y).")
        self._warn_if_mesh_exist("scale()")
        device = self.copy()
        for polygon in device.polygons:
            polygon.scale(xfact=xfact, yfact=yfact, origin=origin, inplace=True)
        if device.voltage_points is not None:
            points = [
                affinity.scale(Point(xy), xfact=xfact, yfact=yfact, origin=origin)
                for xy in device.voltage_points
            ]
            device.voltage_points = np.concatenate(
                [point.coords for point in points], axis=0
            )
        return device

    def rotate(self, degrees: float, origin: Tuple[float, float] = (0, 0)) -> "Device":
        """Returns a new device with polygons rotated a given amount
        counterclockwise about specified origin.

        Args:
            degrees: The amount by which to rotate the polygons.
            origin: (x, y) coorindates of the origin.

        Returns:
            The rotated ``Device``.
        """
        if not (
            isinstance(origin, tuple)
            and len(origin) == 2
            and all(isinstance(val, (int, float)) for val in origin)
        ):
            raise TypeError("Origin must be a tuple of floats (x, y).")
        self._warn_if_mesh_exist("rotate()")
        device = self.copy()
        for polygon in device.polygons:
            polygon.rotate(degrees, origin=origin, inplace=True)
        if self.voltage_points is not None:
            points = [
                affinity.rotate(Point(xy), degrees, origin=origin)
                for xy in self.voltage_points
            ]
            device.voltage_points = np.concatenate(
                [point.coords for point in points], axis=0
            )
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
            device = self.copy()
        for polygon in device.polygons:
            polygon.translate(dx, dy, inplace=True)
        if self.voltage_points is not None:
            device.voltage_points = self.voltage_points + np.array([[dx, dy]])
        if device.mesh is not None:
            points = device.points
            points += np.array([[dx, dy]])
            device._create_dimensionless_mesh(points, device.triangles)
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
        max_edge_length: Union[float, None] = None,
        min_points: Union[float, None] = None,
        optimesh_steps: Union[int, None] = None,
        optimesh_method: str = "cvt-block-diagonal",
        optimesh_tolerance: float = 1e-3,
        optimesh_verbose: bool = False,
        **meshpy_kwargs,
    ) -> None:
        """Generates and optimizes the triangular mesh.

        Args:
            min_points: Minimum number of vertices in the mesh. If None, then
                the number of vertices will be determined by meshpy_kwargs, the
                number of vertices in the underlying polygons, and ``max_edge_length``.
            max_edge_length: The maximum distance between vertices in the mesh.
                Passing a value <= 0 means that the number of mesh points will be
                determined solely by the density of points in the Device's film
                and abstract regions. Defaults to 1.5 * self.coherence_length.
            optimesh_steps: Maximum number of optimesh steps. If None, then no
                optimization is done.
            optimesh_method: Name of the optimization method to use.
            optimesh_tolerance: Optimesh quality tolerance.
            optimesh_verbose: Whether to use verbose mode in optimesh.
            **meshpy_kwargs: Passed to meshpy.triangle.build().
        """
        logger.info("Generating mesh...")
        boundary = self.film.points
        if max_edge_length is None:
            max_edge_length = 1.5 * self.coherence_length
        points, triangles = mesh.generate_mesh(
            self.poly_points,
            hole_coords=[hole.points for hole in self.holes],
            min_points=min_points,
            max_edge_length=max_edge_length,
            boundary=boundary,
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
        self._create_dimensionless_mesh(points, triangles)

    def _create_dimensionless_mesh(
        self, points: np.ndarray, triangles: np.ndarray
    ) -> Mesh:
        """Creates the dimensionless mesh.

        Args:
            points: Mesh vertices in ``length_units``.
            triangles: Mesh triangle indices.

        Returns:
            The dimensionless ``Mesh`` object.
        """
        self.mesh = Mesh.from_triangulation(
            points[:, 0] / self.coherence_length,
            points[:, 1] / self.coherence_length,
            triangles,
        )

    def mesh_stats_dict(self) -> Dict[str, Union[int, float, str]]:
        """Returns a dictionary of information about the mesh."""
        edge_lengths = self.edge_lengths
        areas = self.areas

        def _min(arr):
            if arr is not None:
                return arr.min()

        def _max(arr):
            if arr is not None:
                return arr.max()

        def _mean(arr):
            if arr is not None:
                return arr.mean()

        return dict(
            num_sites=len(self.mesh.x) if self.mesh else None,
            num_elements=len(self.mesh.elements) if self.mesh else None,
            min_edge_length=_min(edge_lengths),
            max_edge_length=_max(edge_lengths),
            mean_edge_length=_mean(edge_lengths),
            min_area=_min(areas),
            max_area=_max(areas),
            mean_area=_mean(areas),
            coherence_length=self.coherence_length,
            length_units=self.length_units,
        )

    def mesh_stats(self, precision: int = 3) -> HTML:
        """When called with in Jupyter notebook, displays
        a table of information about the mesh.

        Args:
            precision: Number of digits after the decimal for float values.

        Returns:
            An HTML table of mesh statistics.
        """
        stats = self.mesh_stats_dict()
        html = ["<table>", "<tr><b>Mesh Statistics</b></tr>"]
        for key, value in stats.items():
            if isinstance(value, float):
                value = f"{value:.{precision}e}"
            html.append(f"<tr><td><b>{key}</b></td><td>{value}</td></tr>")
        html.append("</table>")
        return HTML("".join(html))

    def plot(
        self,
        ax: Union[plt.Axes, None] = None,
        legend: bool = True,
        figsize: Union[Tuple[float, float], None] = None,
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
        voltage_points = self.voltage_point_indices
        if mesh:
            if self.mesh is None:
                raise RuntimeError(
                    "Mesh does not exist. Run device.make_mesh() to generate the mesh."
                )
            x = points[:, 0]
            y = points[:, 1]
            tri = self.triangles
            ax.triplot(x, y, tri, **mesh_kwargs)
        for polygon in self.polygons:
            ax = polygon.plot(ax=ax, **kwargs)
        if self.mesh is None and self.voltage_points is not None:
            ax.plot(*self.voltage_points.T, "ko", label="Voltage points")
        if voltage_points:
            ax.plot(*points[voltage_points, :].T, "ko", label="Voltage points")
        if legend:
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
        units = self.ureg(self.length_units).units
        ax.set_xlabel(f"$x$ $[{units:~L}]$")
        ax.set_ylabel(f"$y$ $[{units:~L}]$")
        ax.set_aspect("equal")
        return fig, ax

    def patches(self) -> Dict[str, PathPatch]:
        """Returns a dict of ``{polygon_name: PathPatch}``
        for visualizing the device.
        """
        abstract_regions = self.abstract_regions
        holes = self.holes
        patches = dict()
        for polygon in self.polygons:
            if polygon.name in holes:
                continue
            coords = polygon.points.tolist()
            codes = [Path.LINETO for _ in coords]
            codes[0] = Path.MOVETO
            codes[-1] = Path.CLOSEPOLY
            poly = polygon.polygon
            for hole in holes:
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
        ax: Union[plt.Axes, None] = None,
        legend: bool = True,
        figsize: Union[Tuple[float, float], None] = None,
        alpha: float = 0.5,
        exclude: Union[Union[str, List[str]], None] = None,
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
        voltage_points = self.voltage_point_indices
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
        if self.mesh is None and self.voltage_points is not None:
            (line,) = ax.plot(*self.voltage_points.T, "ko", label="Voltage points")
            handles.append(line)
            labels.append("Voltage points")
        if voltage_points:
            (line,) = ax.plot(*self.points[voltage_points, :].T, "ko")
            handles.append(line)
            labels.append("Voltage points")
        if legend:
            ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left")
        return fig, ax

    def to_hdf5(
        self,
        path_or_group: Union[str, h5py.File, h5py.Group],
        save_mesh: bool = True,
    ) -> None:
        """Serializes the Device to disk.

        Args:
            path_or_group: A path to an HDF5 file, or an open HDF5 file or group
                into which to save the ``Device``.
            save_mesh: Whether to serialize the full mesh.
        """
        if isinstance(path_or_group, str):
            path = path_or_group
            if not path.endswith(".h5"):
                path = path + ".h5"
            if os.path.exists(path):
                raise IOError(f"Path already exists: {path}.")
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            save_context = h5py.File(path, "w-", libver="latest")
        else:
            h5_group = path_or_group
            save_context = nullcontext(h5_group)
        with save_context as f:
            f.attrs["name"] = self.name
            f.attrs["length_units"] = self.length_units
            self.layer.to_hdf5(f.create_group("layer"))
            self.film.to_hdf5(f.create_group("film"))
            for terminal in self.terminals:
                terminals_grp = f.require_group("terminals")
                terminal.to_hdf5(terminals_grp.create_group(terminal.name))
            if self.voltage_points is not None:
                f["voltage_points"] = self.voltage_points
            for label, polygons in dict(
                holes=self.holes, abstract_regions=self.abstract_regions
            ).items():
                if polygons:
                    group = f.create_group(label)
                    for i, polygon in enumerate(polygons):
                        polygon.to_hdf5(group.create_group(str(i)))
            if save_mesh and self.mesh is not None:
                self.mesh.save_to_hdf5(f.create_group("mesh"))

    @classmethod
    def from_hdf5(cls, path_or_group: Union[str, h5py.File, h5py.Group]) -> "Device":
        """Creates a new Device from one serialized to disk.

        Args:
            path_or_group: A path to an HDF5 file, or an open HDF5 file or group
                containing the serialized Device.

        Returns:
            The loaded Device instance.
        """
        if isinstance(path_or_group, str):
            h5_context = h5py.File(path_or_group, "r", libver="latest")
        else:
            if not isinstance(path_or_group, (h5py.File, h5py.Group)):
                raise TypeError(
                    f"Expected an h5py.File or h5py.Group, but got "
                    f"{type(path_or_group)}."
                )
            h5_context = nullcontext(path_or_group)
        terminals = voltage_points = None
        holes = abstract_regions = mesh = None
        with h5_context as f:
            name = f.attrs["name"]
            length_units = f.attrs["length_units"]
            layer = Layer.from_hdf5(f["layer"])
            film = Polygon.from_hdf5(f["film"])
            if "terminals" in f:
                terminals = []
                for grp in f["terminals"].values():
                    terminals.append(Polygon.from_hdf5(grp))
            if "holes" in f:
                holes = [
                    Polygon.from_hdf5(grp)
                    for _, grp in sorted(f["holes"].items(), key=itemgetter(0))
                ]
            if "abstract_regions" in f:
                abstract_regions = [
                    Polygon.from_hdf5(grp)
                    for _, grp in sorted(
                        f["abstract_regions"].items(), key=itemgetter(0)
                    )
                ]
            if "voltage_points" in f:
                voltage_points = np.array(f["voltage_points"])
            if "mesh" in f:
                mesh = Mesh.load_from_hdf5(f["mesh"])

        device = Device(
            name,
            layer=layer,
            film=film,
            holes=holes,
            abstract_regions=abstract_regions,
            terminals=terminals,
            voltage_points=voltage_points,
            length_units=length_units,
        )

        if mesh is not None:
            device.mesh = mesh

        return device

    def __repr__(self) -> str:
        # Normal tab "\t" renders a bit too big in jupyter if you ask me.
        indent = 4
        t = " " * indent
        nt = "\n" + t

        args = [
            f"{self.name!r}",
            f"layer={self.layer!r}",
            f"film={self.film!r}",
            f"holes={self.holes!r}",
            f"abstract_regions={self.abstract_regions!r}",
            f"terminals={self.terminals!r}",
            f"voltage_points={self.voltage_points!r}",
            f"length_units={self.length_units!r}",
        ]

        return f"{self.__class__.__name__}(" + nt + (", " + nt).join(args) + ",\n)"

    def __eq__(self, other) -> bool:
        if other is self:
            return True

        if not isinstance(other, Device):
            return False

        def compare(seq1, seq2, key="name"):
            key = attrgetter(key)
            return sorted(seq1, key=key) == sorted(seq2, key=key)

        if self.voltage_points is None and other.voltage_points is None:
            same_voltage_points = True
        elif (
            isinstance(self.voltage_points, np.ndarray)
            and isinstance(other.voltage_points, np.ndarray)
            and np.allclose(self.voltage_points, other.voltage_points)
        ):
            same_voltage_points = True
        else:
            same_voltage_points = False

        return (
            self.name == other.name
            and self.layer == other.layer
            and self.film == other.film
            and compare(self.holes, other.holes)
            and compare(self.abstract_regions, other.abstract_regions)
            and compare(self.terminals, other.terminals)
            and same_voltage_points
            and self.length_units == other.length_units
        )
