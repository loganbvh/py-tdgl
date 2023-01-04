import logging
import os
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
from ..finite_volume.util import get_oriented_boundary
from ..geometry import ensure_unique
from .layer import Layer
from .meshing import generate_mesh
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
    """An object representing a thin film superconducting device.

    Args:
        name: Name of the device.
        layer: The superconducting ``Layer``.
        film: The ``Polygon`` representing the superconducting film.
        holes: ``Polygons`` representing holes in the superconducting film.
        abstract_regions: ``Polygons`` representing abstract regions in a device.
            Abstract regions are areas that can be meshed, but need not correspond
            to any physical structres in the device.
        terminals: A sequence of ``Polygons`` representing the current terminals.
            Any points that are on the boundary of the mesh and lie inside a
            terminal will have current source/sink boundary conditions.
        probe_points: A shape ``(n, 2)`` sequence of floats, with each row
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
        probe_points: Sequence[float] = None,
        length_units: str = "um",
    ):
        self.name = name
        self.layer = layer
        self.film = film
        self.holes = holes or []
        self.abstract_regions = abstract_regions or []
        self.terminals = tuple(terminals or [])
        names = set()
        for terminal in self.terminals:
            terminal.mesh = False
            if terminal.name is None or terminal.name in names:
                raise ValueError("All terminals must have a unique name")
            names.add(terminal.name)

        for polygon in [self.film] + self.holes:
            if not polygon.is_valid:
                raise ValueError("Invalid Polygon: {polygon!r}.")

        if probe_points is not None:
            probe_points = np.asarray(probe_points).squeeze()
            if probe_points.ndim != 2 or probe_points.shape[1] != 2:
                raise ValueError(
                    f"Probe points must have shape (n, 2), "
                    f"got {probe_points.shape}."
                )
            if not self.contains_points(probe_points).all():
                raise ValueError("All probe points must lie within the film.")
        self.probe_points = probe_points

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
    def coherence_length(self) -> pint.Quantity:
        """Ginzburg-Landau coherence length, :math:`\\xi`"""
        return self.layer.coherence_length * ureg(self.length_units)

    @property
    def london_lambda(self) -> pint.Quantity:
        """London penetration depth, :math:`\\lambda`"""
        return self.layer.london_lambda * ureg(self.length_units)

    @property
    def thickness(self) -> pint.Quantity:
        """Film thickness, :math:`d`"""
        return self.layer.thickness * ureg(self.length_units)

    @property
    def Lambda(self) -> pint.Quantity:
        """Effective magnetic penetration depth, :math:`\\Lambda=\\lambda^2/d`."""
        return self.london_lambda**2 / self.thickness

    @property
    def conductivity(self) -> Union[pint.Quantity, None]:
        """Film normal state conductivity, :math:`\\sigma`"""
        if self.layer.conductivity is None:
            return None
        return self.layer.conductivity * ureg(f"siemens / {self.length_units}")

    @property
    def kappa(self) -> float:
        """The Ginzburg-Landau parameter, :math:`\\kappa=\\lambda/\\xi`."""
        return (self.london_lambda / self.coherence_length).magnitude

    @property
    def Bc2(self) -> pint.Quantity:
        """Upper critical field, :math:`B_{c2}=\\Phi_0/(2\\pi\\xi^2)`."""
        return (
            ureg("Phi_0") / (2 * np.pi * self.coherence_length**2)
        ).to_base_units()

    @property
    def A0(self) -> pint.Quantity:
        """Scale for the magnetic vector potential, :math:`A_0=\\xi B_{c2}`."""
        return (self.Bc2 * self.coherence_length).to_base_units()

    @property
    def K0(self) -> pint.Quantity:
        """Sheet current density scale (dimensions of current / length),
        :math:`K_0=4\\xi B_{c2}/(\\mu_0\\Lambda)`.
        """
        K0 = 4 * self.coherence_length * self.Bc2 / (ureg("mu_0") * self.Lambda)
        return K0.to_base_units()

    def tau0(self, conductivity: Union[pint.Quantity, None] = None) -> pint.Quantity:
        """Time scale, :math:`\\tau_0=\\mu_0\\sigma\\lambda^2`.

        Args:
            conductivity: The normal state conductivity of the film, which defaults
                to ``device.layer.conductivity``.

        Returns:
            The time scale, :math:`\\tau_0=\\mu_0\\sigma\\lambda^2
        """
        if conductivity is None:
            conductivity = self.conductivity
        if conductivity is None:
            raise ValueError(
                "The time scale tau0 requires the normal state"
                " conductivity to be defined."
            )
        return (ureg("mu_0") * conductivity * self.london_lambda**2).to("seconds")

    def V0(self, conductivity: Union[pint.Quantity, None] = None) -> pint.Quantity:
        """Electric potential scale, :math:`\\V_0=\\xi J_0/\\sigma`.

        Args:
            conductivity: The normal state conductivity of the film, which defaults
                to ``device.layer.conductivity``.

        Returns:
            The electric potential scale, :math:`\\V_0=\\xi J_0/\\sigma`
        """
        if conductivity is None:
            conductivity = self.conductivity
        if conductivity is None:
            raise ValueError(
                "The electric potential scale V_0 requires the normal state"
                " conductivity to be defined."
            )
        J0 = self.K0 / self.thickness
        return (self.coherence_length * J0 / conductivity).to("volts")

    def terminal_info(self) -> Tuple[TerminalInfo, ...]:
        """Returns a tuple of ``TerminalInfo`` objects,
        one for each current terminal in the device.
        """
        xi = self.layer.coherence_length
        mesh = self.mesh
        sites = self.points
        edge_positions = xi * mesh.edge_mesh.centers
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
            (self.film,)
            + tuple(self.holes)
            + tuple(self.abstract_regions)
            + self.terminals
        )

    @property
    def points(self) -> Union[np.ndarray, None]:
        """The mesh vertex coordinates in ``length_units``
        (shape ``(n, 2)``, type ``float``).
        """
        if self.mesh is None:
            return None
        return self.mesh.sites * self.coherence_length.magnitude

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
        return self.mesh.edge_mesh.edge_lengths * self.coherence_length.magnitude

    @property
    def areas(self) -> Union[np.ndarray, None]:
        """An array of the mesh Voronoi cell areas."""
        if self.mesh is None:
            return None
        return self.mesh.areas * self.coherence_length.magnitude**2

    @property
    def probe_point_indices(self) -> Union[List[int], None]:
        """A list of the mesh site indices for the probe points."""
        if self.mesh is None or self.probe_points is None:
            return None
        xi = self.coherence_length.magnitude
        return [self.mesh.closest_site(xy) for xy in self.probe_points / xi]

    def boundary_sites(self) -> Union[Dict[str, np.ndarray], None]:
        """Returns a dict of ``{polygon_name: boundary_indices}``, where ``boundary_indices``
        is an integer array of site indices for mesh sites on the boundary of each polygon.

        The length of the returned dictionary will be the number of holes in the device
        plus one.

        Returns:
            ``{polygon_name: boundary_indices}``
        """
        if self.mesh is None:
            return None
        polygons = [self.film] + list(self.holes)
        points = self.points
        edge_mesh = self.mesh.edge_mesh
        boundary_edges = edge_mesh.edges[edge_mesh.boundary_edge_indices]
        boundary = {}
        for polygon in polygons:
            on_boundary = np.logical_and(
                polygon.on_boundary(points[boundary_edges[:, 0]], radius=1e-6),
                polygon.on_boundary(points[boundary_edges[:, 1]], radius=1e-6),
            )
            boundary_sites = get_oriented_boundary(points, boundary_edges[on_boundary])
            assert len(boundary_sites) == 1, len(boundary_sites)
            boundary[polygon.name] = boundary_sites[0]
        return boundary

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
        mask = self.film.contains_points(points, radius=radius) & ~np.logical_or.reduce(
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
        return ensure_unique(points)

    def copy(self, with_mesh: bool = True) -> "Device":
        """Copy this Device to create a new one.

        Args:
            with_mesh: Whether to copy the mesh.

        Returns:
            A new Device instance, copied from self.
        """
        holes = [hole.copy() for hole in self.holes]
        abstract_regions = [region.copy() for region in self.abstract_regions]
        terminals = [term.copy() for term in self.terminals]
        if self.probe_points is None:
            probe_points = None
        else:
            probe_points = self.probe_points.copy()

        device = Device(
            self.name,
            layer=self.layer.copy(),
            film=self.film.copy(),
            holes=holes,
            abstract_regions=abstract_regions,
            terminals=terminals,
            probe_points=probe_points,
            length_units=self.length_units,
        )
        if with_mesh and self.mesh is not None:
            device.mesh = self.mesh
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
        device = self.copy(with_mesh=False)
        for polygon in device.polygons:
            polygon.scale(xfact=xfact, yfact=yfact, origin=origin, inplace=True)
        if device.probe_points is not None:
            points = [
                affinity.scale(Point(xy), xfact=xfact, yfact=yfact, origin=origin)
                for xy in device.probe_points
            ]
            device.probe_points = np.concatenate(
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
        device = self.copy(with_mesh=False)
        for polygon in device.polygons:
            polygon.rotate(degrees, origin=origin, inplace=True)
        if self.probe_points is not None:
            points = [
                affinity.rotate(Point(xy), degrees, origin=origin)
                for xy in self.probe_points
            ]
            device.probe_points = np.concatenate(
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
        """Translates the device polygons, layer, and mesh in space by a given amount.

        Args:
            dx: Distance by which to translate along the x-axis.
            dy: Distance by which to translate along the y-axis.
            dz: Distance by which to translate layer along the z-axis.
            inplace: If True, modifies the device (``self``) in-place and returns None,
                otherwise, creates a new device, translates it, and returns it.

        Returns:
            The translated device.
        """
        if inplace:
            device = self
        else:
            self._warn_if_mesh_exist("translate(..., inplace=False)")
            device = self.copy(with_mesh=False)
        for polygon in device.polygons:
            polygon.translate(dx, dy, inplace=True)
        if self.probe_points is not None:
            device.probe_points = self.probe_points + np.array([[dx, dy]])
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
        smooth: int = 0,
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
            smooth: Number of Laplacian smoothing iterations to perform.
            **meshpy_kwargs: Passed to meshpy.triangle.build().
        """
        logger.info("Generating mesh...")
        boundary = self.film.points
        if max_edge_length is None:
            max_edge_length = 1.5 * self.coherence_length.magnitude
        points, triangles = generate_mesh(
            self.poly_points,
            hole_coords=[hole.points for hole in self.holes],
            min_points=min_points,
            max_edge_length=max_edge_length,
            boundary=boundary,
            **meshpy_kwargs,
        )
        if smooth:
            mesh = Mesh.from_triangulation(
                points, triangles, create_submesh=False
            ).smooth(smooth, create_submesh=False)
            points = mesh.sites
            triangles = mesh.elements
        logger.info(
            f"Finished generating mesh with {len(points)} points and "
            f"{len(triangles)} triangles."
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
            points / self.coherence_length.magnitude,
            triangles,
            create_submesh=True,
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
            num_sites=len(self.mesh.sites) if self.mesh else None,
            num_elements=len(self.mesh.elements) if self.mesh else None,
            min_edge_length=_min(edge_lengths),
            max_edge_length=_max(edge_lengths),
            mean_edge_length=_mean(edge_lengths),
            min_area=_min(areas),
            max_area=_max(areas),
            mean_area=_mean(areas),
            coherence_length=self.coherence_length.magnitude,
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
        if self.probe_points is not None:
            ax.plot(*self.probe_points.T, "ko", label="Probe points")
        if legend:
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
        units = ureg(self.length_units).units
        ax.set_xlabel(f"$x$ $[{units:~L}]$")
        ax.set_ylabel(f"$y$ $[{units:~L}]$")
        ax.set_aspect("equal")
        return fig, ax

    def patches(self) -> Dict[str, PathPatch]:
        """Returns a dict of ``{polygon_name: PathPatch}``
        for visualizing the device.
        """
        abstract_regions = self.abstract_regions
        hole_names = {hole.name for hole in self.holes}
        patches = dict()
        for polygon in self.polygons:
            if polygon.name in hole_names:
                continue
            coords = polygon.points.tolist()
            codes = [Path.LINETO for _ in coords]
            codes[0] = Path.MOVETO
            codes[-1] = Path.CLOSEPOLY
            poly = polygon.polygon
            for hole in self.holes:
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
        patches = self.patches()
        units = ureg(self.length_units).units
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
        if self.probe_points is not None:
            (line,) = ax.plot(*self.probe_points.T, "ko", label="Probe points")
            handles.append(line)
            labels.append("Probe points")
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
            if self.probe_points is not None:
                f["probe_points"] = self.probe_points
            for hole in sorted(self.holes, key=attrgetter("name")):
                group = f.require_group("holes")
                hole.to_hdf5(group.create_group(hole.name))
            for i, polygon in enumerate(self.abstract_regions):
                group = f.require_group("abstract_regions")
                polygon.to_hdf5(group.create_group(str(i)))
            if save_mesh and self.mesh is not None:
                self.mesh.to_hdf5(f.create_group("mesh"))

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
        terminals = probe_points = None
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
            if "probe_points" in f:
                probe_points = np.array(f["probe_points"])
            if "mesh" in f:
                mesh = Mesh.from_hdf5(f["mesh"])

        device = Device(
            name,
            layer=layer,
            film=film,
            holes=holes,
            abstract_regions=abstract_regions,
            terminals=terminals,
            probe_points=probe_points,
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
            f"probe_points={self.probe_points!r}",
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

        if self.probe_points is None and other.probe_points is None:
            same_probe_points = True
        elif (
            isinstance(self.probe_points, np.ndarray)
            and isinstance(other.probe_points, np.ndarray)
            and np.allclose(self.probe_points, other.probe_points)
        ):
            same_probe_points = True
        else:
            same_probe_points = False

        return (
            self.name == other.name
            and self.layer == other.layer
            and self.film == other.film
            and compare(self.holes, other.holes)
            and compare(self.abstract_regions, other.abstract_regions)
            and compare(self.terminals, other.terminals)
            and same_probe_points
            and self.length_units == other.length_units
        )
