import logging
from typing import Iterable, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import path
from scipy import interpolate
from shapely import affinity
from shapely import geometry as geo
from shapely.validation import explain_validity

from ..finite_volume.mesh import Mesh
from ..geometry import close_curve, ensure_unique
from .meshing import generate_mesh

logger = logging.getLogger(__name__)


PolygonType = Union[
    "Polygon",
    np.ndarray,
    geo.linestring.LineString,
    geo.polygon.LinearRing,
    geo.polygon.Polygon,
]


class Polygon:
    """A polygonal region located in a :class:`tdgl.Layer`.

    Args:
        name: Name of the polygon.
        points: The polygon vertices. This can be a shape ``(n, 2)`` array of x, y
            coordinates or a shapely ``LineString``, ``LinearRing``, or ``Polygon``.
        mesh: Whether to include this polygon when computing a mesh.
    """

    def __init__(
        self,
        name: Union[str, None] = None,
        *,
        points: PolygonType,
        mesh: bool = True,
    ):
        self.name = name
        self.points = points
        self.mesh = mesh

    @property
    def points(self) -> np.ndarray:
        """A shape ``(n, 2)`` array of counter-clockwise-oriented polygon vertices."""
        return self._points

    @points.setter
    def points(self, points) -> None:
        geom_types = (
            geo.linestring.LineString,
            geo.polygon.LinearRing,
            geo.polygon.Polygon,
        )
        if isinstance(points, Polygon):
            points = points.points
        if not isinstance(points, geom_types):
            points = np.asarray(points)
        points = geo.polygon.Polygon(points)
        points = geo.polygon.orient(points)
        if points.interiors:
            raise ValueError("Expected a simply-connected polygon.")
        if not points.is_valid:
            reason = explain_validity(points)
            raise ValueError(
                "The given points do not define a valid polygon for the following "
                f"reason: {reason}."
            )
        points = close_curve(np.array(points.exterior.coords))
        if points.ndim != 2 or points.shape[-1] != 2:
            raise ValueError(f"Expected shape (n, 2), but got {points.shape}.")
        self._points = points

    @property
    def is_valid(self) -> bool:
        """True if the ``Polygon`` has a ``name`` and its geometry is valid."""
        polygon = self.polygon
        return self.name is not None and polygon.is_valid and not polygon.interiors

    @property
    def area(self) -> float:
        """The area of the polygon."""
        return self.polygon.area

    @property
    def bbox(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Returns the coordinates of the lower left and upper right corners
        of the polygon's bounding box.
        """
        minx, miny, maxx, maxy = self.polygon.bounds
        return (minx, miny), (maxx, maxy)

    @property
    def extents(self) -> Tuple[float, float]:
        """Returns the total x, y extent of the polygon, (Delta_x, Delta_y)."""
        minx, miny, maxx, maxy = self.polygon.bounds
        return (maxx - minx), (maxy - miny)

    @property
    def polygon(self) -> geo.polygon.Polygon:
        """A shapely ``Polygon`` representing the Polygon."""
        return geo.polygon.Polygon(self.points)

    @property
    def path(self) -> path.Path:
        """A matplotlib.path.Path representing the polygon boundary."""
        return path.Path(self.points, closed=True)

    def contains_points(
        self,
        points: np.ndarray,
        index: bool = False,
        radius: float = 0,
    ) -> Union[bool, np.ndarray]:
        """Determines whether ``points`` lie within the polygon.

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
        bool_array = self.path.contains_points(np.atleast_2d(points), radius=radius)
        if index:
            return np.where(bool_array)[0]
        return bool_array

    def on_boundary(
        self, points: np.ndarray, radius: float = 1e-3, index: bool = False
    ):
        """Determines whether ``points`` lie within a given radius of the Polygon
        boundary.

        Args:
            points: Shape ``(n, 2)`` array of x, y coordinates.
            radius: Points within ``radius`` of the boundary are considered
                to lie on the boundary.
            index: If True, then return the indices of the points in ``points``
                that lie on the boundary. Otherwise, returns a shape ``(n, )``
                boolean array.

        Returns:
            If index is True, returns the indices of the points in ``points``
            that lie within the polygon. Otherwise, returns a shape ``(n, )``
            boolean array indicating whether each point lies within the polygon.
        """
        points = np.atleast_2d(points)
        p = self.path
        in_outer = p.contains_points(points, radius=radius)
        in_inner = p.contains_points(points, radius=-radius)
        boundary = np.logical_and(in_outer, ~in_inner)
        if index:
            return np.where(boundary)[0]
        return boundary

    def make_mesh(
        self,
        min_points: Union[int, None] = None,
        smooth: int = 0,
        **meshpy_kwargs,
    ) -> Mesh:
        """Returns the vertices and triangles of a Delaunay mesh covering the Polygon.

        Args:
            min_points: Minimum number of vertices in the mesh. If None, then
                the number of vertices will be determined by meshpy_kwargs and the
                number of vertices in the underlying polygons.
            smooth: Number of Laplacian smoothing steps to perform.
            **meshpy_kwargs: Passed to meshpy.triangle.build().

        Returns:
            Mesh vertex coordinates and triangle indices
        """
        points, triangles = generate_mesh(
            self.points,
            min_points=min_points,
            convex_hull=False,
            **meshpy_kwargs,
        )
        if smooth:
            mesh = Mesh.from_triangulation(
                points, triangles, create_submesh=False
            ).smooth(smooth)
        else:
            mesh = Mesh.from_triangulation(points, triangles)
        logger.debug(
            f"Finished generating mesh with {len(mesh.sites)} points and "
            f"{len(mesh.elements)} triangles."
        )
        return mesh

    def rotate(
        self,
        degrees: float,
        origin: Union[str, Tuple[float, float]] = (0.0, 0.0),
        inplace: bool = False,
    ) -> "Polygon":
        """Rotates the polygon counterclockwise by a given angle.

        Args:
            degrees: The amount by which to rotate the polygon.
            origin: (x, y) coorindates about which to rotate, or the strings
                "center" (for the bounding box center) or "centroid"
                (for the polygon center of mass).
            inplace: If True, modify the polygon in place. Otherwise, return
                a modified copy.

        Returns:
            The rotated polygon.
        """
        polygon = self if inplace else self.copy()
        polygon.points = affinity.rotate(
            self.polygon, degrees, origin=origin, use_radians=False
        )
        return polygon

    def translate(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        inplace: bool = False,
    ) -> "Polygon":
        """Translates the polygon by a given distance.

        Args:
            dx: Distance by which to translate along the x-axis.
            dy: Distance by which to translate along the y-axis.
            inplace: If True, modify the polygon in place. Otherwise, return
                a modified copy.

        Returns:
            The translated polygon.
        """
        polygon = self if inplace else self.copy()
        polygon.points = affinity.translate(self.polygon, xoff=dx, yoff=dy)
        return polygon

    def scale(
        self,
        xfact: float = 1.0,
        yfact: float = 1.0,
        origin: Union[str, Tuple[float, float]] = (0, 0),
        inplace: bool = False,
    ) -> "Polygon":
        """Scales the polygon horizontally by ``xfact`` and vertically by ``yfact``.

        Negative ``xfact`` (``yfact``) can be used to reflect the polygon horizontally
        (vertically) about the ``origin``.

        Args:
            xfact: Distance by which to translate along the x-axis.
            yfact: Distance by which to translate along the y-axis.
            origin: (x, y) coorindates for the scaling origin, or the strings
                "center" (for the bounding box center) or "centroid"
                (for the polygon center of mass).
            inplace: If True, modify the polygon in place. Otherwise, return
                a modified copy.

        Returns:
            The scaled polygon.
        """
        polygon = self if inplace else self.copy()
        polygon.points = affinity.scale(
            self.polygon, xfact=xfact, yfact=yfact, origin=origin
        )
        return polygon

    def _join_via(
        self,
        other: PolygonType,
        operation: str,
    ) -> geo.polygon.Polygon:
        """Joins ``self.polygon`` with another polygon-like object
        via a given operation.
        """
        valid_types = (
            np.ndarray,
            geo.linestring.LineString,
            geo.polygon.LinearRing,
            geo.polygon.Polygon,
        )
        valid_operations = (
            "union",
            "intersection",
            "difference",
        )
        if operation not in valid_operations:
            raise ValueError(
                f"Unknown operation: {operation}. "
                f"Valid operations are {valid_operations}."
            )
        if isinstance(other, Polygon):
            other_poly = other.polygon
        elif isinstance(other, valid_types):
            other_poly = geo.polygon.Polygon(other)
        if not isinstance(other_poly, geo.polygon.Polygon):
            raise TypeError(
                f"Valid types are {(Polygon, ) + valid_types}, got {type(other)}."
            )
        joined = getattr(self.polygon, operation)(other_poly)
        reason = None
        if not isinstance(joined, geo.polygon.Polygon):
            reason = f"joined polygon has an unexpected type ({type(joined)})"
        elif joined.is_empty:
            reason = "joined polygon is empty"
        elif not joined.is_valid:
            reason = explain_validity(joined)
        if reason is not None:
            raise ValueError(
                f"The {operation} of the two polygons is not a valid polygon "
                f"for the following reason: {reason}."
            )
        return joined

    def union(
        self,
        *others: PolygonType,
        name: Union[str, None] = None,
    ) -> "Polygon":
        """Returns the union of the polygon and zero or more other polygons.

        Args:
            others: One or more objects with which to join the polygon.
            name: A name for the resulting joined Polygon (defaults to ``self.name``.)

        Returns:
            A new :class:`Polygon` instance representing the union
            of ``self`` and ``others``.
        """
        if not others:
            return self.copy()
        first, *rest = others
        return Polygon(
            name=name or self.name,
            points=self._join_via(first, "union"),
            mesh=self.mesh,
        ).union(*rest, name=name)

    def intersection(
        self,
        *others: PolygonType,
        name: Union[str, None] = None,
    ) -> "Polygon":
        """Returns the intersection of the polygon and zero or more other polygons.

        Args:
            others: One or more objects with which to join the polygon.
            name: A name for the resulting joined Polygon (defaults to ``self.name``.)

        Returns:
            A new :class:`Polygon` instance representing the intersection
            of ``self`` and ``others``.
        """
        if not others:
            return self.copy()
        first, *rest = others
        return Polygon(
            name=name or self.name,
            points=self._join_via(first, "intersection"),
            mesh=self.mesh,
        ).intersection(*rest, name=name)

    def difference(
        self,
        *others: PolygonType,
        name: Union[str, None] = None,
    ) -> "Polygon":
        """Returns the difference of the polygon and zero more other polygons.

        Args:
            others: One or more objects with which to join the polygon.
            name: A name for the resulting joined Polygon (defaults to ``self.name``.)

        Returns:
            A new :class:`Polygon` instance representing the difference
            of ``self`` and ``others``.

        .. _shapely documentation: https://shapely.readthedocs.io/en/stable/manual.html
        """
        if not others:
            return self.copy()
        first, *rest = others
        return Polygon(
            name=name or self.name,
            points=self._join_via(first, "difference"),
            mesh=self.mesh,
        ).difference(*rest, name=name)

    def __add__(self, other: PolygonType) -> "Polygon":
        return self.union(other)

    def __sub__(self, other: PolygonType) -> "Polygon":
        return self.difference(other)

    def __mul__(self, other: PolygonType) -> "Polygon":
        return self.intersection(other)

    def buffer(
        self,
        distance: float,
        join_style: Union[str, int] = "mitre",
        mitre_limit: float = 5.0,
        single_sided: bool = True,
        as_polygon: bool = True,
    ) -> Union[np.ndarray, "Polygon"]:
        """Returns polygon points or a new Polygon object with vertices offset from
        ``self.points`` by a given ``distance``. If ``distance > 0`` this "inflates"
        the polygon, and if ``distance < 0`` this shrinks the polygon.


        Args:
            distance: The amount by which to inflate or deflate the polygon.
            join_style: One of "round" (1), "mitre" (2), or "bevel" (3).
                See the `shapely documentation`_.
            mitre_limit: See the `shapely documentation`_.
            single_sided: See the `shapely documentation`_.
            as_polygon: If True, returns a new ``Polygon`` instance, otherwise
                returns a shape ``(n, 2)`` array of polygon vertices.

        Returns:
            A new ``Polygon`` or an array of vertices offset by ``distance``.

        .. _shapely documentation: https://shapely.readthedocs.io/en/stable/manual.html
        """
        if isinstance(join_style, str):
            join_style = getattr(geo.JOIN_STYLE, join_style)
        poly = self.polygon.buffer(
            distance,
            join_style=join_style,
            mitre_limit=mitre_limit,
            single_sided=single_sided,
        )

        polygon = Polygon(
            name=self.name,
            points=poly,
            mesh=self.mesh,
        )
        npts = max(len(polygon.points), len(self.points))
        polygon = polygon.resample(npts)
        if as_polygon:
            return polygon
        return polygon.points

    def resample(
        self, num_points: Union[int, None] = None, degree: int = 1, smooth: float = 0
    ) -> "Polygon":
        """Resample vertices so that they are approximately uniformly distributed
        along the polygon boundary.

        Args:
            num_points: Number of points to interpolate to. If ``num_points`` is None,
                the polygon is resampled to ``len(self.points)`` points. If
                ``num_points`` is not None and has a boolean value of False,
                then an unaltered copy of the polygon is returned.
            degree: The degree of the spline with which to iterpolate.
                Defaults to 1 (linear spline).
            smooth: Smoothing condition.

        """
        if num_points is None:
            num_points = len(self.points)
        if not num_points:
            return self.copy()
        points = ensure_unique(self.points.copy())
        tck, _ = interpolate.splprep(points.T, k=degree, s=smooth)
        x, y = interpolate.splev(np.linspace(0, 1, num_points - 1), tck)
        points = close_curve(np.array([x, y]).T)
        return Polygon(
            name=self.name,
            points=points,
            mesh=self.mesh,
        )

    def set_name(self, name: Union[str, None]) -> "Polygon":
        """Sets the Polygon's name and returns ``self``."""
        self.name = name
        return self

    def plot(self, ax: Union[plt.Axes, None] = None, **kwargs) -> plt.Axes:
        """Plots the Polygon's vertices.

        Args:
            ax: The matplotlib Axes on which to plot. If None is given, a new one
                is created.
            kwargs: Passed to ``ax.plot()``.

        Returns:
            The matplotlib Axes.
        """
        if ax is None:
            _, ax = plt.subplots()
        kwargs = kwargs.copy()
        kwargs["label"] = self.name
        ax.plot(*self.points.T, **kwargs)
        ax.set_aspect("equal")
        return ax

    @classmethod
    def from_union(
        cls,
        items: Iterable[PolygonType],
        *,
        name: Union[str, None] = None,
        mesh: bool = True,
    ) -> "Polygon":
        """Creates a new :class:`Polygon` from the union of a sequence of polygons.

        Args:
            items: A sequence of polygon-like objects to join.
            name: Name of the polygon.
            mesh: Whether to include this polygon when computing a mesh.

        Returns:
            A new :class:`Polygon`.
        """
        first, *rest = items
        polygon = cls(name=name, points=first, mesh=mesh)
        return polygon.union(*rest)

    @classmethod
    def from_intersection(
        cls,
        items: Iterable[PolygonType],
        *,
        name: Union[str, None] = None,
        mesh: bool = True,
    ) -> "Polygon":
        """Creates a new :class:`Polygon` from the intersection
        of a sequence of polygons.

        Args:
            items: A sequence of polygon-like objects to join.
            name: Name of the polygon.
            mesh: Whether to include this polygon when computing a mesh.

        Returns:
            A new :class:`Polygon`.
        """
        first, *rest = items
        polygon = cls(name=name, points=first, mesh=mesh)
        return polygon.intersection(*rest)

    @classmethod
    def from_difference(
        cls,
        items: Iterable[PolygonType],
        *,
        name: Union[str, None] = None,
        mesh: bool = True,
    ) -> "Polygon":
        """Creates a new :class:`Polygon` from the difference
        of a sequence of polygons.

        Args:
            items: A sequence of polygon-like objects to join.
            name: Name of the polygon.
            mesh: Whether to include this polygon when computing a mesh.

        Returns:
            A new :class:`Polygon`.
        """
        first, *rest = items
        polygon = cls(name=name, points=first, mesh=mesh)
        return polygon.difference(*rest)

    def to_hdf5(self, h5_group: h5py.Group) -> None:
        """Save the ``Polygon`` to an :class:`h5py.Group`."""
        if self.name is not None:
            h5_group.attrs["name"] = self.name
        h5_group.attrs["mesh"] = self.mesh
        h5_group["points"] = self.points

    @classmethod
    def from_hdf5(cls, h5_group: h5py.Group) -> "Polygon":
        """Load a ``Polygon`` from an :class:`h5py.Group`."""
        name = None
        if "name" in h5_group.attrs:
            name = h5_group.attrs["name"]
        return Polygon(
            name=name,
            points=np.array(h5_group["points"]),
            mesh=h5_group.attrs["mesh"],
        )

    def __repr__(self) -> str:
        name = f'"{self.name}"' if self.name is not None else None
        return (
            f"{self.__class__.__name__}(name={name}, "
            f"points=<ndarray: shape={self.points.shape}>, mesh={self.mesh})"
        )

    def __eq__(self, other) -> bool:
        if other is self:
            return True

        if not isinstance(other, Polygon):
            return False

        return self.name == other.name and np.allclose(self.points, other.points)

    def copy(self) -> "Polygon":
        """Returns a deep copy of the :class:`Polygon`"""
        return Polygon(
            self.name,
            points=self.points.copy(),
            mesh=self.mesh,
        )
