from typing import List, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ..geometry import close_curve
from .edge_mesh import EdgeMesh
from .util import (
    compute_voronoi_polygon_areas,
    convex_polygon_centroid,
    generate_voronoi_vertices,
    get_edges,
    get_voronoi_polygon_indices,
)


class Mesh:
    """A triangular mesh of a simply- or multiply-connected polygon.

    .. tip::

        Use :meth:`Mesh.from_triangulation` to create a new mesh from a triangulation.

    Args:
        sites: The (x, y) coordinates of the mesh vertices.
        elements: A list of triplets that correspond to the indices of he vertices that
            form a triangle. [[0, 1, 2], [0, 1, 3]] corresponds to a triangle
            connecting vertices 0, 1, and 2 and another triangle connecting vertices
            0, 1, and 3.
        boundary_indices: Indices corresponding to the boundary.
        areas: The areas corresponding to the sites.
        dual_sites: The (x, y) coordinates of the dual (Voronoi) mesh vertices
        edge_mesh: The edge mesh.
        voronoi_polygons: A list of Voronoi polygon vertices. There is one set of
            Voronoi polygon vertices for each mesh site.
    """

    def __init__(
        self,
        sites: Sequence[Tuple[float, float]],
        elements: Sequence[Tuple[int, int, int]],
        boundary_indices: Sequence[int],
        areas: Union[Sequence[float], None] = None,
        dual_sites: Union[Sequence[Tuple[float, float]], None] = None,
        edge_mesh: Union[EdgeMesh, None] = None,
        voronoi_polygons: Union[List[Sequence[Tuple[float, float]]], None] = None,
    ):
        self.sites = np.asarray(sites).squeeze()
        # Setting dtype to int64 is important when running on Windows.
        # Using default dtype uint64 does not work as Scipy indices in some
        # instances.
        self.elements = np.asarray(elements, dtype=np.int64)
        self.boundary_indices = np.asarray(boundary_indices, dtype=np.int64)
        if areas is not None:
            areas = np.asarray(areas)
        if dual_sites is not None:
            dual_sites = np.asarray(dual_sites)
        self.areas = areas
        self.dual_sites = dual_sites
        self.edge_mesh = edge_mesh
        self.voronoi_polygons = voronoi_polygons

    @property
    def x(self) -> np.ndarray:
        """The x-coordinates of the mesh sites."""
        return self.sites[:, 0]

    @property
    def y(self) -> np.ndarray:
        """The y-coordinates of the mesh sites."""
        return self.sites[:, 1]

    def closest_site(self, xy: Tuple[float, float]) -> int:
        """Returns the index of the mesh site closest to ``(x, y)``.

        Args:
            xy: A shape ``(2, )`` or ``(2, 1)`` sequence of floats, ``(x, y)``.

        Returns:
            The index of the mesh site closest to ``(x, y)``.
        """
        return np.argmin(np.linalg.norm(self.sites - np.atleast_2d(xy), axis=1))

    @staticmethod
    def from_triangulation(
        sites: Sequence[Tuple[float, float]],
        elements: Sequence[Tuple[int, int, int]],
        create_submesh: bool = True,
    ) -> "Mesh":
        """Create a triangular mesh from the coordinates of the triangle vertices
        and a list of indices corresponding to the vertices that connect to triangles.

        Args:
            sites: The (x, y) coordinates of the mesh sites.
            elements: A list of triplets that correspond to the indices of the vertices
                that form a triangle.   E.g. [[0, 1, 2], [0, 1, 3]] corresponds to a
                triangle connecting vertices 0, 1, and 2 and another triangle
                connecting vertices 0, 1, and 3.
            create_submesh: Whether to generate the corresponding
                :class:`tdgl.finit_volume.EdgeMesh` and Voronoi dual mesh.

        Returns:
            A new :class:`tdgl.finite_volume.Mesh` instance
        """
        sites = np.asarray(sites).squeeze()
        elements = np.asarray(elements).squeeze()
        if sites.ndim != 2 or sites.shape[1] != 2:
            raise ValueError(
                f"The site coordinates must have shape (n, 2), got {sites.shape!r}"
            )
        if elements.ndim != 2 or elements.shape[1] != 3:
            raise ValueError(
                f"The elements must have shape (m, 3), got {elements.shape!r}."
            )
        boundary_indices = Mesh.find_boundary_indices(elements)
        dual_sites = edge_mesh = polygons = areas = None
        if create_submesh:
            dual_sites = generate_voronoi_vertices(sites, elements)
            edge_mesh = EdgeMesh.from_mesh(sites, elements, dual_sites)
            areas, polygons = Mesh.compute_voronoi_areas_polygons(
                sites, elements, dual_sites, edge_mesh, boundary_indices
            )
        return Mesh(
            sites=sites,
            elements=elements,
            boundary_indices=boundary_indices,
            edge_mesh=edge_mesh,
            voronoi_polygons=polygons,
            dual_sites=dual_sites,
            areas=areas,
        )

    @staticmethod
    def find_boundary_indices(elements: np.ndarray) -> np.ndarray:
        """Find the boundary vertices.

        Args:
            elements: The triangular elements.

        Returns:
            An array of site indices corresponding to the boundary.
        """
        edges, is_boundary = get_edges(elements)
        # Get the boundary edges and all boundary points
        boundary_edges = edges[is_boundary]
        return np.unique(boundary_edges.flatten())

    @staticmethod
    def compute_voronoi_areas_polygons(
        sites: np.ndarray,
        elements: np.ndarray,
        dual_sites: np.ndarray,
        edge_mesh: EdgeMesh,
        boundary_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the area and indices of the Voronoi region for each vertex.

        Args:
            sites: The (x, y) coordinates of the mesh sites.
            elements: The mesh triangle indices.
            dual_sites: The (x, y) coordinates of the dual mesh vertices.
            edge_mesh: A :class:`tdgl.finite_volume.EdgeMesh` instance for the
                triangulation defined by ``sites`` and ``elements``.
            boundary_indices: The site indices corresponding to the boundary.

        Returns:
            The Voronoi cell areas and the counterclockwise-oriented vertices
            of the Voronoi cells.
        """
        # Compute polygons to use when computing area
        polygon_indices = get_voronoi_polygon_indices(elements, len(sites))
        # Get the areas for each vertex
        areas, voronoi_polygons = compute_voronoi_polygon_areas(
            sites=sites,
            dual_sites=dual_sites,
            boundary=boundary_indices,
            edges=edge_mesh.edges,
            boundary_edge_indices=edge_mesh.boundary_edge_indices,
            polygons=polygon_indices,
        )
        return areas, voronoi_polygons

    def get_quantity_on_site(
        self, quantity_on_edge: np.ndarray, vector: bool = True
    ) -> np.ndarray:
        """Compute the quantity on site by averaging over all edges
        connecting to each site.

        Args:
            quantity_on_edge: Observable on the edges.
            vector: Whether ``quantity_on_edge`` is a vector quantity.

        Returns:
            The quantity vector or scalar at each site.
        """
        # Normalize the edge direction
        directions = self.edge_mesh.directions
        normalized_directions = (
            directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        )
        if vector:
            flux_x = quantity_on_edge * normalized_directions[:, 0]
            flux_y = quantity_on_edge * normalized_directions[:, 1]
        else:
            flux_x = flux_y = quantity_on_edge
        # Sum x and y components for every edge connecting to the vertex
        vertices = np.concatenate(
            [self.edge_mesh.edges[:, 0], self.edge_mesh.edges[:, 1]]
        )
        x_values = np.concatenate([flux_x, flux_x])
        y_values = np.concatenate([flux_y, flux_y])
        counts = np.bincount(vertices)
        x_group_values = np.bincount(vertices, weights=x_values) / counts
        y_group_values = np.bincount(vertices, weights=y_values) / counts
        vector_val = np.array([x_group_values, y_group_values]).T / 2
        if vector:
            return vector_val
        return vector_val[:, 0]

    def smooth(self, iterations: int, create_submesh: bool = True) -> "Mesh":
        """Perform Laplacian smoothing of the mesh, i.e., moving each interior vertex
        to the arithmetic average of its neighboring points.

        Args:
            iterations: The number of smoothing iterations to perform.
            create_submesh: Whether to create the dual mesh and edge mesh.

        Returns:
            A new :class:`tdgl.finite_volume.Mesh` with relaxed vertex positions.
        """
        mesh = self
        elements = mesh.elements
        edges, _ = get_edges(elements)
        n = len(mesh.sites)
        shape = (n, 2)
        boundary = mesh.boundary_indices
        for i in range(iterations):
            sites = mesh.sites
            num_neighbors = np.bincount(edges.ravel(), minlength=shape[0])

            new_sites = np.zeros(shape)
            vals = sites[edges[:, 1]].T
            new_sites += np.array(
                [np.bincount(edges[:, 0], val, minlength=n) for val in vals]
            ).T
            vals = sites[edges[:, 0]].T
            new_sites += np.array(
                [np.bincount(edges[:, 1], val, minlength=n) for val in vals]
            ).T
            new_sites /= num_neighbors[:, np.newaxis]
            # reset boundary points
            new_sites[boundary] = sites[boundary]
            mesh = Mesh.from_triangulation(
                new_sites,
                elements,
                create_submesh=(create_submesh and (i == (iterations - 1))),
            )
        return mesh

    def plot(
        self,
        ax: Union[plt.Axes, None] = None,
        show_sites: bool = True,
        show_edges: bool = False,
        show_dual_edges: bool = True,
        show_voronoi_centroids: bool = False,
        site_color: Union[str, Sequence[float], None] = None,
        edge_color: Union[str, Sequence[float], None] = "k",
        centroid_color: Union[str, Sequence[float], None] = None,
        dual_edge_color: Union[str, Sequence[float], None] = "k",
        linewidth: float = 0.75,
        linestyle: str = "-",
        marker: str = ".",
    ) -> plt.Axes:
        """Plot the mesh.

        Args:
            ax: A :class:`plt.Axes` instance on which to plot the mesh.
            show_sites: Whether to show the mesh sites.
            show_edges: Whether to show the mesh edges.
            show_dual_edges: Whether to show the dual mesh edges.
            show_voronoi_centroids: Whether to show the centroid of each Voronoi cell.
            site_color: The color for the sites.
            edge_color: The color for the edges.
            dual_edge_color: The color for the dual edges.
            centroid_color: The color for the Voronoi centroids.
            linewidth: The line width for all edges.
            linestyle: The line style for all edges.
            marker: The marker to use for the mesh sites and Voronoi centroids.

        Returns:
            The resulting :class:`plt.Axes`
        """

        if ax is None:
            _, ax = plt.subplots()
        ax.set_aspect("equal")

        x, y = self.sites.T
        tri = self.elements

        if show_edges:
            ax.triplot(x, y, tri, color=edge_color, ls=linestyle, lw=linewidth)
        if show_dual_edges:
            for poly in self.voronoi_polygons:
                ax.plot(
                    *close_curve(poly).T,
                    color=dual_edge_color,
                    ls=linestyle,
                    lw=linewidth,
                )
        if show_sites:
            ax.plot(x, y, marker=marker, ls="", color=site_color)
        if show_voronoi_centroids:
            centroids = [convex_polygon_centroid(p) for p in self.voronoi_polygons]
            ax.plot(*np.array(centroids).T, marker=marker, ls="", color=centroid_color)

        return ax

    def to_hdf5(self, h5group: h5py.Group, compress: bool = False) -> None:
        """Save the mesh to a :class:`h5py.Group`.

        Args:
            h5group: The :class:`h5py.Group` into which to store the mesh.
            compress: If ``True``, store only the sites and elements.
        """
        h5group["sites"] = self.sites
        h5group["elements"] = self.elements
        if not compress:
            h5group["boundary_indices"] = self.boundary_indices
            h5group["areas"] = self.areas
            self.edge_mesh.to_hdf5(h5group.create_group("edge_mesh"))
            if self.dual_sites is not None:
                h5group["dual_sites"] = self.dual_sites
            # Save the Voronoi polygon vertices in a single shape (n, 2) array.
            # The ragged list of polygon vertices can be recovered by calling
            # np.split(polygons_flat, split_indices)
            split_indices = np.cumsum(
                [len(polygon) for polygon in self.voronoi_polygons[:-1]]
            )
            polygons_flat = np.concatenate(self.voronoi_polygons, axis=0)
            h5group["voronoi_polygons_flat"] = polygons_flat
            h5group["voronoi_split_indices"] = split_indices

    @staticmethod
    def from_hdf5(h5group: h5py.Group) -> "Mesh":
        """Load a mesh from an HDF5 file.

        Args:
            h5group: The HDF5 group to load the mesh from.

        Returns:
            The loaded mesh.
        """
        if not ("sites" in h5group and "elements" in h5group):
            raise IOError("Could not load mesh due to missing data.")

        if Mesh.is_restorable(h5group):
            polygons_flat = np.array(h5group["voronoi_polygons_flat"])
            voronoi_indices = np.array(h5group["voronoi_split_indices"])
            voronoi_polygons = np.split(polygons_flat, voronoi_indices)
            return Mesh(
                sites=np.array(h5group["sites"]),
                elements=np.array(h5group["elements"], dtype=np.int64),
                boundary_indices=np.array(h5group["boundary_indices"], dtype=np.int64),
                areas=np.array(h5group["areas"]),
                dual_sites=np.array(h5group["dual_sites"]),
                voronoi_polygons=voronoi_polygons,
                edge_mesh=EdgeMesh.from_hdf5(h5group["edge_mesh"]),
            )
        # Recreate mesh from triangulation data if not all data is available
        return Mesh.from_triangulation(
            sites=np.array(h5group["sites"]).squeeze(),
            elements=np.array(h5group["elements"]),
        )

    @staticmethod
    def is_restorable(h5group: h5py.Group) -> bool:
        """Returns ``True`` if the :class:`h5py.Group` contains all of the data
        necessary to create a :class:`tdgl.finite_volume.Mesh` without re-computing
        any values.

        Args:
            h5group: The :class:`h5py.Group` to check.

        Returns:
            Whether the mesh can be restored from the given group.
        """
        return (
            "sites" in h5group
            and "elements" in h5group
            and "boundary_indices" in h5group
            and "areas" in h5group
            and "edge_mesh" in h5group
            and "dual_sites" in h5group
            and "voronoi_polygons_flat" in h5group
            and "voronoi_split_indices" in h5group
        )
