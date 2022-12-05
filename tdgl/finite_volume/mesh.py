from typing import List, Sequence, Tuple, Union

import h5py
import numpy as np

from .edge_mesh import EdgeMesh
from .util import (
    compute_surrounding_area,
    generate_voronoi_vertices,
    get_edges,
    get_surrounding_voronoi_polygons,
    orient_convex_polygon_vertices,
)


class Mesh:
    """A triangular mesh of a simply- or multiply-connected polygon.

    .. note::

        Use :meth:`Mesh.from_triangulation` to create a new mesh.
        The ``__init__`` constructor requires that all parameters to be known.

    Args:
        x: The x coordinates for the triangle vertices.
        y: The x coordinates for the triangle vertices.
        elements: A list of triplets that correspond to the indices of he vertices that
            form a triangle. [[0, 1, 2], [0, 1, 3]] corresponds to a triangle
            connecting vertices 0, 1, and 2 and another triangle connecting vertices
            0, 1, and 3.
        boundary_indices: Indices corresponding to the boundary.
        areas: The areas corresponding to the sites.
        x_dual: The x coordinates of the dual mesh vertices.
        y_dual: The y coordinates of the dual mesh vertices.
        edge_mesh: The edge mesh.
    """

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        elements: Sequence[Tuple[int, int, int]],
        boundary_indices: Sequence[int],
        areas: Union[Sequence[float], None] = None,
        voronoi_polygons: Union[List[Sequence[int]], None] = None,
        x_dual: Union[Sequence[float], None] = None,
        y_dual: Union[Sequence[float], None] = None,
        edge_mesh: Union[EdgeMesh, None] = None,
    ):
        self.x = np.asarray(x).squeeze()
        self.y = np.asarray(y).squeeze()
        # Setting dtype to int64 is important when running on Windows.
        # Using default dtype uint64 does not work as Scipy indices in some
        # instances.
        self.elements = np.asarray(elements, dtype=np.int64)
        self.boundary_indices = np.asarray(boundary_indices, dtype=np.int64)
        if areas is not None:
            areas = np.asarray(areas)
        if x_dual is not None:
            x_dual = np.asarray(x_dual)
        if y_dual is not None:
            y_dual = np.asarray(y_dual)
        self.areas = areas
        self.x_dual = x_dual
        self.y_dual = y_dual
        self.edge_mesh = edge_mesh
        self.voronoi_polygons = voronoi_polygons
        if self.voronoi_polygons is not None and self.x_dual is not None:
            self.voronoi_polygons = [
                orient_convex_polygon_vertices(self.dual_sites, indices)
                for indices in self.voronoi_polygons
            ]

    @property
    def sites(self) -> np.ndarray:
        """The mesh sites as a shape ``(n, 2)`` array."""
        return np.array([self.x, self.y]).T

    @property
    def dual_sites(self) -> Union[np.ndarray, None]:
        """Returns the dual mesh sites (Voronoi polygon vertices) as
        a shape ``(m, 2)`` array.
        """
        if self.x_dual is None:
            return None
        return np.array([self.x_dual, self.y_dual]).T

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
        x: Sequence[float],
        y: Sequence[float],
        elements: Sequence[Tuple[int, int, int]],
        create_submesh: bool = True,
    ) -> "Mesh":
        """Create a triangular mesh from the coordinates of the triangle vertices
        and a list of indices corresponding to the vertices that connect to triangles.

        Args:
            x: The x coordinates for the triangle vertices.
            y: The x coordinates for the triangle vertices.
            elements: A list of triplets that correspond to the indices of the vertices
                that form a triangle.   E.g. [[0, 1, 2], [0, 1, 3]] corresponds to a
                triangle connecting vertices 0, 1, and 2 and another triangle
                connecting vertices 0, 1, and 3.
        """
        # Store the data
        x = np.asarray(x).squeeze()
        y = np.asarray(y).squeeze()
        elements = np.asarray(elements).squeeze()
        if x.ndim != 1:
            raise ValueError(
                "The x coordinates need to be stored in " "an one dimensional array."
            )
        if y.ndim != 1:
            raise ValueError(
                "The y coordinates need to be stored in " "an one dimensional array."
            )
        if np.size(x) != np.size(y):
            raise ValueError(
                "The number of x coordinates need to be equal to the "
                "number of y coordinates."
            )
        if elements.ndim != 2 or (elements.shape[0] != 3 and elements.shape[1] != 3):
            raise ValueError("The elements need to be a (n, 3)-vector.")
        boundary_indices = Mesh.find_boundary_indices(elements)
        x_dual = y_dual = edge_mesh = polygons = areas = None
        if create_submesh:
            x_dual, y_dual = generate_voronoi_vertices(x, y, elements)
            edge_mesh = EdgeMesh.from_mesh(x, y, elements, x_dual, y_dual)
            areas, polygons = Mesh.compute_voronoi_areas_polygons(
                x, y, elements, x_dual, y_dual, edge_mesh, boundary_indices
            )
        return Mesh(
            x=x,
            y=y,
            elements=elements,
            boundary_indices=boundary_indices,
            edge_mesh=edge_mesh,
            voronoi_polygons=polygons,
            x_dual=x_dual,
            y_dual=y_dual,
            areas=areas,
        )

    @staticmethod
    def find_boundary_indices(elements: np.ndarray) -> np.ndarray:
        """Find the boundary vertices.

        Args:
            elements: The triangular elements.

        Returns:
            A list of vertex indices corresponding to the boundary.
        """
        edges, is_boundary = get_edges(elements)
        # Get the boundary edges and all boundary points
        boundary_edges = edges[is_boundary]
        return np.unique(boundary_edges.flatten())

    @staticmethod
    def compute_voronoi_areas_polygons(
        x: np.ndarray,
        y: np.ndarray,
        elements: np.ndarray,
        x_dual: np.ndarray,
        y_dual: np.ndarray,
        edge_mesh: EdgeMesh,
        boundary_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the area and indices of the Voronoi region for each vertex."""

        # Compute polygons to use when computing area
        polygons = get_surrounding_voronoi_polygons(elements, len(x))
        # Get the areas for each vertex
        areas = compute_surrounding_area(
            x=x,
            y=y,
            boundary=boundary_indices,
            edges=edge_mesh.edges,
            boundary_edge_indices=edge_mesh.boundary_edge_indices,
            x_dual=x_dual,
            y_dual=y_dual,
            polygons=polygons,
        )
        return areas, polygons

    def get_observable_on_site(
        self, observable_on_edge: np.ndarray, vector: bool = True
    ) -> np.ndarray:
        """Compute the observable on site by averaging over all edges
        connecting to each site.

        Args:
            observable_on_edge: Observable on the edges.

        Returns:
            The observable vector or scalar at each site.
        """
        # Normalize the edge direction
        directions = self.edge_mesh.directions
        normalized_directions = (
            directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        )
        if vector:
            flux_x = observable_on_edge * normalized_directions[:, 0]
            flux_y = observable_on_edge * normalized_directions[:, 1]
        else:
            flux_x = flux_y = observable_on_edge
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
            A new :class:`tdgl.finit_volume.Mesh` with relaxed vertex positions.
        """
        mesh = self
        elements = mesh.elements
        edges, _ = get_edges(elements)
        n = mesh.x.shape[0]
        shape = (n, 2)
        boundary = mesh.boundary_indices
        for i in range(iterations):
            sites = mesh.sites
            num_neighbors = np.bincount(edges.ravel(), minlength=shape[0])

            new_points = np.zeros(shape)
            vals = sites[edges[:, 1]].T
            new_points += np.array(
                [np.bincount(edges[:, 0], val, minlength=n) for val in vals]
            ).T
            vals = sites[edges[:, 0]].T
            new_points += np.array(
                [np.bincount(edges[:, 1], val, minlength=n) for val in vals]
            ).T
            new_points /= num_neighbors[:, np.newaxis]
            # reset boundary points
            new_points[boundary] = sites[boundary]
            mesh = Mesh.from_triangulation(
                new_points[:, 0],
                new_points[:, 1],
                elements,
                create_submesh=(create_submesh and (i == (iterations - 1))),
            )
        return mesh

    def to_hdf5(self, h5group: h5py.Group, compress: bool = False) -> None:
        h5group["x"] = self.x
        h5group["y"] = self.y
        h5group["elements"] = self.elements
        if not compress:
            h5group["boundary_indices"] = self.boundary_indices
            h5group["areas"] = self.areas
            self.edge_mesh.to_hdf5(h5group.create_group("edge_mesh"))
            if self.x_dual is not None:
                h5group["x_dual"] = self.x_dual
            if self.y_dual is not None:
                h5group["y_dual"] = self.y_dual

    @staticmethod
    def from_hdf5(h5group: h5py.Group) -> "Mesh":
        """Load mesh from HDF5 file.

        Args:
            h5group: The HDF5 group to load the mesh from.

        Returns:
            The loaded mesh.
        """
        # Check that the required attributes x, y, and elements are in the group
        if not ("x" in h5group and "y" in h5group and "elements" in h5group):
            raise IOError("Could not load mesh due to missing data.")

        # Check if the mesh can be restored
        if Mesh.is_restorable(h5group):
            # Restore the mesh with the data
            return Mesh(
                x=h5group["x"],
                y=h5group["y"],
                elements=h5group["elements"],
                boundary_indices=h5group["boundary_indices"],
                areas=h5group["areas"],
                x_dual=h5group["x_dual"],
                y_dual=h5group["y_dual"],
                edge_mesh=EdgeMesh.from_hdf5(h5group["edge_mesh"]),
            )
        # Recreate mesh from triangulation data if not all data is available
        return Mesh.from_triangulation(
            x=np.asarray(h5group["x"]).flatten(),
            y=np.asarray(h5group["y"]).flatten(),
            elements=h5group["elements"],
        )

    @staticmethod
    def is_restorable(h5group: h5py.Group) -> bool:
        return (
            "x" in h5group
            and "y" in h5group
            and "elements" in h5group
            and "boundary_indices" in h5group
            and "areas" in h5group
            and "edge_mesh" in h5group
            and "x_dual" in h5group
            and "y_dual" in h5group
        )
