from typing import Sequence, Tuple

import h5py
import numpy as np

from .dual_mesh import DualMesh
from .edge_mesh import EdgeMesh
from .util import compute_surrounding_area, get_edges, get_surrounding_voronoi_polygons


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
        dual_mesh: The dual mesh.
        edge_mesh: The edge mesh.
    """

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        elements: Sequence[Tuple[int, int, int]],
        boundary_indices: Sequence[int],
        areas: Sequence[float],
        dual_mesh: DualMesh,
        edge_mesh: EdgeMesh,
    ):
        self.x = np.asarray(x).squeeze()
        self.y = np.asarray(y).squeeze()
        # Setting dtype to int64 is important when running on Windows.
        # Using default dtype uint64 does not work as Scipy indices in some
        # instances.
        self.elements = np.asarray(elements, dtype=np.int64)
        self.boundary_indices = np.asarray(boundary_indices, dtype=np.int64)
        self.dual_mesh = dual_mesh
        self.edge_mesh = edge_mesh
        self.areas = np.asarray(areas)

    @property
    def sites(self) -> np.ndarray:
        """The mesh sites as a shape ``(n, 2)`` array."""
        return np.array([self.x, self.y]).T

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
        dual_mesh = DualMesh.from_mesh(x, y, elements)
        edge_mesh = EdgeMesh.from_mesh(x, y, elements, dual_mesh)
        areas = Mesh.compute_voronoi_areas(
            x, y, elements, dual_mesh, edge_mesh, boundary_indices
        )
        return Mesh(
            x=x,
            y=y,
            elements=elements,
            boundary_indices=boundary_indices,
            edge_mesh=edge_mesh,
            dual_mesh=dual_mesh,
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
    def compute_voronoi_areas(
        x: np.ndarray,
        y: np.ndarray,
        elements: np.ndarray,
        dual_mesh: DualMesh,
        edge_mesh: EdgeMesh,
        boundary_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute the area of the Voronoi region for each vertex."""

        # Compute polygons to use when computing area
        polygons = get_surrounding_voronoi_polygons(elements, len(x))

        # Get the areas for each vertex
        return compute_surrounding_area(
            x=x,
            y=y,
            boundary=boundary_indices,
            edges=edge_mesh.edges,
            boundary_edge_indices=edge_mesh.boundary_edge_indices,
            x_dual=dual_mesh.x,
            y_dual=dual_mesh.y,
            polygons=polygons,
        )

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

    def to_hdf5(self, h5group: h5py.Group, compress: bool = False) -> None:
        h5group["x"] = self.x
        h5group["y"] = self.y
        h5group["elements"] = self.elements
        if not compress:
            h5group["boundary_indices"] = self.boundary_indices
            h5group["areas"] = self.areas
            self.edge_mesh.to_hdf5(h5group.create_group("edge_mesh"))
            self.dual_mesh.to_hdf5(h5group.create_group("dual_mesh"))

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
                dual_mesh=DualMesh.from_hdf5(h5group["dual_mesh"]),
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
            and "dual_mesh" in h5group
        )
