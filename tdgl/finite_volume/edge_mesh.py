from typing import Sequence, Tuple

import h5py
import numpy as np

from .util import get_dual_edge_lengths, get_edges


class EdgeMesh:
    """A mesh composed of the edges in a triangular mesh.

    .. tip::

        Use :meth:`EdgeMesh.from_mesh` to create from an existing mesh.

    Args:
        centers: The (x, y) coordinates for the edge_centers.
        edges: The edges as a sequence of indices.
        boundary_edge_indices: Edges on the boundary.
        directions: Directions of the edges.
        edge_lengths: Lengths of the edges.
        dual_edge_lengths: Length of the dual edges.
    """

    def __init__(
        self,
        centers: Sequence[Tuple[float, float]],
        edges: Sequence[Tuple[int, int]],
        boundary_edge_indices: Sequence[int],
        directions: Sequence[Tuple[float, float]],
        edge_lengths: Sequence[float],
        dual_edge_lengths: Sequence[float],
    ):
        self.centers = np.asarray(centers)
        self.edges = np.asarray(edges)
        self.boundary_edge_indices = np.asarray(boundary_edge_indices, dtype=np.int64)
        self.directions = np.asarray(directions)
        self.edge_lengths = np.asarray(edge_lengths)
        self.dual_edge_lengths = np.asarray(dual_edge_lengths)

    @property
    def x(self) -> np.ndarray:
        """The x-coordinates of the edge centers."""
        return self.centers[:, 0]

    @property
    def y(self) -> np.ndarray:
        """The y-coordinates of the edge centers."""
        return self.centers[:, 1]

    @staticmethod
    def from_mesh(
        sites: np.ndarray,
        elements: np.ndarray,
        dual_sites: np.ndarray,
    ) -> "EdgeMesh":
        """Create edge mesh from mesh.

        Args:
            sites: The (x, y) coordinates for the mesh vertices.
            elements: Elements for the mesh.
            dual_sites: The (xm y) coordinates for the dual mesh vertices.

        Returns:
            The edge mesh.
        """
        edges, is_boundary = get_edges(elements)
        # Get the indices of the boundary edges
        boundary_edge_indices = np.where(is_boundary)[0]
        # Shape (m, 2, 2), (sites, edges, spatial dimensions)
        edge_coords = sites[edges]
        edge_centers = edge_coords.mean(axis=1)
        directions = np.diff(edge_coords, axis=1).squeeze()
        edge_lengths = np.linalg.norm(directions, axis=1)
        dual_edge_lengths = get_dual_edge_lengths(
            edge_centers,
            elements,
            dual_sites,
            edges,
        )
        return EdgeMesh(
            edge_centers,
            edges,
            boundary_edge_indices,
            directions,
            edge_lengths,
            dual_edge_lengths,
        )

    def to_hdf5(self, h5group: h5py.Group) -> None:
        """Save the data to a HDF5 file.

        Args:
            h5group: The HDF5 group to write the data to.
        """
        h5group["centers"] = self.centers
        h5group["edges"] = self.edges
        h5group["boundary_edge_indices"] = self.boundary_edge_indices
        h5group["directions"] = self.directions
        h5group["edge_lengths"] = self.edge_lengths
        h5group["dual_edge_lengths"] = self.dual_edge_lengths

    @classmethod
    def from_hdf5(cls, h5group: h5py.Group) -> "EdgeMesh":
        """Load edge mesh from file.

        Args:
            h5group: The HDF5 group to load from.

        Returns:
            The loaded edge mesh.
        """
        if not (
            "centers" in h5group
            and "edges" in h5group
            and "boundary_edge_indices" in h5group
            and "directions" in h5group
            and "edge_lengths" in h5group
            and "dual_edge_lengths" in h5group
        ):
            raise IOError("Could not load edge mesh due to missing data.")
        return EdgeMesh(
            centers=np.array(h5group["centers"]),
            edges=np.array(h5group["edges"], dtype=np.int64),
            boundary_edge_indices=np.array(h5group["boundary_edge_indices"], np.int64),
            directions=np.array(h5group["directions"]),
            edge_lengths=np.array(h5group["edge_lengths"]),
            dual_edge_lengths=np.array(h5group["dual_edge_lengths"]),
        )
