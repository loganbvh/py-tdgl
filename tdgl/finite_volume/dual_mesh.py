from typing import Sequence

import h5py
import numpy as np

from .util import generate_voronoi_vertices


class DualMesh:
    """A dual Voronoi mesh.

    **Note**: Use :meth:`DualMesh.from_mesh` to create the dual from an
    existing mesh.

    Args:
        x: Coordinates for the dual mesh points.
        y: Coordinates for the dual mesh points.
    """

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

    @classmethod
    def from_mesh(
        cls, x: np.ndarray, y: np.ndarray, elements: np.ndarray
    ) -> "DualMesh":
        """Create a dual mesh from the mesh.

        Args:
            x: Coordinates for the mesh points.
            y: Coordinates for the mesh points.
            elements: The mesh elements.

        Returns:
            The dual mesh.
        """

        # Get the location of the Voronoi vertices from the original mesh
        xc, yc = generate_voronoi_vertices(x, y, elements)
        return DualMesh(xc, yc)

    def to_hdf5(self, h5group: h5py.Group) -> None:
        """Save the mesh to file.

        Args:
            h5group: The HDF5 group to write to.
        """
        h5group["x"] = self.x
        h5group["y"] = self.y

    @classmethod
    def from_hdf5(cls, h5group: h5py.Group) -> "DualMesh":
        """Load mesh from file.

        Args:
            h5group: The HDF5 group to load from.

        Returns:
            The loaded dual mesh
        """
        if not ("x" in h5group and "y" in h5group):
            raise IOError("Could not load dual mesh due to missing data.")
        return DualMesh(x=h5group["x"], y=h5group["y"])
