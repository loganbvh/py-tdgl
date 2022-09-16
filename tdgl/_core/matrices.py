from typing import Union, Sequence

from scipy.sparse import csr_matrix, coo_matrix
import numpy as np

from .enums import SparseFormat, MatrixType
from .mesh.mesh import Mesh


def build_divergence(mesh: Mesh) -> csr_matrix:
    """Build the divergence matrix that takes the divergence of a function living
    on the edge onto the sites.

    Args:
        mesh: The mesh.

    Returns:
        The divergence matrix.
    """
    edge_mesh = mesh.edge_mesh
    # Indices for each edge
    edge_indices = np.arange(len(edge_mesh.edges))
    # Compute the weights for each edge
    weights = edge_mesh.dual_edge_lengths
    # Rows and cols to update
    rows = np.concatenate([edge_mesh.edges[:, 0], edge_mesh.edges[:, 1]])
    cols = np.concatenate([edge_indices, edge_indices])
    # The values
    values = np.concatenate(
        [
            weights / mesh.areas[edge_mesh.edges[:, 0]],
            -weights / mesh.areas[edge_mesh.edges[:, 1]],
        ]
    )
    return coo_matrix(
        (values, (rows, cols)), shape=(len(mesh.x), len(edge_mesh.edges))
    ).tocsr()


def build_gradient(mesh: Mesh, link_exponents: Union[np.ndarray, None] = None):
    """Build the gradient for a function living on the sites onto the edges.

    Args:
        mesh: The mesh.
        link_exponents: The value is integrated, exponentiated and used as
            a link variable.

    Returns:
        The gradient matrix.
    """

    edge_mesh = mesh.edge_mesh
    # Indices for each edge
    edge_indices = np.arange(len(edge_mesh.edges))
    # Compute the weights for each edge
    weights = 1 / edge_mesh.edge_lengths
    # Compute the link variable weights
    link_variable_weights = (
        np.exp(-1j * (np.asarray(link_exponents) * edge_mesh.directions).sum(axis=1))
        if link_exponents is not None
        else np.ones(len(weights))
    )
    # Rows and cols to update
    rows = np.concatenate([edge_indices, edge_indices])
    cols = np.concatenate([edge_mesh.edges[:, 1], edge_mesh.edges[:, 0]])
    # The values
    values = np.concatenate([link_variable_weights * weights, -weights])
    return coo_matrix(
        (values, (rows, cols)), shape=(len(edge_mesh.edges), len(mesh.x))
    ).tocsr()


def build_laplacian(
    mesh: Mesh,
    link_exponents: Union[np.ndarray, None] = None,
    fixed_sites: Union[np.ndarray, None] = None,
    fixed_sites_eigenvalues: float = 1,
    sparse_format: SparseFormat = SparseFormat.CSR,
) -> csr_matrix:
    """Build a Laplacian matrix on a given mesh.

    The default boundary condition is homogenous Neumann conditions. To get
    Dirichlet conditions, add fixed sites. To get non-homogenous Neumann condition,
    the flux needs to be specified using a Neumann boundary Laplacian matrix.

    Args:
        mesh: The mesh.
        link_exponents: The value is integrated, exponentiated and used as a
            link variable.
        fixed_sites: The sites to hold fixed.
        fixed_sites_eigenvalues: The eigenvalues for the fixed sites.
        sparse_format: Sparse format used to save the data.

    Returns:
        The Laplacian matrix.
    """
    edge_mesh = mesh.edge_mesh
    # Compute the weights for each edge
    weights = edge_mesh.dual_edge_lengths / edge_mesh.edge_lengths
    # Compute the link variable weights
    link_variable_weights = (
        np.exp(-1j * (link_exponents * edge_mesh.directions).sum(axis=1))
        if link_exponents is not None
        else np.ones(len(weights))
    )
    # Rows and cols to update
    rows = np.concatenate(
        [
            edge_mesh.edges[:, 0],
            edge_mesh.edges[:, 1],
            edge_mesh.edges[:, 0],
            edge_mesh.edges[:, 1],
        ]
    )
    cols = np.concatenate(
        [
            edge_mesh.edges[:, 1],
            edge_mesh.edges[:, 0],
            edge_mesh.edges[:, 0],
            edge_mesh.edges[:, 1],
        ]
    )
    # The values
    values = np.concatenate(
        [
            weights * link_variable_weights / mesh.areas[edge_mesh.edges[:, 0]],
            weights
            * link_variable_weights.conjugate()
            / mesh.areas[edge_mesh.edges[:, 1]],
            -weights / mesh.areas[edge_mesh.edges[:, 0]],
            -weights / mesh.areas[edge_mesh.edges[:, 1]],
        ]
    )
    # Build the Laplacian
    laplacian = coo_matrix((values, (rows, cols)), shape=(len(mesh.x), len(mesh.x)))
    # Change the rows corresponding to fixed sites to identity
    if fixed_sites is not None:
        # Convert laplacian to list of lists
        # This makes it quick to do slices
        laplacian = laplacian.tolil()
        # Change the rows corresponding to the fixed sites
        laplacian[fixed_sites, :] = 0
        laplacian[fixed_sites, fixed_sites] = fixed_sites_eigenvalues

    return laplacian.asformat(sparse_format.value, copy=False)


def build_neumann_boundary_laplacian(
    mesh: Mesh, fixed_sites: Union[np.ndarray, None] = None
) -> csr_matrix:
    """Build extra matrix for the Laplacian to set non-homogenous Neumann
    boundary conditions.

    Args:
        mesh: The mesh.
        fixed_sites: The fixed sites.

    Returns:
        The Neumann boundary matrix.
    """

    edge_mesh = mesh.edge_mesh
    boundary_index = np.arange(len(edge_mesh.boundary_edge_indices))
    # Get the boundary edges which are stored in the beginning of
    # the edge vector
    boundary_edges = edge_mesh.edges[edge_mesh.boundary_edge_indices]
    boundary_edges_length = edge_mesh.edge_lengths[edge_mesh.boundary_edge_indices]
    # Rows and cols to update
    rows = np.concatenate([boundary_edges[:, 0], boundary_edges[:, 1]])
    cols = np.concatenate([boundary_index, boundary_index])
    # The values
    values = np.concatenate(
        [
            boundary_edges_length / (2 * mesh.areas[boundary_edges[:, 0]]),
            boundary_edges_length / (2 * mesh.areas[boundary_edges[:, 1]]),
        ]
    )
    # Build the matrix
    neumann_laplacian = coo_matrix(
        (values, (rows, cols)), shape=(len(mesh.x), len(boundary_index))
    )
    # Change the rows corresponding to fixed sites to identity
    if fixed_sites is not None:
        # Convert laplacian to list of lists
        # This makes it quick to do slices
        neumann_laplacian = neumann_laplacian.tolil()
        # Change the rows corresponding to the fixed sites
        neumann_laplacian[fixed_sites, :] = 0

    return neumann_laplacian.tocsr()


class MatrixBuilder:
    """A factory that builds the FEM matrixes for a given Mesh.

    Args:
        mesh: The mesh to build matrices on.
    """

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.fixed_sites: Union[np.ndarray, None] = None
        self.fixed_sites_eigenvalue: float = 1
        self.link_exponents: Union[np.ndarray, None] = None

    def with_dirichlet_boundary(
        self, fixed_sites: Sequence[int], fixed_sites_eigenvalues: float = 1
    ) -> "MatrixBuilder":
        """Add sites using Dirichlet boundary condition.

        Args:
            fixed_sites: The sites using Dirichlet boundary condition.
            fixed_sites_eigenvalues: The eigenvalues when applying the
                matrix on one of these sites.

        Returns:
            This builder.
        """
        self.fixed_sites = np.asarray(fixed_sites)
        self.fixed_sites_eigenvalue = fixed_sites_eigenvalues
        return self

    def with_link_exponents(self, link_exponents: Sequence[float]) -> "MatrixBuilder":
        """Add link exponents to the matrix.

        Args:
            link_exponents: The value is integrated, exponentiated and
                used as a link variable.

        Returns:
            This builder.
        """
        self.link_exponents = np.asarray(link_exponents)
        return self

    def build(
        self, matrix_type: MatrixType, sparse_format: SparseFormat = SparseFormat.CSR
    ) -> csr_matrix:
        """Build a matrix.

        Args:
            matrix_type: The type of matrix to build.
            sparse_format: The matrix format to return.

        Returns:
            The matrix
        """
        if matrix_type is MatrixType.LAPLACIAN:
            return build_laplacian(
                self.mesh,
                self.link_exponents,
                self.fixed_sites,
                self.fixed_sites_eigenvalue,
                sparse_format,
            )

        if matrix_type is MatrixType.NEUMANN_BOUNDARY_LAPLACIAN:
            return build_neumann_boundary_laplacian(self.mesh, self.fixed_sites)

        if matrix_type is MatrixType.DIVERGENCE:
            return build_divergence(self.mesh)

        if matrix_type is MatrixType.GRADIENT:
            return build_gradient(self.mesh, self.link_exponents)

        raise ValueError("Unknown matrix type.")

    def clone(self) -> "MatrixBuilder":
        """Make a copy of the matrix builder.

        Returns:
            A copy of the matrix builder
        """
        clone = MatrixBuilder(self.mesh)
        clone.fixed_sites = np.copy(self.fixed_sites)
        clone.fixed_sites_eigenvalue = self.fixed_sites_eigenvalue
        clone.link_exponents = np.copy(self.link_exponents)
        return clone
