import warnings
from typing import Tuple, Union

import numpy as np
import scipy.sparse as sp

from .mesh import Mesh


def build_divergence(mesh: Mesh) -> sp.csr_matrix:
    """Build the divergence matrix that takes the divergence of a function living
    on the edges onto the sites.

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
    edges0 = edge_mesh.edges[:, 0]
    edges1 = edge_mesh.edges[:, 1]
    rows = np.concatenate([edges0, edges1])
    cols = np.concatenate([edge_indices, edge_indices])
    values = np.concatenate(
        [weights / mesh.areas[edges0], -weights / mesh.areas[edges1]]
    )
    return sp.csr_matrix(
        (values, (rows, cols)), shape=(len(mesh.sites), len(edge_mesh.edges))
    )


def build_gradient(
    mesh: Mesh,
    link_exponents: Union[np.ndarray, None] = None,
    weights: Union[np.ndarray, None] = None,
) -> sp.csr_matrix:
    """Build the gradient for a function living on the sites onto the edges.

    Args:
        mesh: The mesh.
        link_exponents: The value is integrated, exponentiated and used as
            a link variable.

    Returns:
        The gradient matrix.
    """
    edge_mesh = mesh.edge_mesh
    edge_indices = np.arange(len(edge_mesh.edges))
    if weights is None:
        weights = 1 / edge_mesh.edge_lengths
    if link_exponents is None:
        link_variable_weights = np.ones(len(weights))
    else:
        link_variable_weights = np.exp(
            -1j * np.einsum("ij, ij -> i", link_exponents, edge_mesh.directions)
        )
    rows = np.concatenate([edge_indices, edge_indices])
    cols = np.concatenate([edge_mesh.edges[:, 1], edge_mesh.edges[:, 0]])
    values = np.concatenate([link_variable_weights * weights, -weights])
    return sp.csr_matrix(
        (values, (rows, cols)), shape=(len(edge_mesh.edges), len(mesh.sites))
    )


def build_laplacian(
    mesh: Mesh,
    link_exponents: Union[np.ndarray, None] = None,
    fixed_sites: Union[np.ndarray, None] = None,
    free_rows: Union[np.ndarray, None] = None,
    fixed_sites_eigenvalues: float = 1,
    weights: Union[np.ndarray, None] = None,
) -> Tuple[sp.csc_matrix, np.ndarray]:
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

    Returns:
        The Laplacian matrix and indices of non-fixed rows.
    """
    if fixed_sites is None:
        fixed_sites = np.array([], dtype=int)

    edge_mesh = mesh.edge_mesh
    if weights is None:
        weights = edge_mesh.dual_edge_lengths / edge_mesh.edge_lengths
    if link_exponents is None:
        link_variable_weights = np.ones(len(weights))
    else:
        link_variable_weights = np.exp(
            -1j * np.einsum("ij, ij -> i", link_exponents, edge_mesh.directions)
        )
    edges0 = edge_mesh.edges[:, 0]
    edges1 = edge_mesh.edges[:, 1]
    rows = np.concatenate([edges0, edges1, edges0, edges1])
    cols = np.concatenate([edges1, edges0, edges0, edges1])
    areas0 = mesh.areas[edges0]
    areas1 = mesh.areas[edges1]
    values = np.concatenate(
        [
            weights * link_variable_weights / areas0,
            weights * link_variable_weights.conjugate() / areas1,
            -weights / areas0,
            -weights / areas1,
        ]
    )
    # Exclude all edges connected to fixed sites and set the
    # fixed site diagonal elements separately.
    if free_rows is None:
        free_rows = np.isin(rows, fixed_sites, invert=True)
    rows = rows[free_rows]
    cols = cols[free_rows]
    values = values[free_rows]
    rows = np.concatenate([rows, fixed_sites])
    cols = np.concatenate([cols, fixed_sites])
    values = np.concatenate(
        [values, fixed_sites_eigenvalues * np.ones(len(fixed_sites))]
    )
    laplacian = sp.csc_matrix(
        (values, (rows, cols)), shape=(len(mesh.sites), len(mesh.sites))
    )
    return laplacian, free_rows


def build_neumann_boundary_laplacian(
    mesh: Mesh, fixed_sites: Union[np.ndarray, None] = None
) -> sp.csr_matrix:
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
    neumann_laplacian = sp.csr_matrix(
        (values, (rows, cols)), shape=(len(mesh.sites), len(boundary_index))
    )
    # Change the rows corresponding to fixed sites to identity
    if fixed_sites is not None:
        # Convert laplacian to list of lists
        # This makes it quick to do slices
        neumann_laplacian = neumann_laplacian.tolil()
        # Change the rows corresponding to the fixed sites
        neumann_laplacian[fixed_sites, :] = 0

    return neumann_laplacian.tocsr()


class MeshOperators:
    """A container for the finite volume operators for a given mesh.

    Args:
        mesh: The :class:`tdgl.finite_volume.Mesh` instance for which to construct
            operators.
        fixed_sites: The indices of any sites for which the value of :math:`\\psi`
            and :math:`\\mu` are fixed as boundary conditions.
    """

    def __init__(self, mesh: Mesh, fixed_sites: Union[np.ndarray, None] = None):
        self.mesh = mesh
        edge_mesh = mesh.edge_mesh
        self.fixed_sites = fixed_sites
        self.laplacian_free_rows: Union[np.ndarray, None] = None
        self.divergence: Union[sp.spmatrix, None] = None
        self.mu_laplacian: Union[sp.spmatrix, None] = None
        self.mu_boundary_laplacian: Union[sp.spmatrix, None] = None
        self.mu_laplacian_lu: Union[sp.linalg.SuperLU, None] = None
        self.psi_gradient: Union[sp.spmatrix, None] = None
        self.psi_laplacian: Union[sp.spmatrix, None] = None
        self.link_exponents: Union[np.ndarray, None] = None
        # Compute these quantities just once, as they never change.
        self.gradient_weights = 1 / edge_mesh.edge_lengths
        self.laplacian_weights = edge_mesh.dual_edge_lengths / edge_mesh.edge_lengths
        self.gradient_link_rows = np.arange(len(edge_mesh.edges), dtype=int)
        self.gradient_link_cols = edge_mesh.edges[:, 1]
        self.laplacian_link_rows = np.concatenate(
            [edge_mesh.edges[:, 0], edge_mesh.edges[:, 1]]
        )
        self.laplacian_link_cols = np.concatenate(
            [edge_mesh.edges[:, 1], edge_mesh.edges[:, 0]]
        )

    def build_operators(self) -> None:
        """Construct the vector potential-independent operators."""
        mesh = self.mesh
        self.mu_laplacian, _ = build_laplacian(mesh, weights=self.laplacian_weights)
        self.mu_laplacian_lu = sp.linalg.splu(self.mu_laplacian)
        self.mu_boundary_laplacian = build_neumann_boundary_laplacian(mesh)
        self.mu_gradient = build_gradient(mesh, weights=self.gradient_weights)
        self.divergence = build_divergence(mesh)

    def set_link_exponents(self, link_exponents: np.ndarray) -> None:
        """Set the link variables and construct the covarient gradient
        and Laplacian for psi.

        Args:
            link_exponents: The value is integrated, exponentiated and used as
                a link variable.
        """
        mesh = self.mesh
        if link_exponents is not None:
            link_exponents = np.asarray(link_exponents)
        self.link_exponents = link_exponents
        if self.psi_gradient is None:
            # Build the matrices from scratch
            self.psi_gradient = build_gradient(
                mesh,
                link_exponents=link_exponents,
                weights=self.gradient_weights,
            )
            self.psi_laplacian, self.laplacian_free_rows = build_laplacian(
                mesh,
                link_exponents=link_exponents,
                fixed_sites=self.fixed_sites,
                free_rows=self.laplacian_free_rows,
                weights=self.laplacian_weights,
            )
            return
        # Just update the link variables
        directions = mesh.edge_mesh.directions
        if self.link_exponents is None:
            link_variables = np.ones(len(directions))
        else:
            link_variables = np.exp(
                -1j * np.einsum("ij, ij -> i", self.link_exponents, directions)
            )
        with warnings.catch_warnings():
            # This is slightly faster than re-creating the sparse matrices
            # from scratch.
            warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)
            # Update gradient for psi
            values = self.gradient_weights * link_variables
            rows, cols = self.gradient_link_rows, self.gradient_link_cols
            self.psi_gradient[rows, cols] = values
            # Update Laplacian for psi
            areas0 = mesh.areas[mesh.edge_mesh.edges[:, 0]]
            areas1 = mesh.areas[mesh.edge_mesh.edges[:, 1]]
            # Only update rows that are not fixed by boundary conditions
            free_rows = self.laplacian_free_rows[: len(self.laplacian_link_rows)]
            rows = self.laplacian_link_rows[free_rows]
            cols = self.laplacian_link_cols[free_rows]
            values = np.concatenate(
                [
                    self.laplacian_weights * link_variables / areas0,
                    self.laplacian_weights * link_variables.conjugate() / areas1,
                ]
            )[free_rows]
            self.psi_laplacian[rows, cols] = values
