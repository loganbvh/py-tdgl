import warnings
from typing import Callable, Tuple, Union

import numpy as np
import scipy.sparse as sp

try:
    import cupy  # type: ignore
except ImportError:
    cupy = None
else:
    from cupyx.scipy.sparse import csc_matrix, csr_matrix  # type: ignore
    from cupyx.scipy.sparse.linalg import factorized  # type: ignore

from ..solver.options import SparseSolver
from .mesh import Mesh


def _get_spmatrix_offsets_cupy(spmatrix, i, j):
    """Calculates the sparse matrix offsets for a set of rows ``i`` and columns ``j``."""
    # See _set_many() at
    # https://github.com/cupy/cupy/blob/5c32e40af32f6f9627e09d47ecfeb7e9281ccab2/cupyx/scipy/sparse/_compressed.py#L525
    i, j, M, N = spmatrix._prepare_indices(i, j)
    new_sp = csr_matrix(
        (
            cupy.arange(spmatrix.nnz, dtype=cupy.float32),
            spmatrix.indices,
            spmatrix.indptr,
        ),
        shape=(M, N),
    )
    offsets = new_sp._get_arrayXarray(i, j, not_found_val=-1).astype(cupy.int32).ravel()
    return offsets, (i, j, M, N)


def _spmatrix_set_many(spmatrix, i, j, x):
    """spmatrix.__setitem__()"""
    if sp.issparse(spmatrix):
        spmatrix[i, j] = x
        return

    i, j = spmatrix._swap(i, j)
    offsets, (i, j, M, N) = _get_spmatrix_offsets_cupy(spmatrix, i, j)

    mask = offsets > -1
    # update where possible
    spmatrix.data[offsets[mask]] = x[mask]

    if not mask.all():
        # only insertions remain
        mask = ~mask
        i = i[mask]
        i[i < 0] += M
        j = j[mask]
        j[j < 0] += N
        spmatrix._insert_many(i, j, x[mask])


def build_divergence(mesh: Mesh) -> sp.csr_array:
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
    return sp.csr_array(
        (values, (rows, cols)), shape=(len(mesh.sites), len(edge_mesh.edges))
    )


def build_gradient(
    mesh: Mesh,
    link_exponents: Union[np.ndarray, None] = None,
    weights: Union[np.ndarray, None] = None,
) -> sp.csr_array:
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
    return sp.csr_array(
        (values, (rows, cols)), shape=(len(edge_mesh.edges), len(mesh.sites))
    )


def build_laplacian(
    mesh: Mesh,
    link_exponents: Union[np.ndarray, None] = None,
    fixed_sites: Union[np.ndarray, None] = None,
    free_rows: Union[np.ndarray, None] = None,
    fixed_sites_eigenvalues: float = 1,
    weights: Union[np.ndarray, None] = None,
) -> Tuple[sp.csc_array, np.ndarray]:
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
    laplacian = sp.csc_array(
        (values, (rows, cols)), shape=(len(mesh.sites), len(mesh.sites))
    )
    return laplacian, free_rows


def build_neumann_boundary_laplacian(
    mesh: Mesh, fixed_sites: Union[np.ndarray, None] = None
) -> sp.csr_array:
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
    neumann_laplacian = sp.csr_array(
        (values, (rows, cols)), shape=(len(mesh.sites), len(boundary_index))
    )
    # Change the rows corresponding to fixed sites to identity
    if fixed_sites is not None:
        # Convert laplacian to list of lists
        # This makes it quick to do slices
        neumann_laplacian = neumann_laplacian.tolil()
        # Change the rows corresponding to the fixed sites
        neumann_laplacian[fixed_sites, :] = 0

    return neumann_laplacian.tocsr(copy=False)


class MeshOperators:
    """A container for the finite volume operators for a given mesh.

    Args:
        mesh: The :class:`tdgl.finite_volume.Mesh` instance for which to construct
            operators.
        sparse_solver: The sparse solver for which to build mesh operators.
        use_cupy: Use CuPy for linear algebra.
        fixed_sites: The indices of any sites for which the value of :math:`\\psi`
            and :math:`\\mu` are fixed as boundary conditions.
    """

    def __init__(
        self,
        mesh: Mesh,
        sparse_solver: SparseSolver,
        use_cupy: bool = False,
        fixed_sites: Union[np.ndarray, None] = None,
        fix_psi: bool = True,
    ):
        self.mesh = mesh
        self.areas = mesh.areas
        edge_mesh = mesh.edge_mesh
        self.edges = edge_mesh.edges
        self.edge_directions = edge_mesh.directions
        self.use_cupy = use_cupy
        self.sparse_solver = sparse_solver
        self.fixed_sites = fixed_sites
        self.fix_psi = fix_psi
        self.laplacian_free_rows: Union[np.ndarray, None] = None
        self.divergence: Union[sp.spmatrix, None] = None
        self.mu_laplacian: Union[sp.spmatrix, None] = None
        self.mu_boundary_laplacian: Union[sp.spmatrix, None] = None
        self.mu_laplacian_lu: Union[Callable, None] = None
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
        self.mu_boundary_laplacian = build_neumann_boundary_laplacian(mesh)
        self.mu_gradient = build_gradient(mesh, weights=self.gradient_weights)
        self.divergence = build_divergence(mesh)
        if self.use_cupy:
            assert cupy is not None
            self.mu_boundary_laplacian = csr_matrix(self.mu_boundary_laplacian)
            self.mu_gradient = csr_matrix(self.mu_gradient)
            self.divergence = csr_matrix(self.divergence)
            self.areas = cupy.array(self.areas)
            self.edge_directions = cupy.array(self.edge_directions)
        if self.sparse_solver is SparseSolver.CUPY:
            assert cupy is not None
            self.mu_laplacian = csc_matrix(self.mu_laplacian)
            self.mu_laplacian_lu = factorized(self.mu_laplacian)
        elif self.sparse_solver is SparseSolver.PARDISO:
            # https://github.com/loganbvh/py-tdgl/issues/74
            # https://github.com/haasad/PyPardiso/issues/68
            self.mu_laplacian = sp.csc_matrix(self.mu_laplacian)
            self.mu_laplacian_lu = None
        else:
            use_umfpack = self.sparse_solver is SparseSolver.UMFPACK
            sp.linalg.use_solver(useUmfpack=use_umfpack)
            self.mu_laplacian_lu = sp.linalg.factorized(self.mu_laplacian)

    def set_link_exponents(self, link_exponents: np.ndarray) -> None:
        """Set the link variables and construct the covarient gradient
        and Laplacian for psi.

        Args:
            link_exponents: The value is integrated, exponentiated and used as
                a link variable.
        """
        mesh = self.mesh
        xp = cupy if self.use_cupy else np
        self.link_exponents = xp.asarray(link_exponents)
        if self.psi_gradient is None:
            # Build the matrices from scratch
            self.psi_gradient = build_gradient(
                mesh,
                link_exponents=link_exponents,
                weights=self.gradient_weights,
            )
            if self.fix_psi:
                fixed_sites = self.fixed_sites
                free_rows = self.laplacian_free_rows
            else:
                fixed_sites = free_rows = None
            self.psi_laplacian, self.laplacian_free_rows = build_laplacian(
                mesh,
                link_exponents=link_exponents,
                fixed_sites=fixed_sites,
                free_rows=free_rows,
                weights=self.laplacian_weights,
            )
            if self.use_cupy:
                self.psi_gradient = csr_matrix(self.psi_gradient)
                self.psi_laplacian = csr_matrix(self.psi_laplacian)
                self.gradient_weights = cupy.asarray(self.gradient_weights)
                self.laplacian_weights = cupy.asarray(self.laplacian_weights)
            return
        # Just update the link variables
        edges = self.edges
        directions = self.edge_directions
        if self.link_exponents is None:
            link_variables = xp.ones(len(directions))
        else:
            link_variables = xp.exp(
                -1j * xp.einsum("ij, ij -> i", self.link_exponents, directions)
            )
        with warnings.catch_warnings():
            # This is faster than re-creating the sparse matrices from scratch.
            warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)
            # Update gradient for psi
            values = self.gradient_weights * link_variables
            rows = self.gradient_link_rows
            cols = self.gradient_link_cols
            # self.psi_gradient[rows, cols] = values
            _spmatrix_set_many(self.psi_gradient, rows, cols, values)
            # Update Laplacian for psi
            areas = self.areas
            weights = self.laplacian_weights
            values = xp.concatenate(
                [
                    weights * link_variables / areas[edges[:, 0]],
                    weights * link_variables.conjugate() / areas[edges[:, 1]],
                ]
            )
            # Only update rows that are not fixed by boundary conditions
            if self.fix_psi:
                free_rows = self.laplacian_free_rows[: len(self.laplacian_link_rows)]
                rows = self.laplacian_link_rows[free_rows]
                cols = self.laplacian_link_cols[free_rows]
                values = values[free_rows]
            else:
                rows = self.laplacian_link_rows
                cols = self.laplacian_link_cols
            # self.psi_laplacian[rows, cols] = values
            _spmatrix_set_many(self.psi_laplacian, rows, cols, values)

    def get_supercurrent(self, psi: np.ndarray):
        """Compute the supercurrent on the edges.

        Args:
            psi: The value of the complex order parameter.

        Returns:
            The supercurrent at each edge.
        """
        return (psi.conjugate()[self.edges[:, 0]] * (self.psi_gradient @ psi)).imag
