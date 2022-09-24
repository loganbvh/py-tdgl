from typing import Optional, Callable, Union
import logging
from datetime import datetime

import numpy as np
from scipy.sparse.linalg import splu
from scipy import spatial

try:
    import jax
    import jax.numpy as jnp
except (ModuleNotFoundError, ImportError):
    jax = None

from .solution import Solution
from .parameter import Parameter
from .device.device import Device
from ._core.runner import Runner
from ._core.io.data_handler import DataHandler
from ._core.matrices import MatrixBuilder
from ._core.enums import MatrixType, SparseFormat
from ._core.tdgl import get_supercurrent, get_observable_on_site


logger = logging.getLogger(__name__)


def solve(
    device: Device,
    applied_vector_potential: Callable,
    output: str,
    source_drain_current: float = 0,
    pinning_sites: Union[float, Callable, None] = None,
    miniters: Optional[int] = None,
    dt: float = 1e-4,
    min_steps: int = 0,
    max_steps: int = 10_000,
    save_every: int = 100,
    skip: int = 0,
    complex_time_scale: float = 5.79,
    gamma: float = 10.0,
    field_units: str = "mT",
    current_units: str = "uA",
    rtol: float = 0,
    include_screening: bool = False,
    seed_solution: Optional[Solution] = None,
    rng_seed: Optional[int] = None,
):

    start_time = datetime.now()

    if rng_seed is None:
        rng_seed = np.random.SeedSequence().entropy

    if not (0 <= rtol <= 1):
        raise ValueError(f"Relative tolerance must be in [0, 1], got {rtol:.2e}.")

    ureg = device.ureg
    mesh = device.mesh
    sites = device.points
    voltage_points = mesh.voltage_points
    length_units = ureg(device.length_units)
    xi = device.coherence_length
    # The vector potential is evaluated on the mesh edges,
    # where the edge coordinates are in dimensionful units.
    x = mesh.edge_mesh.x * xi
    y = mesh.edge_mesh.y * xi
    K0 = device.K0
    Bc2 = device.Bc2
    z = device.layer.z0 * xi * np.ones_like(x)
    if not isinstance(applied_vector_potential, Parameter):
        applied_vector_potential = Parameter(applied_vector_potential)
    # Evaluate the vector potential
    vector_potential = applied_vector_potential(x, y, z)
    vector_potential = np.asarray(vector_potential)[:, :2]
    shape = vector_potential.shape
    if shape != x.shape + (2,):
        raise ValueError(f"Unexpected shape for vector_potential: {shape}.")

    vector_potential = (
        (vector_potential * ureg(field_units) * length_units)
        / (Bc2 * xi * length_units)
    ).to_base_units()
    assert "dimensionless" in str(vector_potential.units)
    vector_potential = vector_potential.magnitude

    u = complex_time_scale

    data_handler = DataHandler(
        mesh,
        output_file=output,
        save_mesh=True,
        logger=logger,
    )

    # Create the matrix builder for fields with Neumann boundary conditions
    # and no link variables.
    builder = MatrixBuilder(mesh)
    # Build matrices for scalar potential.
    mu_laplacian = builder.build(MatrixType.LAPLACIAN, sparse_format=SparseFormat.CSC)
    mu_laplacian_lu = splu(mu_laplacian)
    mu_boundary_laplacian = builder.build(MatrixType.NEUMANN_BOUNDARY_LAPLACIAN)
    mu_gradient = builder.build(MatrixType.GRADIENT)
    # Build divergence for the supercurrent.
    divergence = builder.build(MatrixType.DIVERGENCE)

    edge_positions = np.array([x, y]).transpose()[mesh.edge_mesh.boundary_edge_indices]
    if device.source_terminal is None:
        input_edges_index = np.array([], dtype=np.int64)
        output_edges_index = np.array([], dtype=np.int64)
        input_sites_index = np.array([], dtype=np.int64)
        output_sites_index = np.array([], dtype=np.int64)
    else:
        input_edges_index = device.source_terminal.contains_points(
            edge_positions, index=True
        )
        output_edges_index = device.drain_terminal.contains_points(
            edge_positions, index=True
        )
        input_sites_index = device.source_terminal.contains_points(sites, index=True)
        output_sites_index = device.drain_terminal.contains_points(sites, index=True)

    normal_boundary_index = np.sort(
        np.concatenate([input_sites_index, output_sites_index])
    ).astype(np.int64)

    interior_indices = np.setdiff1d(
        np.arange(mesh.x.shape[0], dtype=int), normal_boundary_index
    )

    if callable(pinning_sites):
        pinning_sites = np.apply_along_axis(pinning_sites, 1, sites).astype(bool)
        (pinning_sites,) = np.where(pinning_sites)
    else:
        if pinning_sites is None:
            pinning_sites = 0
        site_areas = xi**2 * mesh.areas[interior_indices]
        total_area = site_areas.sum()
        n_pinning_sites = int(pinning_sites * total_area)
        if n_pinning_sites > len(interior_indices):
            area_units = (ureg(device.length_units) ** 2).units
            raise ValueError(
                f"The total number of pinning sites ({n_pinning_sites}) for the requested"
                f" areal density of pinning sites ({pinning_sites:.2f} / {area_units:~P})"
                f" exceeds the total number of interior sites ({len(interior_indices)})."
                f" Try setting a smaller density of pinning sites."
            )
        rng = np.random.default_rng(rng_seed)
        pinning_sites = rng.choice(
            interior_indices,
            size=n_pinning_sites,
            p=site_areas / total_area,
            replace=False,
        )

    fixed_sites = np.union1d(normal_boundary_index, pinning_sites)

    # Update the builder and set fixed sites and link variables for
    # the complex field.
    builder.with_dirichlet_boundary(fixed_sites=fixed_sites).with_link_exponents(
        link_exponents=vector_potential
    )
    # Build complex field matrices.
    psi_laplacian = builder.build(MatrixType.LAPLACIAN)
    psi_gradient = builder.build(MatrixType.GRADIENT)
    # Initialize the complex field and the scalar potential.
    psi = np.ones_like(mesh.x, dtype=np.complex128)
    psi[fixed_sites] = 0
    mu = np.zeros(len(mesh.x))
    mu_boundary = np.zeros_like(mesh.edge_mesh.boundary_edge_indices, dtype=np.float64)
    lengths = mesh.edge_mesh.edge_lengths
    if device.source_terminal is not None:
        input_length = lengths[input_edges_index].sum()
        output_length = lengths[output_edges_index].sum()
        mu_boundary[input_edges_index] = source_drain_current / input_length
        mu_boundary[output_edges_index] = -source_drain_current / output_length
    # Create the alpha parameter which weakens the complex field if it
    # is less than unity.
    alpha = np.ones_like(mesh.x, dtype=np.float64)
    sq_gamma = gamma**2

    if include_screening:
        edge_points = np.array([mesh.edge_mesh.x, mesh.edge_mesh.y]).T
        edge_directions = mesh.edge_mesh.directions
        edge_directions = (
            edge_directions / np.linalg.norm(edge_directions, axis=1)[:, np.newaxis]
        )
        site_points = np.array([mesh.x, mesh.y]).T
        weights = mesh.areas
        inv_rho = 1 / spatial.distance.cdist(edge_points, site_points)
        # (edges, sites, spatial dimensions)
        inv_rho = inv_rho[:, :, np.newaxis]
        inv_rho = inv_rho * weights[np.newaxis, :, np.newaxis]
        A_scale = (ureg("mu_0") / (4 * np.pi) * K0 / Bc2).to_base_units().magnitude
        inv_rho *= A_scale

        if jax is None:
            einsum = np.einsum
        else:
            einsum = jnp.einsum
            inv_rho = jax.device_put(inv_rho)

    def update(
        state,
        running_state,
        psi_val,
        mu_val,
        supercurrent_val,
        normal_current_val,
        induced_vector_potential_val,
    ):
        step = state["step"]
        dt_val = state["dt"]
        running_state.append("current", source_drain_current)

        nonlocal psi_laplacian
        nonlocal psi_gradient

        if include_screening and step > 0:
            _ = builder.with_link_exponents(
                vector_potential + induced_vector_potential_val
            )
            psi_laplacian = builder.build(MatrixType.LAPLACIAN)
            psi_gradient = builder.build(MatrixType.GRADIENT)

        # Compute the next time step for psi with the discrete gauge
        # invariant discretization presented in chapter 5 in
        # http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-312132

        abs_sq_psi = np.abs(psi_val) ** 2
        phase = np.exp(-1j * mu_val * dt_val)
        z = phase * sq_gamma / 2 * psi_val
        w = z * abs_sq_psi + phase * (
            psi_val
            + (dt_val / u)
            * np.sqrt(1 + sq_gamma * abs_sq_psi)
            * ((alpha - abs_sq_psi) * psi_val + psi_laplacian @ psi_val)
        )
        c = w.real * z.real + w.imag * z.imag
        # Find the modulus squared for the next time step
        two_c_1 = 2 * c + 1
        w2 = np.abs(w) ** 2
        new_sq_psi = (2 * w2) / (
            two_c_1 + np.sqrt(two_c_1**2 - 4 * np.abs(z) ** 2 * w2)
        )
        if not np.isfinite(new_sq_psi).all():
            raise ValueError(f"NaN or inf encountered at step {step}.")
        psi_val = w - z * new_sq_psi

        old_current = supercurrent_val + normal_current_val
        # Get the supercurrent
        supercurrent_val = get_supercurrent(psi_val, psi_gradient, mesh.edge_mesh.edges)
        supercurrent_divergence = divergence @ supercurrent_val
        # Solve for mu
        lhs = supercurrent_divergence - (mu_boundary_laplacian @ mu_boundary)
        mu_val = mu_laplacian_lu.solve(lhs)
        normal_current_val = -mu_gradient @ mu_val
        # Update the voltage
        if device.voltage_points is None:
            d_mu = 0
        else:
            d_mu = mu_val[voltage_points[0]] - mu_val[voltage_points[1]]
        state["flow"] += d_mu * state["dt"]
        running_state.append("voltage", d_mu)

        if include_screening:
            # Update the vector potential and link variables
            # 3D current density
            J_site = get_observable_on_site(supercurrent_val + normal_current_val, mesh)
            # i: edges, j: sites, k: spatial dimensions
            induced_vector_potential_val = np.asarray(
                einsum("jk, ijk -> ik", J_site, inv_rho)
            )

        new_current = supercurrent_val + normal_current_val
        max_current = np.max(np.abs(old_current))
        converged = (
            max_current > 0
            and np.max(np.abs(new_current - old_current)) / max_current < rtol
        )
        return (
            converged,
            psi_val,
            mu_val,
            supercurrent_val,
            normal_current_val,
            induced_vector_potential_val,
        )

    if seed_solution is None:
        parameters = {
            "psi": psi,
            "mu": mu,
            "supercurrent": np.zeros(len(mesh.edge_mesh.edges)),
            "normal_current": np.zeros(len(mesh.edge_mesh.edges)),
            "induced_vector_potential": np.zeros((len(mesh.edge_mesh.edges), 2)),
        }
    else:
        seed_data = seed_solution.tdgl_data
        parameters = {
            "psi": seed_data.psi,
            "mu": seed_data.mu,
            "supercurrent": seed_data.supercurrent,
            "normal_current": seed_data.normal_current,
            "induced_vector_potential": seed_data.induced_vector_potential,
        }

    Runner(
        function=update,
        data_handler=data_handler,
        initial_values=list(parameters.values()),
        names=list(parameters),
        fixed_values=[vector_potential],
        fixed_names=("applied_vector_potential",),
        state={
            "current": source_drain_current,
            "flow": 0,
            "u": u,
            "gamma": gamma,
        },
        running_names=(
            "voltage",
            "current",
        ),
        min_steps=min_steps,
        max_steps=max_steps,
        dt=dt,
        save_every=save_every,
        logger=logger,
        skip=skip,
        miniters=miniters,
    ).run()

    data_handler.close()

    end_time = datetime.now()
    logger.info(f"Simulation ended on {end_time}")
    logger.info(f"Simulation took {end_time - start_time}")

    solution = Solution(
        device=device,
        filename=data_handler.output_path,
        applied_vector_potential=applied_vector_potential,
        source_drain_current=source_drain_current,
        total_seconds=(end_time - start_time).total_seconds(),
        field_units=field_units,
        current_units=current_units,
    )
    return solution
