import logging
from datetime import datetime
from typing import Callable, Optional, Union

import numpy as np
from scipy import spatial
from scipy.sparse.linalg import splu

try:
    import jax
    import jax.numpy as jnp
except (ModuleNotFoundError, ImportError):
    jax = None

from ..device.device import Device
from ..enums import MatrixType
from ..finite_volume.matrices import MatrixBuilder
from ..finite_volume.util import get_supercurrent
from ..parameter import Parameter
from ..solution import Solution
from ..sources.constant import ConstantField
from .options import SolverOptions
from .runner import DataHandler, Runner

logger = logging.getLogger(__name__)


def _solve_for_psi_squared(
    psi,
    abs_sq_psi,
    alpha,
    gamma,
    u,
    mu,
    dt,
    psi_laplacian,
):
    phase = np.exp(-1j * mu * dt)
    z = phase * gamma**2 / 2 * psi
    w = z * abs_sq_psi + phase * (
        psi
        + (dt / u)
        * np.sqrt(1 + gamma**2 * abs_sq_psi)
        * ((alpha - abs_sq_psi) * psi + psi_laplacian @ psi)
    )
    c = w.real * z.real + w.imag * z.imag
    # Find the modulus squared for the next time step
    two_c_1 = 2 * c + 1
    w2 = np.abs(w) ** 2
    sqrt_arg = two_c_1**2 - 4 * np.abs(z) ** 2 * w2
    if np.any(sqrt_arg < 0):
        return z, w, None
    new_sq_psi = (2 * w2) / (two_c_1 + np.sqrt(sqrt_arg))
    return z, w, new_sq_psi


def solve(
    device: Device,
    output: str,
    options: SolverOptions,
    applied_vector_potential: Optional[Callable] = None,
    source_drain_current: Union[float, Callable] = 0,
    pinning_sites: Union[float, Callable, None] = None,
    complex_time_scale: float = 5.79,
    gamma: float = 10.0,
    field_units: str = "mT",
    current_units: str = "uA",
    include_screening: bool = False,
    seed_solution: Optional[Solution] = None,
    rng_seed: Optional[int] = None,
):

    start_time = datetime.now()

    if rng_seed is None:
        rng_seed = np.random.SeedSequence().entropy

    ureg = device.ureg
    mesh = device.mesh
    sites = device.points
    voltage_points = mesh.voltage_points
    length_units = ureg(device.length_units)
    xi = device.coherence_length
    K0 = device.K0
    # The vector potential is evaluated on the mesh edges,
    # where the edge coordinates are in dimensionful units.
    x = mesh.edge_mesh.x * xi
    y = mesh.edge_mesh.y * xi
    edge_positions = np.array([x, y]).T
    Bc2 = device.Bc2
    z = device.layer.z0 * xi * np.ones_like(x)
    if applied_vector_potential is None:
        applied_vector_potential = ConstantField(
            0,
            field_units=field_units,
            length_units=device.length_units,
        )
    if not isinstance(applied_vector_potential, Parameter):
        applied_vector_potential = Parameter(applied_vector_potential)
    # Evaluate the vector potential
    vector_potential = applied_vector_potential(x, y, z)
    vector_potential = np.asarray(vector_potential)[:, :2]
    shape = vector_potential.shape
    if shape != x.shape + (2,):
        raise ValueError(f"Unexpected shape for vector_potential: {shape}.")

    current_density_scale = ((ureg(current_units) / length_units) / K0).to_base_units()
    current_density_scale = current_density_scale.magnitude

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

    builder = MatrixBuilder(mesh)
    mu_laplacian, _ = builder.build(MatrixType.LAPLACIAN)
    mu_laplacian = mu_laplacian.asformat("csc", copy=False)
    mu_laplacian_lu = splu(mu_laplacian)
    mu_boundary_laplacian = builder.build(MatrixType.NEUMANN_BOUNDARY_LAPLACIAN)
    mu_gradient = builder.build(MatrixType.GRADIENT)
    divergence = builder.build(MatrixType.DIVERGENCE)

    if device.source_terminal is None:
        input_edges_index = np.array([], dtype=np.int64)
        output_edges_index = np.array([], dtype=np.int64)
        input_sites_index = np.array([], dtype=np.int64)
        output_sites_index = np.array([], dtype=np.int64)
    else:
        ix_boundary = mesh.edge_mesh.boundary_edge_indices
        boundary_edge_positions = edge_positions[ix_boundary]
        input_edges_index = device.source_terminal.contains_points(
            boundary_edge_positions,
            index=True,
        )
        output_edges_index = device.drain_terminal.contains_points(
            boundary_edge_positions,
            index=True,
        )
        input_sites_index = np.intersect1d(
            device.source_terminal.contains_points(sites, index=True),
            mesh.boundary_indices,
        )
        output_sites_index = np.intersect1d(
            device.drain_terminal.contains_points(sites, index=True),
            mesh.boundary_indices,
        )
        # edge_lengths = device.edge_lengths[ix_boundary]
        # input_edge_length = edge_lengths[
        #     device.source_terminal.contains_points(boundary_edge_positions, index=True)
        # ].sum()
        # output_edge_length = edge_lengths[
        #     device.source_terminal.contains_points(boundary_edge_positions, index=True)
        # ].sum()

    if callable(source_drain_current):
        current_func = source_drain_current
    else:
        source_drain_current = float(source_drain_current)

        def current_func(t):
            return source_drain_current

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
    builder = builder.with_dirichlet_boundary(
        fixed_sites=fixed_sites
    ).with_link_exponents(link_exponents=vector_potential)
    # Build complex field matrices.
    psi_laplacian, free_rows = builder.build(MatrixType.LAPLACIAN)
    psi_gradient = builder.build(MatrixType.GRADIENT)
    # Initialize the complex field and the scalar potential.
    psi = np.ones_like(mesh.x, dtype=np.complex128)
    psi[fixed_sites] = 0
    mu = np.zeros(len(mesh.x))
    mu_boundary = np.zeros_like(mesh.edge_mesh.boundary_edge_indices, dtype=np.float64)

    # Create the alpha parameter which weakens the complex field if it
    # is less than unity.
    alpha = np.ones_like(mesh.x, dtype=np.float64)

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

    d_psi_sq_vals = []

    def update(
        state,
        running_state,
        psi_val,
        mu_val,
        supercurrent_val,
        normal_current_val,
        induced_vector_potential_val,
        dt_val,
    ):
        step = state["step"]
        time = state["time"]
        current = current_density_scale * current_func(time)
        running_state.append("current", current)
        state["current"] = current

        mu_boundary[input_edges_index] = current
        mu_boundary[output_edges_index] = -current

        nonlocal psi_laplacian, psi_gradient, free_rows

        if include_screening and step > 0:
            builder.link_exponents = vector_potential + induced_vector_potential_val
            psi_laplacian, _ = builder.build(MatrixType.LAPLACIAN, free_rows=free_rows)
            psi_gradient = builder.build(MatrixType.GRADIENT)

        # Compute the next time step for psi with the discrete gauge
        # invariant discretization presented in chapter 5 in
        # http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-312132
        abs_sq_psi = np.abs(psi_val) ** 2
        z, w, new_sq_psi = _solve_for_psi_squared(
            psi_val, abs_sq_psi, alpha, gamma, u, mu_val, dt_val, psi_laplacian
        )
        while new_sq_psi is None:
            old_dt = dt_val
            dt_val = max(options.dt_min, dt_val / 2)
            logger.debug(
                f"\nFailed to converge at step {step} with dt = {old_dt:.3e}."
                f" Retrying with dt = {dt_val:.3e}."
            )
            z, w, new_sq_psi = _solve_for_psi_squared(
                psi_val, abs_sq_psi, alpha, gamma, u, mu_val, dt_val, psi_laplacian
            )
        psi_val = w - z * new_sq_psi

        running_state.append("dt", dt_val)
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
            # Update the vector potential and link variables.
            # 3D current density
            J_site = mesh.get_observable_on_site(supercurrent_val + normal_current_val)
            # i: edges, j: sites, k: spatial dimensions
            induced_vector_potential_val = np.asarray(
                einsum("jk, ijk -> ik", J_site, inv_rho)
            )

        d_psi_sq = np.abs(new_sq_psi - abs_sq_psi).max()
        d_psi_sq_vals.append(d_psi_sq)
        if step > options.adaptive_window:
            new_dt = options.dt_min / max(
                1e-10, 1e3 * np.mean(d_psi_sq_vals[-options.adaptive_window :])
            )
            dt_val = max(options.dt_min, min(options.dt_max, new_dt))

        return (
            psi_val,
            mu_val,
            supercurrent_val,
            normal_current_val,
            induced_vector_potential_val,
            dt_val,
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
        options=options,
        data_handler=data_handler,
        initial_values=list(parameters.values()),
        names=list(parameters),
        fixed_values=(vector_potential,),
        fixed_names=("applied_vector_potential",),
        state={
            "current": current_density_scale * current_func(0),
            "flow": 0,
            "u": u,
            "gamma": gamma,
        },
        running_names=(
            "voltage",
            "current",
            "dt",
        ),
        logger=logger,
    ).run()

    data_handler.close()

    end_time = datetime.now()
    logger.info(f"Simulation ended at {end_time}")
    logger.info(f"Simulation took {end_time - start_time}")

    solution = Solution(
        device=device,
        filename=data_handler.output_path,
        options=options,
        applied_vector_potential=applied_vector_potential,
        source_drain_current=source_drain_current,
        total_seconds=(end_time - start_time).total_seconds(),
        field_units=field_units,
        current_units=current_units,
    )
    return solution