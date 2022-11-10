import logging
from datetime import datetime
from typing import Callable, Tuple, Union

import numpy as np
import scipy.sparse as sp
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
    psi: np.ndarray,
    abs_sq_psi: np.ndarray,
    alpha: np.ndarray,
    gamma: float,
    u: float,
    mu: np.ndarray,
    dt: float,
    psi_laplacian: sp.spmatrix,
) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
    phase = np.exp(-1j * mu * dt)
    z = phase * gamma**2 / 2 * psi
    with np.errstate(all="raise"):
        try:
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
            discriminant = two_c_1**2 - 4 * np.abs(z) ** 2 * w2
        except FloatingPointError:
            return None, None, None
    if np.any(discriminant < 0):
        return None, None, None
    new_sq_psi = (2 * w2) / (two_c_1 + np.sqrt(discriminant))
    return z, w, new_sq_psi


def solve(
    device: Device,
    output_file: str,
    options: SolverOptions,
    applied_vector_potential: Union[Callable, None] = None,
    source_drain_current: Union[float, Callable] = 0,
    pinning_sites: Union[str, Callable, None] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    include_screening: bool = False,
    seed_solution: Union[Solution, None] = None,
    rng_seed: Union[int, None] = None,
):
    """Solve a TDGL model.

    Args:
        device: The :class:`tdgl.Device` to solve.
        output_file: Path to an HDF5 file in which to save the data.
            If the file name already exists, a unique name will be generated.
        applied_vector_potential: A function or :class:`tdgl.Parameter` that computes
            the applied vector potential as a function of position ``(x, y, z)``.
        source_drain_current: The applied source-drain current. A constant current can
            be specified by a float. A time-dependent current can be specified by
            a callable with signature ``source_drain_current(time: float) -> float``,
            where ``time`` is the dimensionless time.
        pinning_sites: Pinning sites are sites in the mesh where the order parameter
            fixed to :math:`\\psi(\\mathbf{r}, t)=0`. If ``pinning_sites``
            is given as a pint-parseable string with dimensions of ``length ** (-2)``,
            the argument represents the areal density of pinning sites. A corresponding
            number of pinning sites will be chose at random according to ``rng_seed``.
            If ``pinning_sites`` is a callable, it must have a signature
            ``pinning_sites(r: Sequence[float, float]) -> bool``, where ``r`` is position
            in the device (in ``device.length_units``). All sites for which
            ``pinning_sites`` returns ``True`` will be fixed as pinning sites.
        field_units: The units for magnetic fields.
        current_units: The units for currents.
        include_screening: Whether to include screening in the simulation.
        seed_solution: A :class:`tdgl.Solution` instance to use as the initial state
            for the simulation.
        rng_seed: An integer to used as a seed for the pseudorandom number generator.

    Returns:
        A :class:`tdgl.Solution` instance.
    """

    start_time = datetime.now()

    if rng_seed is None:
        rng_seed = np.random.SeedSequence().entropy

    options.validate()

    ureg = device.ureg
    mesh = device.mesh
    sites = device.points
    voltage_points = mesh.voltage_points
    length_units = ureg(device.length_units)
    xi = device.coherence_length
    u = device.layer.u
    gamma = device.layer.gamma
    K0 = device.K0
    # The vector potential is evaluated on the mesh edges,
    # where the edge coordinates are in dimensionful units.
    x = mesh.edge_mesh.x * xi
    y = mesh.edge_mesh.y * xi
    edge_positions = np.array([x, y]).T
    Bc2 = device.Bc2
    z = device.layer.z0 * xi * np.ones_like(x)
    current_density_scale = ((ureg(current_units) / length_units) / K0).to_base_units()
    current_density_scale = current_density_scale.magnitude
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
    vector_potential = (
        (vector_potential * ureg(field_units) * length_units)
        / (Bc2 * xi * length_units)
    ).to_base_units()
    assert "dimensionless" in str(vector_potential.units)
    vector_potential = vector_potential.magnitude

    if options.adaptive:
        dt_max = options.dt_max
        max_solve_retries = options.max_solve_retries
    else:
        dt_max = options.dt_init
        max_solve_retries = 0

    data_handler = DataHandler(
        mesh,
        output_file=output_file,
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
    # Find the source and drain terminal sites.
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
    normal_boundary_index = np.sort(
        np.concatenate([input_sites_index, output_sites_index])
    ).astype(np.int64)
    interior_indices = np.setdiff1d(
        np.arange(mesh.x.shape[0], dtype=int), normal_boundary_index
    )

    # Define the source-drain current.
    if source_drain_current and device.voltage_points is None:
        logger.warning(
            "The source-drain current is non-null, but the device has no voltage points."
        )
    if callable(source_drain_current):
        current_func = source_drain_current
    else:
        source_drain_current = float(source_drain_current)

        def current_func(t):
            return source_drain_current

    # Define the pinning sites.
    if callable(pinning_sites):
        pinning_sites_indices = np.apply_along_axis(pinning_sites, 1, sites)
        (pinning_sites_indices,) = np.where(pinning_sites_indices.astype(bool))
    else:
        if pinning_sites is None:
            pinning_sites = f"0 * {device.length_units}**(-2)"
        if not isinstance(pinning_sites, str):
            raise ValueError(
                f"Expected pinning sites to be a callable or str, "
                f"but got {type(pinning_sites)}."
            )
        pinning_sites_density = (
            ureg(pinning_sites).to(f"{device.length_units}**(-2)").magnitude
        )
        site_areas = xi**2 * mesh.areas[interior_indices]
        total_area = site_areas.sum()
        n_pinning_sites = int(pinning_sites_density * total_area)
        if n_pinning_sites > len(interior_indices):
            raise ValueError(
                f"The total number of pinning sites ({n_pinning_sites}) for the requested"
                f" areal density of pinning sites ({pinning_sites_density:~P})"
                f" exceeds the total number of interior sites ({len(interior_indices)})."
                f" Try setting a smaller density of pinning sites."
            )
        rng = np.random.default_rng(rng_seed)
        pinning_sites_indices = rng.choice(
            interior_indices,
            size=n_pinning_sites,
            p=site_areas / total_area,
            replace=False,
        )

    fixed_sites = np.union1d(normal_boundary_index, pinning_sites_indices)
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
    # is less than one.
    alpha = np.ones_like(mesh.x, dtype=np.float64)

    if include_screening:
        # Pre-compute the kernel for the screening integral.
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
            # Even without a GPU, jax.numpy.einsum seems to be much faster
            # than numpy.einsum.
            einsum = jnp.einsum
            inv_rho = jax.device_put(inv_rho)

    # Running list of the max abs change in |psi|^2 between subsequent solve steps.
    # This list is used to calculate the adaptive time step.
    d_psi_sq_vals = []

    # This is the function called at each step of the solver.
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
        # Compute the current density for this step
        # and update the current boundary conditions.
        current = current_density_scale * current_func(time)
        running_state.append("total_current", current)
        state["total_current"] = current
        mu_boundary[input_edges_index] = current
        mu_boundary[output_edges_index] = -current

        # If screening is included, update the link variables in the covariant
        # Laplacian and gradient for psi based on the induced vector potential
        # from the previous iteration.
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
        # Adjust the time step if the calculation failed to converge.
        retries = 0
        while new_sq_psi is None:
            retries += 1
            if retries > max_solve_retries:
                raise RuntimeError(
                    f"Solver failed to converge in {options.max_solve_retries} retries at"
                    f" step {step} with dt = {dt_val:.3e}."
                    f" Try using a smaller dt_init."
                )
            old_dt = dt_val
            dt_val = dt_val / 2
            logger.debug(
                f"\nFailed to converge at step {step} with dt = {old_dt:.3e}."
                f" Retrying with dt = {dt_val:.3e}."
            )
            z, w, new_sq_psi = _solve_for_psi_squared(
                psi_val, abs_sq_psi, alpha, gamma, u, mu_val, dt_val, psi_laplacian
            )
        psi_val = w - z * new_sq_psi
        running_state.append("dt", dt_val)
        # Compute the supercurrent and scalar potential
        supercurrent_val = get_supercurrent(psi_val, psi_gradient, mesh.edge_mesh.edges)
        supercurrent_divergence = divergence @ supercurrent_val
        lhs = supercurrent_divergence - (mu_boundary_laplacian @ mu_boundary)
        mu_val = mu_laplacian_lu.solve(lhs)
        normal_current_val = -(mu_gradient @ mu_val)
        # Update the voltage and phase difference
        if device.voltage_points is None:
            d_mu = 0
            d_theta = 0
        else:
            d_mu = mu_val[voltage_points[0]] - mu_val[voltage_points[1]]
            d_theta = np.angle(psi_val[voltage_points[0]]) - np.angle(
                psi_val[voltage_points[1]]
            )
        running_state.append("voltage", d_mu)
        running_state.append("phase_difference", d_theta)

        # If screening is included, update the vector potential and link variables.
        if include_screening:
            # 3D current density
            J_site = mesh.get_observable_on_site(supercurrent_val + normal_current_val)
            # i: edges, j: sites, k: spatial dimensions
            induced_vector_potential_val = np.asarray(
                einsum("jk, ijk -> ik", J_site, inv_rho)
            )
        if options.adaptive:
            # Compute the max abs change in |psi|^2, averaged over the adaptive window,
            # and use it to select a new time step.
            d_psi_sq = np.abs(new_sq_psi - abs_sq_psi).max()
            d_psi_sq_vals.append(d_psi_sq)
            if step > options.adaptive_window:
                new_dt = options.dt_init / max(
                    1e-10, np.mean(d_psi_sq_vals[-options.adaptive_window :])
                )
                dt_val = np.clip(new_dt, 0, dt_max)

        return (
            psi_val,
            mu_val,
            supercurrent_val,
            normal_current_val,
            induced_vector_potential_val,
            dt_val,
        )

    # Set the initial conditions.
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
            "total_current": current_density_scale * current_func(0),
            "u": u,
            "gamma": gamma,
        },
        running_names=(
            "voltage",
            "phase_difference",
            "total_current",
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
        path=data_handler.output_path,
        options=options,
        applied_vector_potential=applied_vector_potential,
        source_drain_current=source_drain_current,
        pinning_sites=pinning_sites,
        rng_seed=rng_seed,
        total_seconds=(end_time - start_time).total_seconds(),
        field_units=field_units,
        current_units=current_units,
    )
    return solution
