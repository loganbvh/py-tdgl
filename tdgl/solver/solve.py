import logging
import os
from datetime import datetime
from typing import Callable, Dict, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy import spatial

try:
    import jax
    import jax.numpy as jnp
except (ModuleNotFoundError, ImportError):
    jax = None

from ..device.device import Device, TerminalInfo
from ..em import ureg
from ..finite_volume.operators import MeshOperators
from ..finite_volume.util import get_supercurrent
from ..solution.solution import Solution
from ..sources.constant import ConstantField
from .options import SolverOptions
from .runner import DataHandler, Runner

logger = logging.getLogger(__name__)


if jax is None:
    einsum = np.einsum
else:
    einsum = jnp.einsum


def validate_terminal_currents(
    terminal_currents: Union[Callable, Dict[str, float]],
    terminal_info: Sequence[TerminalInfo],
    solver_options: SolverOptions,
    num_evals: int = 100,
) -> None:
    def check_total_current(currents: Dict[str, float]):
        names = set([t.name for t in terminal_info])
        if unknown := set(currents).difference(names):
            raise ValueError(
                f"Unknown terminal(s) in terminal currents: {list(unknown)}."
            )
        if missing := names.difference(set(currents)):
            raise ValueError(
                f"Missing terminal(s) in terminal currents: {list(missing)}."
            )
        total_current = sum(currents.values())
        if total_current:
            raise ValueError(
                f"The sum of all terminal currents must be 0 (got {total_current:.2e})."
            )

    if callable(terminal_currents):
        times = np.random.default_rng().random(num_evals) * solver_options.solve_time
        for t in times:
            check_total_current(terminal_currents(t))
    else:
        check_total_current(terminal_currents)


def select_pinning_sites(
    pinning_sites: Union[Callable, str],
    sites: np.ndarray,
    areas: np.ndarray,
    interior_indices: np.ndarray,
    length_units: str,
    rng_seed: int,
) -> np.ndarray:
    """Define the pinning sites."""
    if callable(pinning_sites):
        pinning_sites_indices = np.apply_along_axis(pinning_sites, 1, sites)
        (pinning_sites_indices,) = np.where(pinning_sites_indices.astype(bool))
    else:
        if pinning_sites is None:
            pinning_sites = f"0 * {length_units}**(-2)"
        if not isinstance(pinning_sites, str):
            raise ValueError(
                f"Expected pinning sites to be a callable or str, "
                f"but got {type(pinning_sites)}."
            )
        pinning_sites_density = (
            ureg(pinning_sites).to(f"{length_units}**(-2)").magnitude
        )
        site_areas = areas[interior_indices]
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
    return pinning_sites_indices


def solve_for_psi_squared(
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
            two_c_1 = 2 * c + 1
            w2 = np.abs(w) ** 2
            discriminant = two_c_1**2 - 4 * np.abs(z) ** 2 * w2
        except Exception:
            logger.debug("Unable to solve for |psi|^2.", exc_info=True)
            return None, None, None
    if np.any(discriminant < 0):
        return None, None, None
    new_sq_psi = (2 * w2) / (two_c_1 + np.sqrt(discriminant))
    return z, w, new_sq_psi


def solve(
    device: Device,
    options: SolverOptions,
    output_file: Union[os.PathLike, None] = None,
    applied_vector_potential: Union[Callable, float] = 0,
    terminal_currents: Union[Callable, Dict[str, float], None] = None,
    disorder_alpha: Union[float, Callable] = 1,
    pinning_sites: Union[str, Callable, None] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    seed_solution: Union[Solution, None] = None,
):
    """Solve a TDGL model.

    Args:
        device: The :class:`tdgl.Device` to solve.
        options: An instance :class:`tdgl.SolverOptions` specifying the solver
            parameters.
        output_file: Path to an HDF5 file in which to save the data.
            If the file name already exists, a unique name will be generated.
            If ``output_file`` is ``None``, the solver results will not be saved
            to disk.
        applied_vector_potential: A function or :class:`tdgl.Parameter` that computes
            the applied vector potential as a function of position ``(x, y, z)``. If a float
            ``B`` is given, the applied vector potential will be that of a uniform magnetic
            field with strength ``B`` ``field_units``.
        terminal_currents: A dict of ``{terminal_name: current}`` or a callable with signature
            ``func(time: float) -> {terminal_name: current}``, where ``current`` is a float
            in units of ``current_units`` and ``time`` is the dimensionless time.
        disorder_alpha: A float in range (0, 1], or a callable with signature
            ``disorder_alpha(r: Tuple[float, float]) -> alpha``, where ``alpha`` is a float
            in range (0, 1]. If :math:`\\alpha(\\mathbf{r}) < 1` suppresses the superfluid
            density at position :math:`\\mathbf{r}`, which can be used to model
            inhomogeneity. :math:`\\alpha(\\mathbf{r})` is the maximum possible
            superfluid density at position :math:`\\mathbf{r}`.
        pinning_sites: Pinning sites are sites in the mesh where the order parameter
            fixed to :math:`\\psi(\\mathbf{r}, t)=0`. If ``pinning_sites``
            is given as a pint-parseable string with dimensions of ``length ** (-2)``,
            the argument represents the areal density of pinning sites. A corresponding
            number of pinning sites will be chose at random according to ``options.rng_seed``.
            If ``pinning_sites`` is a callable, it must have a signature
            ``pinning_sites(r: Tuple[float, float]) -> bool``, where ``r`` is position
            in the device (in ``device.length_units``). All sites for which
            ``pinning_sites`` returns ``True`` will be fixed as pinning sites.
        field_units: The units for magnetic fields.
        current_units: The units for currents.
        seed_solution: A :class:`tdgl.Solution` instance to use as the initial state
            for the simulation.

    Returns:
        A :class:`tdgl.Solution` instance.
    """

    start_time = datetime.now()
    options.validate()
    rng_seed = int(options.rng_seed)

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
    Bc2 = device.Bc2
    z = device.layer.z0 * xi * np.ones_like(x)
    current_density_scale = (
        4 * ((ureg(current_units) / length_units) / K0).to_base_units()
    )
    assert "dimensionless" in str(current_density_scale.units)
    current_density_scale = current_density_scale.magnitude
    if not callable(applied_vector_potential):
        applied_vector_potential = ConstantField(
            applied_vector_potential,
            field_units=field_units,
            length_units=device.length_units,
        )
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

    # Find the current terminal sites.
    terminal_info = device.terminal_info()
    if terminal_info:
        normal_boundary_index = np.concatenate(
            [t.site_indices for t in terminal_info]
        ).astype(np.int64)
    else:
        normal_boundary_index = np.array([], dtype=np.int64)
    interior_indices = np.setdiff1d(
        np.arange(mesh.x.shape[0], dtype=int), normal_boundary_index
    )
    # Define the source-drain current.
    if terminal_currents and device.voltage_points is None:
        logger.warning(
            "The terminal currents are non-null, but the device has no voltage points."
        )
    if terminal_currents is None:
        terminal_currents = {t.name: 0 for t in device.terminals}
    if callable(terminal_currents):
        current_func = terminal_currents
    else:
        terminal_currents = {k: float(v) for k, v in terminal_currents.items()}

        def current_func(t):
            return terminal_currents

    validate_terminal_currents(current_func, terminal_info, options)

    pinning_sites_indices = select_pinning_sites(
        pinning_sites,
        sites,
        xi**2 * mesh.areas,
        interior_indices,
        device.length_units,
        rng_seed,
    )
    fixed_sites = np.union1d(normal_boundary_index, pinning_sites_indices)

    # Construct finite-volume operators
    operators = MeshOperators(mesh, fixed_sites=fixed_sites)
    operators.build_operators()
    operators.set_link_exponents(vector_potential)
    divergence = operators.divergence
    mu_boundary_laplacian = operators.mu_boundary_laplacian
    mu_laplacian_lu = operators.mu_laplacian_lu
    mu_gradient = operators.mu_gradient

    psi = np.ones_like(mesh.x, dtype=np.complex128)
    psi[fixed_sites] = 0
    mu = np.zeros(len(mesh.x))
    mu_boundary = np.zeros_like(mesh.edge_mesh.boundary_edge_indices, dtype=np.float64)
    # Create the alpha parameter, which is the maximum value of |psi| at each position.
    if not callable(disorder_alpha):
        disorder_alpha_val = disorder_alpha

        def disorder_alpha(r: Tuple[float, float]) -> float:
            return disorder_alpha_val

    alpha = np.apply_along_axis(disorder_alpha, 1, sites)
    if np.any(alpha <= 0) or np.any(alpha > 1):
        raise ValueError("The disorder parameter alpha must be in range (0, 1].")

    if options.include_screening:
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

        if jax is not None:
            # Even without a GPU, jax.numpy.einsum seems to be much faster
            # than numpy.einsum.
            einsum = jnp.einsum
            inv_rho = jax.device_put(inv_rho)

    # Running list of the max abs change in |psi|^2 between subsequent solve steps.
    # This list is used to calculate the adaptive time step.
    d_psi_sq_vals = []
    tentative_dt = options.dt_init

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
        nonlocal tentative_dt
        step = state["step"]
        time = state["time"]
        # Compute the current density for this step
        # and update the current boundary conditions.
        currents = current_func(time)
        for term in terminal_info:
            current_density = (-1 / term.length) * sum(
                current for name, current in currents.items() if name != term.name
            )
            mu_boundary[term.boundary_edge_indices] = (
                current_density_scale * current_density
            )

        # If screening is included, update the link variables in the covariant
        # Laplacian and gradient for psi based on the induced vector potential
        # from the previous iteration.
        if options.include_screening:
            # 3D current density
            J_edge = supercurrent_val + normal_current_val
            J_site = mesh.get_observable_on_site(J_edge)
            # i: edges, j: sites, k: spatial dimensions
            induced_vector_potential_val = np.asarray(
                einsum("jk, ijk -> ik", J_site, inv_rho)
            )
            operators.set_link_exponents(
                vector_potential + induced_vector_potential_val
            )

        # Compute the next time step for psi with the discrete gauge
        # invariant discretization presented in chapter 5 in
        # http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-312132
        abs_sq_psi = np.abs(psi_val) ** 2
        dt_val = tentative_dt
        z, w, new_sq_psi = solve_for_psi_squared(
            psi_val,
            abs_sq_psi,
            alpha,
            gamma,
            u,
            mu_val,
            dt_val,
            operators.psi_laplacian,
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
            z, w, new_sq_psi = solve_for_psi_squared(
                psi_val,
                abs_sq_psi,
                alpha,
                gamma,
                u,
                mu_val,
                dt_val,
                operators.psi_laplacian,
            )
        psi_val = w - z * new_sq_psi
        # Compute the supercurrent and scalar potential
        supercurrent_val = get_supercurrent(
            psi_val, operators.psi_gradient, mesh.edge_mesh.edges
        )
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

        running_state.append("dt", dt_val)
        running_state.append("voltage", d_mu)
        running_state.append("phase_difference", d_theta)

        if options.adaptive:
            # Compute the max abs change in |psi|^2, averaged over the adaptive window,
            # and use it to select a new time step.
            d_psi_sq = np.abs(new_sq_psi - abs_sq_psi).max()
            d_psi_sq_vals.append(d_psi_sq)
            if step > options.adaptive_window:
                new_dt = options.dt_init / max(
                    1e-10, np.mean(d_psi_sq_vals[-options.adaptive_window :])
                )
                tentative_dt = np.clip(new_dt, 0, dt_max)

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

    with DataHandler(output_file=output_file, logger=logger) as data_handler:
        data_handler.save_mesh(mesh)
        Runner(
            function=update,
            options=options,
            data_handler=data_handler,
            initial_values=list(parameters.values()),
            names=list(parameters),
            fixed_values=(vector_potential, alpha),
            fixed_names=("applied_vector_potential", "alpha"),
            state={
                "u": u,
                "gamma": gamma,
            },
            running_names=(
                "voltage",
                "phase_difference",
                "dt",
            ),
            logger=logger,
        ).run()
        end_time = datetime.now()
        logger.info(f"Simulation ended at {end_time}")
        logger.info(f"Simulation took {end_time - start_time}")

        solution = Solution(
            device=device,
            path=data_handler.output_path,
            options=options,
            applied_vector_potential=applied_vector_potential,
            terminal_currents=terminal_currents,
            disorder_alpha=disorder_alpha,
            pinning_sites=pinning_sites,
            total_seconds=(end_time - start_time).total_seconds(),
            field_units=field_units,
            current_units=current_units,
        )
        solution.to_hdf5()
    return solution
