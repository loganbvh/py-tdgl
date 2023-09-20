# flake8: noqa

import itertools
import logging
import math
from datetime import datetime
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np

try:
    import cupy  # type: ignore
except ModuleNotFoundError:
    cupy = None

from ..device.device import Device, TerminalInfo
from ..finite_volume.operators import MeshOperators
from ..parameter import Parameter
from ..solution.solution import Solution
from ..sources.constant import ConstantField
from .euler import adaptive_euler_step
from .options import SolverOptions, SparseSolver
from .runner import DataHandler, Runner
from .screening import get_A_induced_cupy, get_A_induced_numba

logger = logging.getLogger("solver")


def validate_terminal_currents(
    terminal_currents: Union[Callable, Dict[str, float]],
    terminal_info: Sequence[TerminalInfo],
    solver_options: SolverOptions,
    num_evals: int = 100,
) -> None:
    """Ensure that the terminal currents satisfy current conservation."""

    def check_total_current(currents: Dict[str, float]):
        names = set([t.name for t in terminal_info])
        if unknown := set(currents).difference(names):
            raise ValueError(
                f"Unknown terminal(s) in terminal currents: {list(unknown)}."
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


def solve(
    device: Device,
    options: SolverOptions,
    applied_vector_potential: Union[Callable, float] = 0,
    terminal_currents: Union[Callable, Dict[str, float], None] = None,
    disorder_epsilon: Union[float, Callable] = 1,
    seed_solution: Optional[Solution] = None,
) -> Union[Solution, None]:
    """Solve a TDGL model.

    Args:
        device: The :class:`tdgl.Device` to solve.
        options: An instance :class:`tdgl.SolverOptions` specifying the solver
            parameters.
        applied_vector_potential: A function or :class:`tdgl.Parameter` that computes
            the applied vector potential as a function of position ``(x, y, z)``,
            or of position and time ``(x, y, z, *, t)``. If a float ``B`` is given,
            the applied vector potential will be that of a uniform magnetic field with
            strength ``B`` ``field_units``.
        terminal_currents: A dict of ``{terminal_name: current}`` or a callable with signature
            ``func(time: float) -> {terminal_name: current}``, where ``current`` is a float
            in units of ``current_units`` and ``time`` is the dimensionless time.
        disorder_epsilon: A float in range [-1, 1], or a callable with signature
            ``disorder_epsilon(r: Tuple[float, float]) -> epsilon``, where ``epsilon``
            is a float in range [-1, 1]. Setting
            :math:`\\epsilon(\\mathbf{r})=T_c(\\mathbf{r})/T_c - 1 < 1` suppresses the
            critical temperature at position :math:`\\mathbf{r}`, which can be used
            to model inhomogeneity.
        seed_solution: A :class:`tdgl.Solution` instance to use as the initial state
            for the simulation.

    Returns:
        A :class:`tdgl.Solution` instance. Returns ``None`` if the simulation was
        cancelled during the thermalization stage.
    """

    start_time = datetime.now()
    options.validate()
    current_units = options.current_units
    field_units = options.field_units
    output_file = options.output_file
    dt_max = options.dt_max if options.adaptive else options.dt_init

    ureg = device.ureg
    mesh = device.mesh
    sites = device.points
    edges = mesh.edge_mesh.edges
    num_edges = len(edges)
    edge_directions = mesh.edge_mesh.directions
    length_units = ureg(device.length_units)
    xi = device.coherence_length.magnitude
    probe_points = device.probe_point_indices
    u = device.layer.u
    gamma = device.layer.gamma
    K0 = device.K0
    kappa = device.kappa

    # The vector potential is evaluated on the mesh edges,
    # where the edge coordinates are in dimensionful units.
    x, y = xi * mesh.edge_mesh.centers.T
    Bc2 = device.Bc2
    z = device.layer.z0 * xi * np.ones_like(x)
    J_scale = 4 * ((ureg(current_units) / length_units) / K0).to_base_units()
    assert "dimensionless" in str(J_scale.units), str(J_scale.units)
    J_scale = J_scale.magnitude
    time_dependent_vector_potential = (
        isinstance(applied_vector_potential, Parameter)
        and applied_vector_potential.time_dependent
    )
    if not callable(applied_vector_potential):
        applied_vector_potential = ConstantField(
            applied_vector_potential,
            field_units=field_units,
            length_units=device.length_units,
        )
    applied_vector_potential_ = applied_vector_potential
    # Evaluate the vector potential
    vector_potential_scale = (
        (ureg(field_units) * length_units / (Bc2 * xi * length_units))
        .to_base_units()
        .magnitude
    )
    A_kwargs = dict(t=0) if time_dependent_vector_potential else dict()
    vector_potential = applied_vector_potential_(x, y, z, **A_kwargs)
    vector_potential = np.asarray(vector_potential)[:, :2]
    if vector_potential.shape != x.shape + (2,):
        raise ValueError(
            f"Unexpected shape for vector_potential: {vector_potential.shape}."
        )
    vector_potential *= vector_potential_scale

    # Find the current terminal sites.
    terminal_info = device.terminal_info()
    for term_info in terminal_info:
        if term_info.length == 0:
            raise ValueError(
                f"Terminal {term_info.name!r} does not contain any points"
                " on the boundary of the mesh."
            )
    if terminal_info:
        normal_boundary_index = np.concatenate(
            [t.site_indices for t in terminal_info], dtype=np.int64
        )
    else:
        normal_boundary_index = np.array([], dtype=np.int64)
    # Define the source-drain current.
    if terminal_currents and device.probe_points is None:
        logger.warning(
            "The terminal currents are non-null, but the device has no probe points."
        )
    terminal_names = [term.name for term in terminal_info]
    if terminal_currents is None:
        terminal_currents = {name: 0 for name in terminal_names}
    if callable(terminal_currents):
        current_func = terminal_currents
    else:
        terminal_currents = {
            name: terminal_currents.get(name, 0) for name in terminal_names
        }

        def current_func(t):
            return terminal_currents

    validate_terminal_currents(current_func, terminal_info, options)

    # Construct finite-volume operators
    logger.info("Constructing finite volume operators.")
    terminal_psi = options.terminal_psi
    operators = MeshOperators(
        mesh,
        options.sparse_solver,
        fixed_sites=normal_boundary_index,
        fix_psi=(terminal_psi is not None),
    )
    operators.build_operators()
    operators.set_link_exponents(vector_potential)
    divergence = operators.divergence
    A_laplacian = operators.A_laplacian
    mu_boundary_laplacian = operators.mu_boundary_laplacian
    mu_laplacian_lu = operators.mu_laplacian_lu
    A_laplacian_lu = operators.A_laplacian_lu
    mu_gradient = operators.mu_gradient
    use_cupy = options.sparse_solver is SparseSolver.CUPY
    use_pardiso = options.sparse_solver is SparseSolver.PARDISO
    if use_pardiso:
        assert mu_laplacian_lu is None
        mu_laplacian = operators.mu_laplacian
        import pypardiso  # type: ignore

    # Initialize the order parameter and electric potential
    psi_init = np.ones(len(mesh.sites), dtype=np.complex128)
    if terminal_psi is not None:
        psi_init[normal_boundary_index] = terminal_psi
    mu_init = np.zeros(len(mesh.sites))
    mu_boundary = np.zeros_like(mesh.edge_mesh.boundary_edge_indices, dtype=float)
    # Create the epsilon parameter, which sets the local critical temperature.
    if callable(disorder_epsilon):
        epsilon = np.array([float(disorder_epsilon(r)) for r in sites])
    else:
        epsilon = disorder_epsilon * np.ones(len(sites), dtype=float)
    if np.any(epsilon < -1) or np.any(epsilon > 1):
        raise ValueError("The disorder parameter epsilon must be in range [-1, 1].")
    if use_cupy:
        epsilon = cupy.asarray(epsilon)
        mu_boundary = cupy.asarray(mu_boundary)
        edge_directions = cupy.asarray(edge_directions)
        vector_potential = cupy.asarray(vector_potential)

    new_A_induced = laplacian_A_applied = None
    if options.include_screening:
        A_scale = (ureg("mu_0") / (4 * np.pi) * K0 / Bc2).to_base_units().magnitude
        areas = A_scale * mesh.areas
        edge_centers = mesh.edge_mesh.centers
        xp = cupy if use_cupy else np
        A_dot_dr = xp.einsum("ij, ij -> i", vector_potential, edge_directions)
        if use_cupy:
            areas = cupy.asarray(areas)
            edge_centers = cupy.asarray(edge_centers)
            sites = cupy.asarray(sites)
            new_A_induced = cupy.empty((num_edges, 2), dtype=float)
        laplacian_A_applied = A_laplacian @ mesh.get_quantity_on_site(
            A_dot_dr, use_cupy=use_cupy
        )

    # Running list of the max abs change in |psi|^2 between subsequent solve steps.
    # This list is used to calculate the adaptive time step.
    d_psi_sq_vals = []
    tentative_dt = options.dt_init
    # Parameters for the self-consistent screening calculation.
    step_size = options.screening_step_size
    drag = options.screening_step_drag

    terminal_current_densities = {name: 0 for name in terminal_names}

    def update(
        state,
        running_state,
        dt,
        *,
        psi,
        mu,
        supercurrent,
        normal_current,
        induced_vector_potential,
        applied_vector_potential=None,
    ):
        nonlocal tentative_dt, vector_potential, new_A_induced, laplacian_A_applied
        if isinstance(psi, np.ndarray):
            xp = np
        else:
            assert cupy is not None
            assert isinstance(psi, cupy.ndarray)
            xp = cupy
        A_induced = induced_vector_potential
        A_applied = applied_vector_potential
        step = state["step"]
        time = state["time"]
        old_sq_psi = xp.absolute(psi) ** 2
        # Compute the current density for this step
        # and update the current boundary conditions.
        currents = current_func(time)
        for term in terminal_info:
            current_density = (-1 / term.length) * sum(
                currents.get(name, 0) for name in terminal_names if name != term.name
            )
            # Only update mu_boundary if the terminal current has changed
            if current_density != terminal_current_densities[term.name]:
                terminal_current_densities[term.name] = current_density
                mu_boundary[term.boundary_edge_indices] = J_scale * current_density

        # Evaluate the time-dependent vector potential and its time-derivative
        dA_dt = 0.0
        if time_dependent_vector_potential:
            vector_potential = (
                vector_potential_scale
                * applied_vector_potential_(x, y, z, t=time)[:, :2]
            )
            if use_cupy:
                vector_potential = cupy.asarray(vector_potential)
            A_dot_dr = xp.einsum("ij, ij -> i", vector_potential, edge_directions)
            laplacian_A_applied = A_laplacian @ mesh.get_quantity_on_site(
                A_dot_dr, use_cupy=use_cupy
            )
            dA_dt = xp.einsum(
                "ij, ij -> i",
                (vector_potential - A_applied) / dt,
                edge_directions,
            )
            if not options.include_screening:
                if xp.any(xp.absolute(dA_dt) > 0):
                    operators.set_link_exponents(vector_potential)
        else:
            assert A_applied is None

        screening_error = np.inf
        A_induced_vals = []
        v = [0]  # Velocity for Polyak's method
        # This loop runs only once if options.include_screening is False
        for screening_iteration in itertools.count():
            if screening_error < options.screening_tolerance:
                break  # Screening calculation converged.
            if screening_iteration > options.max_iterations_per_step:
                raise RuntimeError(
                    f"Screening calculation failed to converge at step {step} after"
                    f" {options.max_iterations_per_step} iterations. Relative error in"
                    f" induced vector potential: {screening_error:.2e}"
                    f" (tolerance: {options.screening_tolerance:.2e})."
                )
            if options.include_screening:
                # Update the link variables in the covariant Laplacian and gradient
                # for psi based on the induced vector potential from the previous iteration.
                operators.set_link_exponents(vector_potential + A_induced)
                if time_dependent_vector_potential:
                    dA_dt = xp.einsum(
                        "ij, ij -> i",
                        (vector_potential + A_induced - A_applied) / dt,
                        edge_directions,
                    )

            # Adjust the time step and calculate the new the order parameter
            if screening_iteration == 0:
                # Find a new time step only for the first screening iteration.
                dt = tentative_dt
            psi, abs_sq_psi, dt = adaptive_euler_step(
                step,
                psi,
                old_sq_psi,
                mu,
                epsilon,
                gamma,
                u,
                dt,
                operators.psi_laplacian,
                options,
            )
            # Compute the supercurrent, scalar potential, and normal current
            supercurrent = operators.get_supercurrent(psi)
            rhs = (divergence @ supercurrent) - (mu_boundary_laplacian @ mu_boundary)
            if use_pardiso:
                mu = pypardiso.spsolve(mu_laplacian, rhs)
            else:
                mu = mu_laplacian_lu(rhs)
            normal_current = -(mu_gradient @ mu)
            if time_dependent_vector_potential:
                normal_current -= dA_dt

            if not options.include_screening:
                break

            # Evaluate the induced vector potential.
            J_site = mesh.get_quantity_on_site(
                supercurrent + normal_current, use_cupy=use_cupy
            )
            # if use_cupy:
            #     threads_per_block = 512
            #     num_blocks = math.ceil(num_edges / threads_per_block)
            #     get_A_induced_cupy(
            #         (num_blocks,),
            #         (threads_per_block, 2),
            #         (J_site, areas, sites, edge_centers, new_A_induced),
            #     )
            # else:
            #     new_A_induced = get_A_induced_numba(J_site, areas, sites, edge_centers)
            rhs = -(laplacian_A_applied + (1 / kappa**2 * J_site))
            if use_pardiso:
                new_A_induced = pypardiso.spsolve(A_laplacian, rhs)
            else:
                new_A_induced = A_laplacian_lu(rhs)
            new_A_induced = new_A_induced[edges].mean(axis=1)
            # Update induced vector potential using Polyak's method
            dA = new_A_induced - A_induced
            v.append((1 - drag) * v[-1] + step_size * dA)
            A_induced = A_induced + v[-1]
            A_induced_vals.append(A_induced)
            if len(A_induced_vals) > 1:
                screening_error = xp.max(
                    xp.linalg.norm(dA, axis=1) / xp.linalg.norm(A_induced, axis=1)
                )
                screening_error = float(screening_error)
                print(screening_error)
                del v[:-2]
                del A_induced_vals[:-2]

        running_state.append("dt", dt)
        if probe_points is not None:
            # Update the voltage and phase difference
            running_state.append("mu", mu[probe_points])
            running_state.append("theta", xp.angle(psi[probe_points]))
        if options.include_screening:
            running_state.append("screening_iterations", screening_iteration)

        if options.adaptive:
            # Compute the max abs change in |psi|^2, averaged over the adaptive window,
            # and use it to select a new time step.
            d_psi_sq_vals.append(float(xp.absolute(abs_sq_psi - old_sq_psi).max()))
            window = options.adaptive_window
            if step > window:
                new_dt = options.dt_init / max(1e-10, np.mean(d_psi_sq_vals[-window:]))
                tentative_dt = np.clip(0.5 * (new_dt + dt), 0, dt_max)

        results = (
            dt,
            psi,
            mu,
            supercurrent,
            normal_current,
            A_induced,
        )
        if time_dependent_vector_potential:
            results = results + (vector_potential,)
        return results

    # Set the initial conditions.
    if seed_solution is None:
        parameters = {
            "psi": psi_init,
            "mu": mu_init,
            "supercurrent": np.zeros(num_edges),
            "normal_current": np.zeros(num_edges),
            "induced_vector_potential": np.zeros((num_edges, 2)),
        }
    else:
        if seed_solution.device != device:
            raise ValueError(
                "The seed_solution.device must be equal to the device being simulated."
            )
        seed_data = seed_solution.tdgl_data
        parameters = {
            "psi": seed_data.psi,
            "mu": seed_data.mu,
            "supercurrent": seed_data.supercurrent,
            "normal_current": seed_data.normal_current,
            "induced_vector_potential": seed_data.induced_vector_potential,
        }
    if time_dependent_vector_potential:
        parameters["applied_vector_potential"] = vector_potential
        fixed_values = (epsilon,)
        fixed_names = ("epsilon",)
    else:
        fixed_values = (vector_potential, epsilon)
        fixed_names = ("applied_vector_potential", "epsilon")
    if use_cupy:
        assert cupy is not None
        for key, val in parameters.items():
            parameters[key] = cupy.asarray(val)
        fixed_values = tuple(cupy.asarray(val) for val in fixed_values)
    running_names_and_sizes = {"dt": 1}
    if probe_points is not None:
        running_names_and_sizes["mu"] = len(probe_points)
        running_names_and_sizes["theta"] = len(probe_points)
    if options.include_screening:
        running_names_and_sizes["screening_iterations"] = 1

    with DataHandler(output_file=output_file, logger=logger) as data_handler:
        data_handler.save_mesh(mesh)
        logger.info(
            f"Simulation started at {start_time}"
            f" using solver {options.sparse_solver.value!r}."
        )
        result = Runner(
            function=update,
            options=options,
            data_handler=data_handler,
            initial_values=list(parameters.values()),
            names=list(parameters),
            fixed_values=fixed_values,
            fixed_names=fixed_names,
            running_names_and_sizes=running_names_and_sizes,
            logger=logger,
        ).run()
        end_time = datetime.now()
        logger.info(f"Simulation ended at {end_time}")
        logger.info(f"Simulation took {end_time - start_time}")

        if result:
            solution = Solution(
                device=device,
                path=data_handler.output_path,
                options=options,
                applied_vector_potential=applied_vector_potential_,
                terminal_currents=terminal_currents,
                disorder_epsilon=disorder_epsilon,
                total_seconds=(end_time - start_time).total_seconds(),
            )
            solution.to_hdf5()
            return solution
        return None
