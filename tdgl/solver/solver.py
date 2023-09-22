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

try:
    import pypardiso  # type: ignore
except ModuleNotFoundError:
    pypardiso = None

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


class TDGLSolver:
    def __init__(
        self,
        device: Device,
        options: SolverOptions,
        applied_vector_potential: Union[Callable, float] = 0,
        terminal_currents: Union[Callable, Dict[str, float], None] = None,
        disorder_epsilon: Union[float, Callable] = 1,
        seed_solution: Optional[Solution] = None,
    ):
        self.device = device
        self.options = options
        self.options.validate()
        self.applied_vector_potential = applied_vector_potential
        self.terminal_currents = terminal_currents
        self.disorder_epsilon = disorder_epsilon
        self.seed_solution = seed_solution

        if self.options.sparse_solver is SparseSolver.CUPY:
            import cupy  # type: ignore

            self.xp = cupy
            self.use_cupy = True
        else:
            self.xp = np
            self.use_cupy = False

        mesh = self.device.mesh
        ureg = self.device.ureg
        self.probe_points = device.probe_point_indices
        length_units = ureg(self.device.length_units)
        field_units = options.field_units
        current_units = options.current_units

        edges = mesh.edge_mesh.edges
        self.num_edges = len(edges)
        normalized_directions = mesh.edge_mesh.normalized_directions
        length_units = ureg(device.length_units)
        xi = device.coherence_length.magnitude
        self.u = device.layer.u
        self.gamma = device.layer.gamma
        K0 = device.K0
        A0 = device.A0
        Bc2 = device.Bc2

        # The vector potential is evaluated on the mesh edges,
        # where the edge coordinates are in dimensionful units.
        self.sites = xi * mesh.sites
        self.edge_centers = xi * mesh.edge_mesh.centers
        self.z0 = device.layer.z0 * np.ones(len(self.edge_centers), dtype=float)

        J_scale = 4 * ((ureg(current_units) / length_units) / K0).to_base_units()
        assert "dimensionless" in str(J_scale.units), str(J_scale.units)
        self.J_scale = J_scale = J_scale.magnitude
        self.time_dependent_vector_potential = (
            isinstance(applied_vector_potential, Parameter)
            and applied_vector_potential.time_dependent
        )
        if not callable(applied_vector_potential):
            applied_vector_potential = ConstantField(
                applied_vector_potential,
                field_units=field_units,
                length_units=device.length_units,
            )
        self.applied_vector_potential_ = applied_vector_potential
        # Evaluate the vector potential
        vector_potential_scale = (
            (ureg(field_units) * length_units / (Bc2 * xi * length_units))
            .to_base_units()
            .magnitude
        )
        A_kwargs = dict(t=0) if self.time_dependent_vector_potential else dict()
        vector_potential = self.applied_vector_potential_(
            self.edge_centers[:, 0], self.edge_centers[:, 1], self.z0, **A_kwargs
        )
        vector_potential = np.asarray(vector_potential)[:, :2]
        if vector_potential.shape != self.edge_centers.shape:
            raise ValueError(
                f"Unexpected shape for vector_potential: {vector_potential.shape}."
            )
        vector_potential *= vector_potential_scale
        self.vector_potential_scale = vector_potential_scale
        self.vector_potential = vector_potential

        # Find the current terminal sites.
        self.terminal_info = device.terminal_info()
        self.terminal_names = [term.name for term in self.terminal_info]
        for term_info in self.terminal_info:
            if term_info.length == 0:
                raise ValueError(
                    f"Terminal {term_info.name!r} does not contain any points"
                    " on the boundary of the mesh."
                )

        # Define the source-drain current.
        if terminal_currents and device.probe_points is None:
            logger.warning(
                "The terminal currents are non-null, but the device has no probe points."
            )
        terminal_names = [term.name for term in self.terminal_info]
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

        self.current_func = lambda t: {
            key: J_scale * value for key, value in current_func(t).items()
        }
        validate_terminal_currents(self.current_func, self.terminal_info, self.options)
        self.terminal_current_densities = {name: 0 for name in self.terminal_names}

        if self.terminal_info:
            normal_boundary_index = np.concatenate(
                [t.site_indices for t in self.terminal_info], dtype=np.int64
            )
        else:
            normal_boundary_index = np.array([], dtype=np.int64)

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
        self.operators = operators
        if options.sparse_solver is SparseSolver.PARDISO:
            assert self.operators.mu_laplacian_lu is None
            assert pypardiso is not None

        # Initialize the order parameter and electric potential
        psi_init = np.ones(len(mesh.sites), dtype=np.complex128)
        if terminal_psi is not None:
            psi_init[normal_boundary_index] = terminal_psi
        mu_init = np.zeros(len(mesh.sites))
        mu_boundary = np.zeros_like(mesh.edge_mesh.boundary_edge_indices, dtype=float)
        # Create the epsilon parameter, which sets the local critical temperature.
        if callable(disorder_epsilon):
            epsilon = np.array([float(disorder_epsilon(r)) for r in self.sites])
        else:
            epsilon = disorder_epsilon * np.ones(len(self.sites), dtype=float)
        if np.any(epsilon < -1) or np.any(epsilon > 1):
            raise ValueError("The disorder parameter epsilon must be in range [-1, 1].")
        if self.use_cupy:
            epsilon = cupy.asarray(epsilon)
            mu_boundary = cupy.asarray(mu_boundary)
            normalized_directions = cupy.asarray(normalized_directions)
            vector_potential = cupy.asarray(vector_potential)

        self.psi_init = psi_init
        self.mu_init = mu_init
        self.epsilon = epsilon
        self.mu_boundary = mu_boundary
        self.normalize_directions = normalized_directions
        self.vector_potential = vector_potential

        self.new_A_induced = None
        self.areas = None
        if options.include_screening:
            A_scale = (ureg("mu_0") / (4 * np.pi) * K0 / A0).to(1 / length_units)
            areas = A_scale.magnitude * mesh.areas * xi**2
            if self.use_cupy:
                self.areas = cupy.asarray(areas)
                self.edge_centers = cupy.asarray(self.edge_centers)
                self.sites = cupy.asarray(self.sites)
                self.new_A_induced = cupy.empty((self.num_edges, 2), dtype=float)

        # Running list of the max abs change in |psi|^2 between subsequent solve steps.
        # This list is used to calculate the adaptive time step.
        self.d_psi_sq_vals = []
        self.tentative_dt = options.dt_init
        self.dt_max = options.dt_max if options.adaptive else options.dt_init

    def solve(self) -> Optional[Solution]:
        start_time = datetime.now()
        options = self.options
        options.validate()
        output_file = options.output_file
        seed_solution = self.seed_solution
        num_edges = self.num_edges
        probe_points = self.probe_points
        # Set the initial conditions.
        if self.seed_solution is None:
            parameters = {
                "psi": self.psi_init,
                "mu": self.mu_init,
                "supercurrent": np.zeros(num_edges),
                "normal_current": np.zeros(num_edges),
                "induced_vector_potential": np.zeros((num_edges, 2)),
            }
        else:
            if self.seed_solution.device != self.device:
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
        if self.time_dependent_vector_potential:
            parameters["applied_vector_potential"] = self.vector_potential
            fixed_values = (self.epsilon,)
            fixed_names = ("epsilon",)
        else:
            fixed_values = (self.vector_potential, self.epsilon)
            fixed_names = ("applied_vector_potential", "epsilon")
        if self.use_cupy:
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
            data_handler.save_mesh(self.device.mesh)
            logger.info(
                f"Simulation started at {start_time}"
                f" using solver {options.sparse_solver.value!r}."
            )
            result = Runner(
                function=self.update,
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
                    device=self.device,
                    path=data_handler.output_path,
                    options=options,
                    applied_vector_potential=self.applied_vector_potential_,
                    terminal_currents=self.terminal_currents,
                    disorder_epsilon=self.disorder_epsilon,
                    total_seconds=(end_time - start_time).total_seconds(),
                )
                solution.to_hdf5()
                return solution
            return None

    def update(
        self,
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
        mesh = self.device.mesh
        xp = self.xp
        use_cupy = self.use_cupy
        options = self.options

        operators = self.operators
        divergence = operators.divergence
        psi_laplacian = operators.psi_laplacian
        mu_gradient = operators.mu_gradient
        mu_laplacian = operators.mu_laplacian
        mu_laplacian_lu = operators.mu_laplacian_lu
        mu_boundary_laplacian = operators.mu_boundary_laplacian

        areas = self.areas
        sites = self.sites
        edge_centers = self.edge_centers
        new_A_induced = self.new_A_induced

        vector_potential = self.vector_potential
        current_func = self.current_func
        terminal_info = self.terminal_info
        mu_boundary = self.mu_boundary
        A_scale = self.vector_potential_scale
        applied_vector_potential_ = self.applied_vector_potential_
        normalized_directions = self.normalize_directions
        terminal_current_densities = self.terminal_current_densities

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
                currents.get(name, 0)
                for name in self.terminal_names
                if name != term.name
            )
            # Only update mu_boundary if the terminal current has changed
            if current_density != terminal_current_densities[term.name]:
                terminal_current_densities[term.name] = current_density
                mu_boundary[term.boundary_edge_indices] = current_density

        # Evaluate the time-dependent vector potential and its time-derivative
        dA_dt = 0.0
        if self.time_dependent_vector_potential:
            vector_potential = (
                A_scale
                * applied_vector_potential_(
                    self.edge_centers[:, 0], self.edge_centers[:, 1], self.z0, t=time
                )[:, :2]
            )
            if use_cupy:
                vector_potential = cupy.asarray(vector_potential)
            self.vector_potential = vector_potential
            dA_dt = xp.einsum(
                "ij, ij -> i",
                (vector_potential - A_applied) / dt,
                normalized_directions,
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
                if self.time_dependent_vector_potential:
                    dA_dt = xp.einsum(
                        "ij, ij -> i",
                        (vector_potential + A_induced - A_applied) / dt,
                        normalized_directions,
                    )

            # Adjust the time step and calculate the new the order parameter
            if screening_iteration == 0:
                # Find a new time step only for the first screening iteration.
                dt = self.tentative_dt
            psi, abs_sq_psi, dt = adaptive_euler_step(
                step,
                psi,
                old_sq_psi,
                mu,
                self.epsilon,
                self.gamma,
                self.u,
                dt,
                psi_laplacian,
                options,
            )
            # Compute the supercurrent, scalar potential, and normal current
            supercurrent = operators.get_supercurrent(psi)
            rhs = (divergence @ (supercurrent - dA_dt)) - (
                mu_boundary_laplacian @ mu_boundary
            )
            if options.sparse_solver is SparseSolver.PARDISO:
                mu = pypardiso.spsolve(mu_laplacian, rhs)
            else:
                mu = mu_laplacian_lu(rhs)
            normal_current = -(mu_gradient @ mu) - dA_dt

            if not options.include_screening:
                break

            # Evaluate the induced vector potential.
            J_site = mesh.get_quantity_on_site(
                supercurrent + normal_current, use_cupy=use_cupy
            )
            if use_cupy:
                threads_per_block = 512
                num_blocks = math.ceil(self.num_edges / threads_per_block)
                get_A_induced_cupy(
                    (num_blocks,),
                    (threads_per_block, 2),
                    (J_site, areas, sites, edge_centers, new_A_induced),
                )
            else:
                new_A_induced = get_A_induced_numba(J_site, areas, sites, edge_centers)
            # Update induced vector potential using Polyak's method
            dA = new_A_induced - A_induced
            v.append(
                (1 - options.screening_step_drag) * v[-1]
                + options.screening_step_size * dA
            )
            A_induced = A_induced + v[-1]
            A_induced_vals.append(A_induced)
            if len(A_induced_vals) > 1:
                screening_error = xp.max(
                    xp.linalg.norm(dA, axis=1) / xp.linalg.norm(A_induced, axis=1)
                )
                screening_error = float(screening_error)
                del v[:-2]
                del A_induced_vals[:-2]

        running_state.append("dt", dt)
        if self.probe_points is not None:
            # Update the voltage and phase difference
            running_state.append("mu", mu[self.probe_points])
            running_state.append("theta", xp.angle(psi[self.probe_points]))
        if options.include_screening:
            running_state.append("screening_iterations", screening_iteration)

        if options.adaptive:
            # Compute the max abs change in |psi|^2, averaged over the adaptive window,
            # and use it to select a new time step.
            self.d_psi_sq_vals.append(xp.absolute(abs_sq_psi - old_sq_psi).max())
            window = options.adaptive_window
            if step > window:
                new_dt = options.dt_init / xp.clip(
                    xp.array(self.d_psi_sq_vals[-window:]).mean(),
                    1e-10,
                    xp.inf,
                )
                self.tentative_dt = xp.clip(0.5 * (new_dt + dt), 0, self.dt_max)

        results = (
            dt,
            psi,
            mu,
            supercurrent,
            normal_current,
            A_induced,
        )
        if self.time_dependent_vector_potential:
            results = results + (vector_potential,)
        return results
