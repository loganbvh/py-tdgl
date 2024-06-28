import inspect
import itertools
import logging
import math
import numbers
import os
from datetime import datetime
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp

try:
    import cupy  # type: ignore
except ImportError:
    cupy = None

try:
    import pypardiso  # type: ignore
except ImportError:
    pypardiso = None

from ..device.device import Device, TerminalInfo
from ..finite_volume.operators import MeshOperators
from ..parameter import Parameter
from ..solution.solution import Solution
from ..sources.constant import ConstantField
from .options import SolverOptions, SparseSolver
from .runner import DataHandler, Runner, RunningState
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


class SolverResult(NamedTuple):
    """A container for the results of a single solve step.

    dt: The time step size used for the solve step
    psi: The order parameter
    mu: The scalar potential
    supercurrent: The supercurrent density
    normal_current: The normal current density
    A_induced: The induced vector potential
    A_applied: The applied vector potential. This will be ``None`` in the case of
        a time-independent vector potential.
    epsilon: The disorder parameter, ``epsilon``. This will be ``None`` in the case of
        a time-independent ``epsilon``.
    """

    dt: float
    psi: np.ndarray
    mu: np.ndarray
    supercurrent: np.ndarray
    normal_current: np.ndarray
    A_induced: np.ndarray
    A_applied: Optional[np.ndarray] = None
    epsilon: Optional[np.ndarray] = None


class TDGLSolver:
    """Solver for a TDGL model.

    An instance of :class:`tdgl.TDGLSolver` is created and executed
    by calling :func:`tdgl.solve`.

    Args:
        device: The :class:`tdgl.Device` to solve.
        options: An instance :class:`tdgl.SolverOptions` specifying the solver
            parameters.
        applied_vector_potential: A function or :class:`tdgl.Parameter` that computes
            the applied vector potential as a function of position ``(x, y, z)``,
            or of position and time ``(x, y, z, *, t)``. If a float ``B`` is given,
            the applied vector potential will be that of a uniform magnetic field with
            strength ``B`` ``field_units``. If the applied vector potential is time-dependent,
            this argument must be a :class:`tdgl.Parameter`.
        terminal_currents: A dict of ``{terminal_name: current}`` or a callable with signature
            ``func(time: float) -> {terminal_name: current}``, where ``current`` is a float
            in units of ``current_units`` and ``time`` is the dimensionless time.
        disorder_epsilon: A float <= 1, or a function that returns
            :math:`\\epsilon\\leq 1` as a function of position ``r=(x, y)`` or
            position and time ``(x, y, *, t)``.
            Setting :math:`\\epsilon(\\mathbf{r}, t)=T_c/T - 1 < 1` suppresses the
            order parameter at position :math:`\\mathbf{r}=(x, y)`, which can be used
            to model inhomogeneity.
        seed_solution: A :class:`tdgl.Solution` instance to use as the initial state
            for the simulation.
    """

    def __init__(
        self,
        device: Device,
        options: SolverOptions,
        applied_vector_potential: Union[Callable, float] = 0.0,
        terminal_currents: Union[Callable, Dict[str, float], None] = None,
        disorder_epsilon: Union[Callable, float] = 1.0,
        seed_solution: Optional[Solution] = None,
    ):
        self.device = device
        self.options = options
        self.options.validate()
        self.terminal_currents = terminal_currents
        self.seed_solution = seed_solution

        if self.options.gpu:
            assert cupy is not None
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

        self.dynamic_vector_potential = (
            isinstance(applied_vector_potential, Parameter)
            and applied_vector_potential.time_dependent
        )
        if not callable(applied_vector_potential):
            applied_vector_potential = ConstantField(
                applied_vector_potential,
                field_units=field_units,
                length_units=device.length_units,
            )
        self.applied_vector_potential = applied_vector_potential
        # Evaluate the vector potential
        self.A_scale = (
            (ureg(field_units) * length_units / (Bc2 * xi * length_units))
            .to_base_units()
            .magnitude
        )
        A_kwargs = dict(t=0) if self.dynamic_vector_potential else dict()
        current_A_applied = self.applied_vector_potential(
            self.edge_centers[:, 0], self.edge_centers[:, 1], self.z0, **A_kwargs
        )
        current_A_applied = self.A_scale * np.asarray(current_A_applied)[:, :2]
        if current_A_applied.shape != self.edge_centers.shape:
            raise ValueError(
                f"Unexpected shape for vector_potential: {current_A_applied.shape}."
            )

        # Create the epsilon parameter, which sets the local critical temperature.
        if callable(disorder_epsilon):
            argspec = inspect.getfullargspec(disorder_epsilon)
            self.dynamic_epsilon = "t" in argspec.kwonlyargs
            self.vectorized_epsilon = (
                argspec.kwonlydefaults is not None
                and argspec.kwonlydefaults.get("vectorized", False)
            )
        else:
            # epsilon constant as a function of both position and time
            _disorder_epsilon = disorder_epsilon

            def disorder_epsilon(r):
                return _disorder_epsilon * np.ones(len(r), dtype=float)

            self.vectorized_epsilon = True
            self.dynamic_epsilon = False

        self.disorder_epsilon = disorder_epsilon
        kw = dict(t=0) if self.dynamic_epsilon else dict()
        if self.vectorized_epsilon:
            epsilon = disorder_epsilon(self.sites, **kw)
        else:
            epsilon = np.array([float(disorder_epsilon(r, **kw)) for r in self.sites])
        if np.any(epsilon > 1):
            raise ValueError("The disorder parameter epsilon must be <= 1")

        # Clear the Parameter caches
        if isinstance(self.applied_vector_potential, Parameter):
            self.applied_vector_potential._clear_cache()
        if isinstance(self.disorder_epsilon, Parameter):
            self.disorder_epsilon._clear_cache()

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

        J_scale = 4 * ((ureg(current_units) / length_units) / K0).to_base_units()
        assert J_scale.dimensionless, str(J_scale)
        J_scale = J_scale.magnitude
        self.current_func = lambda t: {
            key: J_scale * value for key, value in current_func(t).items()
        }
        validate_terminal_currents(self.current_func, self.terminal_info, self.options)
        terminal_indices = [t.site_indices for t in self.terminal_info]
        if terminal_indices:
            normal_boundary_index = np.concatenate(terminal_indices, dtype=np.int64)
        else:
            normal_boundary_index = np.array([], dtype=np.int64)
        # Cache the terminal current densities at each time step.
        # Only update the mu boundary conditions if the current has changed.
        self.terminal_current_densities = {name: 0 for name in self.terminal_names}

        # Construct finite-volume operators
        terminal_psi = options.terminal_psi
        logger.info("Constructing finite volume operators.")
        operators = MeshOperators(
            mesh,
            options.sparse_solver,
            use_cupy=self.use_cupy,
            fixed_sites=normal_boundary_index,
            fix_psi=(terminal_psi is not None),
        )
        operators.build_operators()
        operators.set_link_exponents(current_A_applied)
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

        if self.use_cupy:
            epsilon = cupy.asarray(epsilon)
            mu_boundary = cupy.asarray(mu_boundary)
            normalized_directions = cupy.asarray(normalized_directions)
            current_A_applied = cupy.asarray(current_A_applied)

        self.psi_init = psi_init
        self.mu_init = mu_init
        self.epsilon = epsilon
        self.mu_boundary = mu_boundary
        self.normalized_directions = normalized_directions
        self.current_A_applied = current_A_applied

        self.new_A_induced = None
        self.areas = None
        if options.include_screening:
            A_scale = (ureg("mu_0") / (4 * np.pi) * K0 / A0).to(1 / length_units)
            self.new_A_induced = np.empty((self.num_edges, 2), dtype=float)
            self.areas = A_scale.magnitude * mesh.areas * xi**2
            if self.use_cupy:
                self.areas = cupy.asarray(self.areas)
                self.edge_centers = cupy.asarray(self.edge_centers)
                self.sites = cupy.asarray(self.sites)
                self.new_A_induced = cupy.asarray(self.new_A_induced)

        # Running list of the max abs change in |psi|^2 between subsequent solve steps.
        # This list is used to calculate the adaptive time step.
        self.d_psi_sq_vals = []
        self.tentative_dt = options.dt_init
        self.dt_max = options.dt_max if options.adaptive else options.dt_init

        if options.monitor:
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    def update_mu_boundary(self, time: float) -> None:
        """Computes the terminal current density for a given time step and
        updates the scalar potential boundary conditions accordingly.

        Args:
            time: The current value of the dimensionless time.
        """
        # Compute the current density for this step
        # and update the current boundary conditions.
        currents = self.current_func(time)
        terminal_current_densities = self.terminal_current_densities
        for terminal in self.terminal_info:
            current_density = (-1 / terminal.length) * sum(
                currents.get(name, 0)
                for name in self.terminal_names
                if name != terminal.name
            )
            # Only update mu_boundary if the terminal current has changed
            if current_density != terminal_current_densities[terminal.name]:
                terminal_current_densities[terminal.name] = current_density
                self.mu_boundary[terminal.boundary_edge_indices] = current_density

    def update_applied_vector_potential(self, time: float) -> np.ndarray:
        """Evaluates the time-dependent vector potential.

        Args:
            time: The current value of the dimensionless time.

        Returns:
            The new value of the applied vector potential.
        """
        A_applied = self.applied_vector_potential(
            self.edge_centers[:, 0], self.edge_centers[:, 1], self.z0, t=time
        )
        A_applied = self.A_scale * A_applied[:, :2]
        if self.use_cupy:
            A_applied = cupy.asarray(A_applied)
        return A_applied

    def update_epsilon(self, time: float) -> np.ndarray:
        """Evaluates the time-dependent disorder parameter :math:`\\epsilon`.

        Args:
            time: The current value of the dimensionless time.

        Returns:
            The new value of :math:`\\epsilon`
        """
        if self.vectorized_epsilon:
            epsilon = self.disorder_epsilon(self.sites, t=time)
        else:
            epsilon = np.array(
                [float(self.disorder_epsilon(r, t=time)) for r in self.sites]
            )
        if self.use_cupy:
            epsilon = cupy.asarray(epsilon)
        return epsilon

    @staticmethod
    def solve_for_psi_squared(
        *,
        psi: np.ndarray,
        abs_sq_psi: np.ndarray,
        mu: np.ndarray,
        epsilon: np.ndarray,
        gamma: float,
        u: float,
        dt: float,
        psi_laplacian: sp.spmatrix,
    ) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        """Solves for :math:`\\psi^{n+1}` and :math:`|\\psi^{n+1}|^2` given
        :math:`\\psi^n` and :math:`\\mu^n`.

        Args:
            psi: The current value of the order parameter, :math:`\\psi^n`
            abs_sq_psi: The current value of the superfluid density, :math:`|\\psi^n|^2`
            mu: The current value of the electric potential, :math:`\\mu^n`
            epsilon: The disorder parameter, :math:`\\epsilon`
            gamma: The inelastic scattering parameter, :math:`\\gamma`.
            u: The ratio of relaxation times for the order parameter, :math:`u`
            dt: The time step
            psi_laplacian: The covariant Laplacian for the order parameter

        Returns:
            ``None`` if the calculation failed to converge, otherwise the new order
            parameter :math:`\\psi^{n+1}` and superfluid density :math:`|\\psi^{n+1}|^2`.
        """
        if isinstance(psi, np.ndarray):
            xp = np
        else:
            assert cupy is not None
            assert isinstance(psi, cupy.ndarray)
            xp = cupy
        U = xp.exp(-1j * mu * dt)
        z = U * gamma**2 / 2 * psi
        with np.errstate(all="raise"):
            try:
                w = z * abs_sq_psi + U * (
                    psi
                    + (dt / u)
                    * xp.sqrt(1 + gamma**2 * abs_sq_psi)
                    * ((epsilon - abs_sq_psi) * psi + psi_laplacian @ psi)
                )
                c = w.real * z.real + w.imag * z.imag
                two_c_1 = 2 * c + 1
                w2 = xp.absolute(w) ** 2
                discriminant = two_c_1**2 - 4 * xp.absolute(z) ** 2 * w2
            except Exception:
                logger.warning("Unable to solve for |psi|^2.", exc_info=True)
                return None
        if xp.any(discriminant < 0):
            return None
        new_sq_psi = (2 * w2) / (two_c_1 + xp.sqrt(discriminant))
        psi = w - z * new_sq_psi
        return psi, new_sq_psi

    def adaptive_euler_step(
        self,
        step: int,
        psi: np.ndarray,
        abs_sq_psi: np.ndarray,
        mu: np.ndarray,
        epsilon: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Updates the order parameter and time step in an adaptive Euler step.

        Args:
            step: The solve step index, :math:`n`
            psi: The current value of the order parameter, :math:`\\psi^n`
            abs_sq_psi: The current value of the superfluid density, :math:`|\\psi^n|^2`
            mu: The current value of the electric potential, :math:`\\mu^n`
            epsilon: The disorder parameter :math:`\\epsilon^n`
            dt: The tentative time step, which will be updated

        Returns:
            :math:`\\psi^{n+1}`, :math:`|\\psi^{n+1}|^2`, and :math:`\\Delta t^{n}`.
        """
        options = self.options
        kwargs = dict(
            psi=psi,
            abs_sq_psi=abs_sq_psi,
            mu=mu,
            epsilon=epsilon,
            gamma=self.gamma,
            u=self.u,
            dt=dt,
            psi_laplacian=self.operators.psi_laplacian,
        )
        result = self.solve_for_psi_squared(**kwargs)
        for retries in itertools.count():
            if result is not None:
                break  # First evaluation of |psi|^2 was successful.
            if not options.adaptive or retries > options.max_solve_retries:
                raise RuntimeError(
                    f"Solver failed to converge in {options.max_solve_retries}"
                    f" retries at step {step} with dt = {dt:.2e}."
                    f" Try using a smaller dt_init."
                )
            kwargs["dt"] = dt = dt * options.adaptive_time_step_multiplier
            result = self.solve_for_psi_squared(**kwargs)
        psi, new_sq_psi = result
        return psi, new_sq_psi, dt

    def solve_for_observables(
        self, psi: np.ndarray, dA_dt: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solves for the scalar potential :math:`\\mu`, the supercurrent density
        :math:`\\mathbf{J}_s`, and the normal current density :math:`\\mathbf{J}_n`.

        Args:
            psi: The order parameter.
            dA_dt: The time-derivative of the vector potential.

        Returns:
            :math:`\\mu`, :math:`\\mathbf{J}_s`, and :math:`\\mathbf{J}_n`
        """
        use_cupy = self.use_cupy
        options = self.options
        use_cupy_solver = options.sparse_solver is SparseSolver.CUPY
        operators = self.operators
        # Compute the supercurrent, scalar potential, and normal current
        supercurrent = operators.get_supercurrent(psi)
        rhs = (operators.divergence @ (supercurrent - dA_dt)) - (
            operators.mu_boundary_laplacian @ self.mu_boundary
        )
        if use_cupy and not use_cupy_solver:
            rhs = cupy.asnumpy(rhs)
        if self.options.sparse_solver is SparseSolver.PARDISO:
            mu = pypardiso.spsolve(operators.mu_laplacian, rhs)
        else:
            mu = operators.mu_laplacian_lu(rhs)
        if use_cupy and not use_cupy_solver:
            mu = cupy.asarray(mu)
        normal_current = -(operators.mu_gradient @ mu) - dA_dt
        return mu, supercurrent, normal_current

    def get_induced_vector_potential(
        self,
        current_density: np.ndarray,
        A_induced_vals: List[np.ndarray],
        velocity: List[np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        """Computes a new value of the induced vector potential based on Polyak's method.

        Args:
            current_density: The total current density :math:`\\mathbf{J}_s + \\mathbf{J}_n`
            A_induced_vals: A running list of the induced vector potential for previous
                iterations of Polyak's method.
            velocity: A running list of the "velocities" for previous iterations of
                Polyak's method.

        Returns:
            A new value for the induced vector potential, and the relative error in the
            induced vector potential between this iteration of Polyak's method and the
            previous iteration.
        """
        xp = self.xp
        use_cupy = self.use_cupy
        options = self.options
        mesh = self.device.mesh
        alpha = options.screening_step_size
        beta = options.screening_step_drag
        # Evaluate the induced vector potential.
        J_site = mesh.get_quantity_on_site(current_density, use_cupy=use_cupy)
        areas = self.areas
        sites = self.sites
        edge_centers = self.edge_centers
        if use_cupy:
            threads_per_block = 512
            num_blocks = math.ceil(self.num_edges / threads_per_block)
            get_A_induced_cupy(
                (num_blocks,),
                (threads_per_block, 2),
                (J_site, areas, sites, edge_centers, self.new_A_induced),
            )
        else:
            get_A_induced_numba(J_site, areas, sites, edge_centers, self.new_A_induced)
        new_A_induced = self.new_A_induced
        # Update induced vector potential using Polyak's method
        A_induced = A_induced_vals[-1]
        dA = new_A_induced - A_induced
        velocity.append((1 - beta) * velocity[-1] + alpha * dA)
        A_induced = A_induced + velocity[-1]
        A_induced_vals.append(A_induced)
        if len(A_induced_vals) > 1:
            numerator = xp.linalg.norm(dA, axis=1)
            denominator = xp.linalg.norm(A_induced, axis=1)
            # Avoid division by zero in the case of zero A_induced
            denominator = xp.maximum(denominator, 1e-20, out=denominator)
            screening_error = float(xp.max(numerator / denominator))
            del velocity[:-2]
            del A_induced_vals[:-2]
        return A_induced, screening_error

    def update(
        self,
        state: Dict[str, numbers.Real],
        running_state: RunningState,
        dt: float,
        *,
        psi: np.ndarray,
        mu: np.ndarray,
        supercurrent: np.ndarray,
        normal_current: np.ndarray,
        induced_vector_potential: np.ndarray,
        applied_vector_potential: Optional[np.ndarray] = None,
        epsilon: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """This method is called at each time step to update the state of the system.

        Args:
            state: The solver state, i.e., the solve step, time, and time step
            running_state: A container for scalar data that is saved at each time step
            dt: The time step for the previous solve step
            psi: The order parameter
            mu: The scalar potential
            supercurrent: The supercurrent density
            normal_current: The normal current density
            induced_vector_potential: The induced vector potential
            applied_vector_potential: The applied vector potential. This will be ``None``
                in the case of a time-independent vector potential.
            epsilon: The disorder parameter ``epsilon``. This will be ``None``
                in the case of a time-independent ``epsilon``.

        Returns:
            A :class:`tdgl.SolverResult` instance for the solve step.
        """
        xp = self.xp
        options = self.options
        operators = self.operators

        step = state["step"]
        time = state["time"]
        A_induced = induced_vector_potential
        prev_A_applied = A_applied = applied_vector_potential

        # Update the scalar potential boundary conditions.
        self.update_mu_boundary(time)

        # Update the applied vector potential.
        dA_dt = 0.0
        current_A_applied = self.current_A_applied
        if self.dynamic_vector_potential:
            current_A_applied = self.update_applied_vector_potential(time)
            dA_dt = xp.einsum(
                "ij, ij -> i",
                (current_A_applied - prev_A_applied) / dt,
                self.normalized_directions,
            )
            if not xp.allclose(current_A_applied, self.current_A_applied):
                # Update the link exponents only if the applied vector potential
                # has actually changed.
                operators.set_link_exponents(current_A_applied)
        else:
            assert A_applied is None
            prev_A_applied = A_applied = current_A_applied
        self.current_A_applied = current_A_applied

        # Update the value of epsilon
        if self.dynamic_epsilon:
            self.epsilon = self.update_epsilon(time)

        epsilon = self.epsilon
        old_sq_psi = xp.absolute(psi) ** 2
        screening_error = np.inf
        A_induced_vals = [A_induced]
        velocity = [0.0]  # Velocity for Polyak's method
        # This loop runs only once if options.include_screening is False
        for screening_iteration in itertools.count():
            if screening_error < options.screening_tolerance:
                break
            if screening_iteration > options.max_iterations_per_step:
                raise RuntimeError(
                    f"Screening calculation failed to converge at step {step} after"
                    f" {options.max_iterations_per_step} iterations. Relative error in"
                    f" induced vector potential: {screening_error:.2e}"
                    f" (tolerance: {options.screening_tolerance:.2e})."
                )

            # Adjust the time step and calculate the new the order parameter
            if screening_iteration == 0:
                # Find a new time step only for the first screening iteration.
                dt = self.tentative_dt

            if options.include_screening:
                # Update the link variables in the covariant Laplacian and gradient
                # for psi based on the induced vector potential from the previous iteration.
                operators.set_link_exponents(current_A_applied + A_induced)

            # Update the order parameter using an adaptive time step
            psi, abs_sq_psi, dt = self.adaptive_euler_step(
                step, psi, old_sq_psi, mu, epsilon, dt
            )
            # Update the scalar potential, supercurrent density, and normal current density
            mu, supercurrent, normal_current = self.solve_for_observables(psi, dA_dt)

            if options.include_screening:
                # Evaluate the induced vector potential
                A_induced, screening_error = self.get_induced_vector_potential(
                    supercurrent + normal_current, A_induced_vals, velocity
                )
            else:
                break

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
            self.d_psi_sq_vals.append(float(xp.absolute(abs_sq_psi - old_sq_psi).max()))
            window = options.adaptive_window
            if step > window:
                new_dt = options.dt_init / max(
                    1e-10, np.mean(self.d_psi_sq_vals[-window:])
                )
                self.tentative_dt = np.clip(0.5 * (new_dt + dt), 0, self.dt_max)

        results = [dt, psi, mu, supercurrent, normal_current, A_induced]
        if self.dynamic_vector_potential:
            results.append(current_A_applied)
        if self.dynamic_epsilon:
            results.append(epsilon)
        return SolverResult(*results)

    def solve(self) -> Optional[Solution]:
        """Runs the solver.

        Returns:
            A :class:`tdgl.Solution` instance. Returns ``None`` if the simulation was
            cancelled during the thermalization stage.
        """
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

        fixed_values = []
        fixed_names = []
        if self.dynamic_vector_potential:
            parameters["applied_vector_potential"] = self.current_A_applied
        else:
            fixed_values.append(self.current_A_applied)
            fixed_names.append("applied_vector_potential")
        if self.dynamic_epsilon:
            parameters["epsilon"] = self.epsilon
        else:
            fixed_values.append(self.epsilon)
            fixed_names.append("epsilon")

        if self.use_cupy:
            # Move arrays to the GPU
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
            if data_handler.tmp_file is not None:
                self.device.to_hdf5(
                    data_handler.tmp_file.create_group("solution/device")
                )
            logger.info(
                f"Simulation started at {start_time}"
                f" using sparse solver {options.sparse_solver.value!r}"
                f" and backend {('CuPy' if self.use_cupy else 'NumPy')!r}."
            )
            runner = Runner(
                function=self.update,
                options=options,
                data_handler=data_handler,
                monitor=options.monitor,
                monitor_update_interval=options.monitor_update_interval,
                initial_values=list(parameters.values()),
                names=list(parameters),
                fixed_values=tuple(fixed_values),
                fixed_names=tuple(fixed_names),
                running_names_and_sizes=running_names_and_sizes,
                logger=logger,
            )
            data_was_generated = runner.run()
            end_time = datetime.now()
            logger.info(f"Simulation ended at {end_time}")
            logger.info(f"Simulation took {end_time - start_time}")

            # Clear the Parameter caches
            if isinstance(self.applied_vector_potential, Parameter):
                self.applied_vector_potential._clear_cache()
            if isinstance(self.disorder_epsilon, Parameter):
                self.disorder_epsilon._clear_cache()

            solution = None
            if data_was_generated:
                solution = Solution(
                    device=self.device,
                    path=data_handler.output_path,
                    options=options,
                    applied_vector_potential=self.applied_vector_potential,
                    terminal_currents=self.terminal_currents,
                    disorder_epsilon=self.disorder_epsilon,
                    total_seconds=(end_time - start_time).total_seconds(),
                )
                solution.to_hdf5()
            return solution
