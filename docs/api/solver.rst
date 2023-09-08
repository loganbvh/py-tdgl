.. py-tdgl

.. _api-solve:

******
Solver
******

Simulating the dynamics of a :class:`tdgl.Device` for a given applied magnetic vector potential
and set of bias currents is as simple as calling the :func:`tdgl.solve` function. The solver
implements the finite-volume and implicit Euler methods described in detail in `Theoretical Background <../background.rst>`_.
The behavior of the solver is determined an instance of :class:`tdgl.SolverOptions`.

The applied vector potential can be specified as a scalar (indicating the vector potential associated with a uniform magnetic field),
a function with signature ``func(x, y, z) -> [Ax, Ay, Az]``, or a :class:`tdgl.Parameter`. The physical units for the
applied vector potential are ``field_units * device.length_units``.

The bias or terminal currents (if any) can be specified as a dictionary like ``terminal_currents = {terminal_name: current}``,
where ``current`` is a ``float`` in units of the specified ``current_units``. For time-dependent applied currents, one can provide
a function with signature ``terminal_currents(time: float) -> {terminal_name: current}``, where ``time`` is the dimensionless time.
In either case, the sum of all terminal currents must be zero at every time step and every terminal in the device must be included 
in the dictionary to ensure current conservation.

.. autofunction:: tdgl.solve

.. autoclass:: tdgl.SolverOptions
    :members:

.. autoenum:: tdgl.solver.options.SparseSolver

.. autoclass:: tdgl.Parameter


