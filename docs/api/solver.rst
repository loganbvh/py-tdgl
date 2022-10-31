.. py-tdgl

.. _api-solve:

******
Solver
******

The ``tdgl.solver`` module is the entrypoint for running a TDGL simulation.

Solve
-----

.. autofunction:: tdgl.solver.solve.solve

.. autoclass:: tdgl.solver.runner.SolverOptions
    :members:

.. autoclass:: tdgl.parameter.Parameter


Solution
--------

.. autoclass:: tdgl.solution.Solution
    :members:

Fluxoid
-------

.. autoclass:: tdgl.solution.Fluxoid
    :show-inheritance:

.. autoclass:: tdgl.solution.BiotSavartField
    :show-inheritance:

.. autofunction:: tdgl.fluxoid.make_fluxoid_polygons
