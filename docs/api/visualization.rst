.. py-tdgl

.. _api-visualization:


*************
Visualization
*************

``tdgl.visualize`` CLI tool
----------------------------

``tdgl.visualize`` is a command line interface (CLI) for animating and interactively viewing the
time- and space-dependent results of TDGL simulations.

.. argparse::
    :module: tdgl.visualize
    :func: make_parser
    :prog: python -m tdgl.visualize

Create animations
-----------------

.. autofunction:: tdgl.visualization.create_animation


Plot solutions
--------------

.. seealso::

    :meth:`tdgl.Solution.plot_currents`, :meth:`tdgl.Solution.plot_order_parameter`,
    :meth:`tdgl.Solution.plot_field_at_positions`, :meth:`tdgl.Solution.plot_scalar_potential`
    :meth:`tdgl.Solution.plot_vorticity`

.. autofunction:: tdgl.plot_currents

.. autofunction:: tdgl.plot_order_parameter

.. autofunction:: tdgl.plot_field_at_positions

.. autofunction:: tdgl.plot_vorticity

.. autofunction:: tdgl.plot_scalar_potential

Plotting utilities
------------------

.. autofunction:: tdgl.visualization.auto_range_iqr

.. autofunction:: tdgl.visualization.auto_grid

.. autofunction:: tdgl.non_gui_backend
