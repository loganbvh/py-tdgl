.. py-tdgl

.. _api-visualization:


*************
Visualization
*************

Plot Solutions
--------------

.. autofunction:: tdgl.plot_currents

.. autofunction:: tdgl.plot_order_parameter

.. autofunction:: tdgl.plot_field_at_positions

.. autofunction:: tdgl.plot_vorticity

.. autofunction:: tdgl.plot_scalar_potential

Plotting Utilities
------------------

.. autofunction:: tdgl.solution.plot_solution.auto_range_iqr

.. autofunction:: tdgl.solution.plot_solution.auto_grid

.. autofunction:: tdgl.solution.plot_solution.non_gui_backend

CLI Tool
--------

The ``tdgl.visualize`` module provides a command line interface (CLI) for animating and exploring TDGL data.

.. argparse::
    :module: tdgl.visualize
    :func: make_parser
    :prog: python -m tdgl.visualize
