.. py-tdgl

.. _api-visualization:


*************
Visualization
*************

Plot Solutions
--------------

.. autofunction:: tdgl.plot_solution.plot_currents

.. autofunction:: tdgl.plot_solution.plot_order_parameter

.. autofunction:: tdgl.plot_solution.plot_field_at_positions

.. autofunction:: tdgl.plot_solution.plot_vorticity

.. autofunction:: tdgl.plot_solution.plot_scalar_potential

CLI Tool
--------

The ``tdgl.visualize`` module provides a command line interface (CLI) for animating and exploring TDGL data.

.. argparse::
    :module: tdgl.visualize
    :func: make_parser
    :prog: python -m tdgl.visualize
