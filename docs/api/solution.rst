.. py-tdgl

.. _api-solution:

***************
Post-Processing
***************

The :class:`tdgl.Solution` class provides a convenient container for the results of a TDGL simulation,
including methods for post-processing and visualization results. Calls to :func:`tdgl.solve` return an
instance of :class:`tdgl.Solution`, which can be used for post-processing. A :class:`tdgl.Solution`
can be serialized to and deserialized from disk.

For each instance ``solution`` of :class:`tdgl.Solution`, the raw data from the TDGL simulation (in dimensionless units)
are stored in ``solution.tdgl_data``, which is an instance of :class:`tdgl.solution.data.TDGLData`. Any data
that is measured at each time step in the simulation, i.e., the measured voltage and phase difference between
the :class:`tdgl.Device`'s ``probe_points``, are stored in ``solution.dynamics``, which is an instance of
:class:`tdgl.solution.data.DynamicsData`.

Overview
--------

Post-processing methods:

* :meth:`tdgl.Solution.interp_current_density`
* :meth:`tdgl.Solution.grid_current_density`
* :meth:`tdgl.Solution.interp_order_parameter`
* :meth:`tdgl.Solution.polygon_fluxoid`
* :meth:`tdgl.Solution.hole_fluxoid`
* :meth:`tdgl.Solution.boundary_phases`
* :meth:`tdgl.Solution.current_through_path`
* :meth:`tdgl.Solution.field_at_position`
* :meth:`tdgl.Solution.vector_potential_at_position`
* :meth:`tdgl.solution.data.DynamicsData.mean_voltage`

Visualization methods:

* :meth:`tdgl.Solution.plot_currents`
* :meth:`tdgl.Solution.plot_order_parameter`
* :meth:`tdgl.Solution.plot_scalar_potential`
* :meth:`tdgl.Solution.plot_field_at_positions`
* :meth:`tdgl.Solution.plot_vorticity`
* :meth:`tdgl.solution.data.DynamicsData.plot`
* :meth:`tdgl.solution.data.DynamicsData.plot_dt`

I/O methods:

* :meth:`tdgl.Solution.to_hdf5`
* :meth:`tdgl.Solution.from_hdf5`
* :meth:`tdgl.Solution.delete_hdf5`

Solution
--------

.. autoclass:: tdgl.Solution
    :members:

.. autoclass:: tdgl.solution.data.TDGLData
    :members:

.. autoclass:: tdgl.solution.data.DynamicsData
    :members:

.. autoclass:: tdgl.BiotSavartField
    :show-inheritance:

Fluxoid Quantization
--------------------

.. seealso::

    :meth:`tdgl.Solution.polygon_fluxoid`, :meth:`tdgl.Solution.hole_fluxoid`, :meth:`tdgl.Solution.boundary_phases`

.. autoclass:: tdgl.Fluxoid
    :show-inheritance:

.. autofunction:: tdgl.make_fluxoid_polygons