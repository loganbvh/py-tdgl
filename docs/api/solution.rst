.. py-tdgl

.. _api-solution:

***************
Post-processing
***************

Solution
--------

The :class:`tdgl.Solution` class provides a convenient container for the results of a TDGL simulation,
including methods for post-processing and visualization results.

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

.. autoclass:: tdgl.Fluxoid
    :show-inheritance:

.. autofunction:: tdgl.make_fluxoid_polygons