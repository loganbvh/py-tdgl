.. py-tdgl

.. _api-device:


****************
Device Interface
****************

The ``tdgl.device`` subpackage provides the following functionalities:

* Definition of material properties in instances of :class:`tdgl.Layer`.
* Definition of device geometry in terms of :class:`tdgl.Polygon` instances, which can be created from simple
  :ref:`geometric primitives <api/device:Geometry>` using `constructive solid geometry <https://en.wikipedia.org/wiki/Constructive_solid_geometry>`_.
* :ref:`Mesh generation <api/device:Mesh Generation>` for :class:`tdgl.Polygon` and :class:`tdgl.Device` instances.
* Translation between physical units (e.g., microns and microamperes) and the dimensionless units used in TDGL.
* Visualization and serialization of :class:`tdgl.Device` instances.

Overview
--------

Here is a quick overview of the most useful functions and methods for creating and working with
polygons and devices. For a demonstration of how all of these pieces work together in practice,
see `Working with polygons <../notebooks/polygons.ipynb>`_.

* Geometric primitives:
    * :func:`tdgl.geometry.box`
    * :func:`tdgl.geometry.ellipse`
    * :func:`tdgl.geometry.circle`
    * :func:`tdgl.geometry.rotate`
* :class:`tdgl.Polygon` constructive solid geometry methods:
    * :meth:`tdgl.Polygon.union`, :meth:`tdgl.Polygon.from_union`
    * :meth:`tdgl.Polygon.difference`, :meth:`tdgl.Polygon.from_difference`
    * :meth:`tdgl.Polygon.intersection`, :meth:`tdgl.Polygon.from_intersection`
* :class:`tdgl.Polygon` geometrical transformation methods:
    * :meth:`tdgl.Polygon.rotate`
    * :meth:`tdgl.Polygon.translate`
    * :meth:`tdgl.Polygon.scale`
    * :meth:`tdgl.Polygon.buffer`
    * :meth:`tdgl.Polygon.resample`
* :class:`tdgl.Device` geometrical transformation methods:
    * :meth:`tdgl.Device.rotate`
    * :meth:`tdgl.Device.scale`
    * :meth:`tdgl.Device.translate`
    * :meth:`tdgl.Device.translation`
* Visualization methods:
    * :meth:`tdgl.Polygon.plot`
    * :meth:`tdgl.Device.plot`
    * :meth:`tdgl.Device.draw`
* Mesh generation methods:
    * :meth:`tdgl.Polygon.make_mesh`
    * :meth:`tdgl.Device.make_mesh`
    * :meth:`tdgl.Device.mesh_stats_dict`
    * :meth:`tdgl.Device.mesh_stats`
* Utility methods:
    * :meth:`tdgl.Device.copy`
    * :meth:`tdgl.Polygon.copy`
    * :meth:`tdgl.Layer.copy`
    * :meth:`tdgl.Device.contains_points`
    * :meth:`tdgl.Polygon.contains_points`
    * :meth:`tdgl.Polygon.on_boundary`
* I/O methods:
    * :meth:`tdgl.Layer.to_hdf5`, :meth:`tdgl.Layer.from_hdf5`
    * :meth:`tdgl.Polygon.to_hdf5`, :meth:`tdgl.Polygon.from_hdf5`
    * :meth:`tdgl.Device.to_hdf5`, :meth:`tdgl.Device.from_hdf5`

Layer
-----
.. autoclass:: tdgl.Layer
    :members:

Polygon
-------
.. autoclass:: tdgl.Polygon
    :members:

Device
------
.. autoclass:: tdgl.Device
    :members:

Geometry
--------

The :mod:`tdgl.geometry` module contains functions for creating polygons approximating simple
shapes, i.e., rectangles and ellipses, which can be combined using
`constructive solid geometry <https://en.wikipedia.org/wiki/Constructive_solid_geometry>`_
to create complex shapes.

.. autofunction:: tdgl.geometry.box

.. autofunction:: tdgl.geometry.circle

.. autofunction:: tdgl.geometry.ellipse

.. autofunction:: tdgl.geometry.rotate

Mesh Generation
---------------

.. autofunction:: tdgl.generate_mesh
