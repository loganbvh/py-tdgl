.. py-tdgl

.. _api-device:


****************
Device Interface
****************

The ``tdgl.device`` subpackage provides the following functionalities:

- Definition of device material properties: :class:`tdgl.Layer`.
- Definition of device geometry in terms of :class:`tdgl.Polygon` instances.
- Mesh generation for :class:`tdgl.Polygon` and :class:`tdgl.Device` instances.
- Translation between physical units (e.g., microns and microamperes) and the dimensionless units used in TDGL.
- Visualization and serialization of :class:`tdgl.Device` instances.

Device
------

.. autoclass:: tdgl.Layer
    :members:

.. autoclass:: tdgl.Polygon
    :members:

.. autoclass:: tdgl.Device
    :members:

Geometry
--------

.. autofunction:: tdgl.geometry.box

.. autofunction:: tdgl.geometry.circle

.. autofunction:: tdgl.geometry.ellipse

.. autofunction:: tdgl.geometry.rotate

.. autofunction:: tdgl.geometry.translate

Mesh Generation
---------------

.. autofunction:: tdgl.generate_mesh

.. autofunction:: tdgl.optimize_mesh
