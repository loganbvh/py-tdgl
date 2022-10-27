.. py-tdgl

.. _api-device:


****************
Device Interface
****************

The ``tdgl.device`` subpackage provides the following functionalities:

- Definition of device material properties: :class:`tdgl.device.components.Layer`.
- Definition of device geometry in terms of :class:`tdgl.device.components.Polygon` instances.
- Mesh generation for :class:`tdgl.device.components.Polygon` and :class:`tdgl.device.device.Device` instances.
- Translation between physical units (e.g., microns and microamperes) and the dimensionless units used in TDGL.
- Visualization and serialization of :class:`tdgl.device.device.Device` instances.

Device
------

.. autoclass:: tdgl.device.components.Layer
    :members:

.. autoclass:: tdgl.device.components.Polygon
    :members:

.. autoclass:: tdgl.device.device.Device
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

.. autofunction:: tdgl.device.mesh.generate_mesh

.. autofunction:: tdgl.device.mesh.optimize_mesh
