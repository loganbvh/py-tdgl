.. py-tdgl

.. _api-finite-volume:


*********************
Finite Volume Methods
*********************

Finite Volume Meshes
--------------------

.. autoclass:: tdgl.finite_volume.Mesh
    :members:

.. autoclass:: tdgl.finite_volume.EdgeMesh
    :members:

Matrices
--------

.. autoclass:: tdgl.finite_volume.MeshOperators
    :members:

.. autofunction:: tdgl.finite_volume.operators.build_divergence

.. autofunction:: tdgl.finite_volume.operators.build_gradient

.. autofunction:: tdgl.finite_volume.operators.build_laplacian

.. autofunction:: tdgl.finite_volume.operators.build_neumann_boundary_laplacian
