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

.. autoclass:: tdgl.finite_volume.DualMesh
    :members:

Matrices
--------

.. autoclass:: tdgl.finite_volume.MatrixBuilder
    :members:

.. autofunction:: tdgl.finite_volume.matrices.build_divergence

.. autofunction:: tdgl.finite_volume.matrices.build_gradient

.. autofunction:: tdgl.finite_volume.matrices.build_laplacian

.. autofunction:: tdgl.finite_volume.matrices.build_neumann_boundary_laplacian

.. autoenum:: tdgl.enums.MatrixType

.. autoenum:: tdgl.enums.SparseFormat
