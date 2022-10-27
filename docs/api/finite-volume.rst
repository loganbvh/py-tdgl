.. py-tdgl

.. _api-finite-volume:


*********************
Finite Volume Methods
*********************

Finite Volume Meshes
--------------------

.. autoclass:: tdgl.finite_volume.mesh.Mesh
    :members:

.. autoclass:: tdgl.finite_volume.edge_mesh.EdgeMesh
    :members:

.. autoclass:: tdgl.finite_volume.dual_mesh.DualMesh
    :members:

Matrices
--------

.. autoclass:: tdgl.finite_volume.matrices.MatrixBuilder
    :members:

.. autofunction:: tdgl.finite_volume.matrices.build_divergence

.. autofunction:: tdgl.finite_volume.matrices.build_gradient

.. autofunction:: tdgl.finite_volume.matrices.build_laplacian

.. autofunction:: tdgl.finite_volume.matrices.build_neumann_boundary_laplacian

.. autoenum:: tdgl.enums.MatrixType

.. autoenum:: tdgl.enums.SparseFormat
