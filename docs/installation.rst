************
Installation
************

.. image:: images/logo-transparent-large.png
  :width: 300
  :alt: pyTDGL logo.
  :align: center

.. role:: bash(code)
   :language: bash

.. role:: python(code)
  :language: python

``pyTDGL`` requires ``python`` ``3.8``,  ``3.9``, ``3.10``, or ``3.11``. We recommend creating a new
`conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
for ``pyTDGL`` to avoid dependency conflicts with other packages. To create and activate a ``conda`` environment called
``tdgl``, run:

.. code-block:: bash

  conda create --name tdgl python="3.10"
  conda activate tdgl

Install via ``pip``
-------------------

* From  `PyPI <https://pypi.org/project/tdgl/>`_, the Python Package index:
    
    .. code-block:: bash
    
      pip install tdgl

* From `GitHub <https://github.com/loganbvh/py-tdgl/>`_:

    .. code-block:: bash
    
      pip install git+https://github.com/loganbvh/py-tdgl.git

Editable installation
=====================

To install an editable version of ``pyTDGL`` for development, run:

.. code-block:: bash

  git clone https://github.com/loganbvh/py-tdgl.git
  cd py-tdgl
  pip install -e ".[dev,docs]"

.. seealso::

  :ref:`Contributing to pyTDGL <about/contributing:Contributing>`

Optional dependencies
---------------------

``tdgl`` supports multiple solvers for sparse systems of linear equations: `SuperLU <https://portal.nersc.gov/project/sparse/superlu/>`_ (the default),
`UMFPACK <https://people.engr.tamu.edu/davis/suitesparse.html>`_, and `MKL PARDISO <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/onemkl-pardiso-parallel-direct-sparse-solver-iface.html>`_.

``SuperLU``, the default solver, is included in ``scipy`` and therefore requires no additional installation.

The other solvers, ``UMFPACK`` and ``PARDISO``, may outperform ``SuperLU`` on certain problems and certain CPU hardware.
In particular, ``PARDISO`` is optimized for Intel CPUs and, unlike ``SuperLU`` and ``UMFPACK``, is multithreaded.
This means that ``PARDISO`` may perform best when solving models with very large meshes on an Intel CPU.

Your mileage may vary, so we encourage you to try the different solvers if you are looking to optimize the run time of your ``tdgl`` simulations.
The sparse solver can be specified by setting the ``sparse_solver`` attribute of :class:`tdgl.SolverOptions` to one of ``{"superlu", "umfpack", "pardiso"}``.

Installing UMFPACK
==================

``UMFPACK`` requires the `SuiteSparse <https://people.engr.tamu.edu/davis/suitesparse.html>`_ library, which can be installed using ``conda``.

.. code-block:: bash

  # After activating your conda environment for tdgl
  conda install -c conda-forge suitesparse

  pip install swig scikit-umfpack
  # or pip install tdgl[umfpack]


Installing PARDISO
==================

.. note::

  The ``MKL PARDISO`` solver can only be used with Intel CPUs.

``tdgl`` supports the `PyPardiso <https://github.com/haasad/PyPardisoProject>`_ interface to the ``PARDISO`` solver.
``PyPardiso`` can be installed using either ``pip`` or ``conda``.

.. code-block:: bash

  # After activating your conda environment for tdgl
  pip install pypardiso
  # or conda install -c conda-forge pypardiso
  # or pip install tdgl[pardiso]

Verify the installation
-----------------------

To verify your installation by running the ``tdgl`` test suite,
execute the following command in a terminal:

.. code-block:: bash

    python -m tdgl.testing

If you prefer, you can instead run the following commands in a Python session:

.. code-block:: python

    >>> import tdgl.testing
    >>> tdgl.testing.run()

