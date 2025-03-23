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

``pyTDGL`` requires ``python`` ``3.8``,  ``3.9``, ``3.10``, ``3.11``, ``3.12``, or ``3.13``. We recommend creating a new
`conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
for ``pyTDGL`` to avoid dependency conflicts with other packages. To create and activate a ``conda`` environment called
``tdgl``, run:

.. code-block:: bash

  conda create --name tdgl python="3.12"
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


Alternative sparse solvers
--------------------------

``tdgl`` supports multiple solvers for sparse systems of linear equations: `SuperLU <https://portal.nersc.gov/project/sparse/superlu/>`_ (the default),
`UMFPACK <https://people.engr.tamu.edu/davis/suitesparse.html>`_, and `MKL PARDISO <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/onemkl-pardiso-parallel-direct-sparse-solver-iface.html>`_.

``SuperLU``, the default solver, is included in ``scipy`` and therefore requires no additional installation.

The other solvers, ``UMFPACK`` and ``PARDISO``, may outperform ``SuperLU`` on certain problems and certain CPU hardware.
In particular, ``PARDISO`` is optimized for Intel CPUs and, unlike ``SuperLU`` and ``UMFPACK``, ``PARDISO`` is multithreaded.
This means that ``PARDISO`` may perform best when solving models with very large meshes on Intel CPUs.

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

GPU acceleration
----------------

For users with an NVIDIA or AMD GPU, ``tdgl`` can be accelerated using the `CuPy <https://cupy.dev/>`_ library.
First install the appropriate version of ``cupy`` for your GPU hardware and driver version
(see installation instructions `here <https://docs.cupy.dev/en/stable/install.html>`_).
Then set the ``gpu`` attribute of :class:`tdgl.SolverOptions` to ``True``. Setting ``tdgl.SolverOptions.gpu = True``
means that essentially all portions of the simulation *except* the sparse linear solve used to compute the scalar electric potential
:math:`\mu(\mathbf{r}, t)` will be performed on the GPU. The sparse linear solve will be performed on the CPU using the solver specified by
``tdgl.SolverOptions.sparse_solver`` (by default, ``SuperLU``). One can also perform the sparse linear solve on the GPU using
``cupy`` by setting ``tdgl.SolverOptions.sparse_solver = "cupy"``, however emperically it seems that this is slower than
performing the sparse linear solve on the CPU.

.. important::

  Using ``cupy`` with an NVIDIA GPU requires the `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_. You can check whether the CUDA toolkit is
  installed and on the system path by running 

  .. code-block:: bash

    nvcc --version

  from the command line. The version of ``cupy`` you install must be compatible with the version of the CUDA Toolkit you have installed.
  
  You can install the CUDA Toolkit either directly from the `NVIDIA website <https://developer.nvidia.com/cuda-toolkit>`_
  or from the `NVIDIA conda channel <https://anaconda.org/nvidia>`_. To install the CUDA Toolkit using ``conda``, activate the ``conda`` environment
  for ``tdgl`` and run

  .. code-block:: bash

    conda install cuda -c nvidia

  If you have installed the CUDA Toolkit but ``nvcc --version`` still fails, you may need to update the ``PATH`` and ``LD_LIBRARY_PATH``
  environment variables to point to your CUDA installation.

  .. code-block:: bash

    # If you installed CUDA Toolkit directly from the NVIDIA website,
    # resulting in CUDA being installed in /usr/local/cuda:
    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

    # If you installed CUDA using conda, activate the appropriate conda environment and run:
    export PATH=${CONDA_PREFIX}/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

  The exact path to your CUDA installation may vary depending on operating system and configuration. You may want to add the appropriate
  ``PATH`` and ``LD_LIBRARY_PATH`` modifications to your ``~/.bashrc`` file.

  For more detailed installation instructions, see the `NVIDIA documentation <https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html>`_.

Due to overheads related to transferring data between the CPU and GPU, it is expected that ``cupy`` will provide
a significant speedup only for models with relatively large meshes and/or models that include `screening <notebooks/screening.ipynb>`_.
Please open a `GitHub issue <https://github.com/loganbvh/py-tdgl/issues>`_ if you have any problems using ``tdgl`` with ``cupy``.

.. note::

  Note that ``cupy`` support for AMD GPUs is `currently experimental <https://docs.cupy.dev/en/stable/install.html#using-cupy-on-amd-gpu-experimental>`_.


Verifying the installation
--------------------------

If you would like to verify your installation by running the ``tdgl`` test suite,
execute the following command in a terminal:

.. code-block:: bash

    python -m tdgl.testing

If you prefer, you can instead run the following commands in a Python session:

.. code-block:: python

    >>> import tdgl.testing
    >>> tdgl.testing.run()

