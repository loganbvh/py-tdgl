**********
Change Log
**********

View release history on `PyPI <https://pypi.org/project/tdgl/#history>`_ or `GitHub <https://github.com/loganbvh/py-tdgl/releases>`_.

.. note::

    ``pyTDGL`` uses `semantic versioning <https://semver.org/>`_, with version numbers specified as
    ``MAJOR.MINOR.PATCH``. In particular, note that:

    - Major version zero (0.y.z) is for initial development. Anything MAY change at any time.
      The public API SHOULD NOT be considered stable.
    - Version 1.0.0 defines the public API.

----

.. contents::
    :depth: 2

----

Version 0.7.0
-------------

Release date: 2023-10-05 | `PyPI <https://pypi.org/project/tdgl/0.7.0/>`_ | `GitHub <https://github.com/loganbvh/py-tdgl/releases/tag/v0.7.0>`_

Changes
=======

* Add :meth:`tdgl.Solution.magnetic_moment` to compute the total magnetic dipole moment of a film (`#46 <https://github.com/loganbvh/py-tdgl/pull/46>`_)
* Add option to plot the results of a running simulation in real time (`#47 <https://github.com/loganbvh/py-tdgl/pull/47>`_)

  * To use, you can 1) set ``tdgl.SolverOptions.monitor = True``,
    or 2) run ``tdgl.visualize --input=<h5_file_path> monitor`` from a separate terminal,
    where ``<h5_file_path>`` is the path to the HDF5 file for the currently-running simulation. See :doc:`../api/visualization` for more details.

----

Version 0.6.1
-------------

Release date: 2023-10-03 | `PyPI <https://pypi.org/project/tdgl/0.6.1/>`_ | `GitHub <https://github.com/loganbvh/py-tdgl/releases/tag/v0.6.1>`_

Changes
=======

* Simplify spmatrix _set_many (`#42 <https://github.com/loganbvh/py-tdgl/pull/42>`_)
* Fix typo in solver with time-dependent applied vector potential (`#43 <https://github.com/loganbvh/py-tdgl/pull/43>`_)

----

Version 0.6.0
-------------

Release date: 2023-09-28 | `PyPI <https://pypi.org/project/tdgl/0.6.0/>`_ | `GitHub <https://github.com/loganbvh/py-tdgl/releases/tag/v0.6.0>`_

Changes
=======

* Add GPU support using `CuPy <https://cupy.dev/>`_ (`#36 <https://github.com/loganbvh/py-tdgl/pull/36>`_)

  * Refactor solver into a class, :class:`tdgl.TDGLSolver`, for better readability and easier profiling
  * Add GPU support using CuPy

----

Version 0.5.1
-------------

Release date: 2023-09-21 | `PyPI <https://pypi.org/project/tdgl/0.5.1/>`_ | `GitHub <https://github.com/loganbvh/py-tdgl/releases/tag/v0.5.1>`_

Changes
=======

* Fix scaling issue in screening code (`#37 <https://github.com/loganbvh/py-tdgl/pull/37>`_)

----

Version 0.5.0
-------------

Release date: 2023-09-08 | `PyPI <https://pypi.org/project/tdgl/0.5.0/>`_ | `GitHub <https://github.com/loganbvh/py-tdgl/releases/tag/v0.5.0>`_

Changes
=======

* Add ``--dimensionless`` and ``--axis-scale options`` in ``tdgl.visualize`` (`#30 <https://github.com/loganbvh/py-tdgl/pull/30>`_)
* Add time-dependent ``Parameters`` (`#33 <https://github.com/loganbvh/py-tdgl/pull/33>`_)
* Add umfpack and pardiso solvers (`#35 <https://github.com/loganbvh/py-tdgl/pull/35>`_)

----

Version 0.4.0
-------------

Release date: 2023-08-30 | `PyPI <https://pypi.org/project/tdgl/0.4.0/>`_ | `GitHub <https://github.com/loganbvh/py-tdgl/releases/tag/v0.4.0>`_

Changes
=======

* Use matplotlib tri interpolators (`#25 <https://github.com/loganbvh/py-tdgl/pull/25>`_)

----

Version 0.3.1
-------------

Release date: 2023-07-24 | `PyPI <https://pypi.org/project/tdgl/0.3.1/>`_ | `GitHub <https://github.com/loganbvh/py-tdgl/releases/tag/v0.3.1>`_

Changes
=======

* Add python 3.11 to docs (`#21 <https://github.com/loganbvh/py-tdgl/pull/21>`_)
* Add Google Colab (`#22 <https://github.com/loganbvh/py-tdgl/pull/22>`_)
* Add ``get_current_through_paths`` (`#24 <https://github.com/loganbvh/py-tdgl/pull/24>`_)

----

Version 0.3.0
-------------

Release date: 2023-06-08 | `PyPI <https://pypi.org/project/tdgl/0.3.0/>`_ | `GitHub <https://github.com/loganbvh/py-tdgl/releases/tag/v0.3.0>`_

Changes
=======

* Save dynamics data (`#16 <https://github.com/loganbvh/py-tdgl/pull/16>`_)
* Add autoscale option in animate (`#17 <https://github.com/loganbvh/py-tdgl/pull/17>`_)
* Boundary conditions (`#18 <https://github.com/loganbvh/py-tdgl/pull/18>`_)
  
  * Use `numba <https://numba.pydata.org/>`_ where possible to avoid allocation of large intermediate arrays
  * Allow ``psi != 0`` on transport terminals

* Optimize dual mesh construction (`#20 <https://github.com/loganbvh/py-tdgl/pull/20>`_)
  
  * Significantly speeds up mesh generation for large meshes

----

Version 0.2.1
-------------

Release date: 2023-02-07 | `PyPI <https://pypi.org/project/tdgl/0.2.1/>`_ | `GitHub <https://github.com/loganbvh/py-tdgl/releases/tag/v0.2.1>`_

Changes
=======

* Fix typos in docs (`# 15 <https://github.com/loganbvh/py-tdgl/pull/15>`_).

----

Version 0.1.1
-------------

Release date: 2023-01-05 | `PyPI <https://pypi.org/project/tdgl/0.1.1/>`_ | `GitHub <https://github.com/loganbvh/py-tdgl/releases/tag/v0.1.1>`_

Changes
=======

* Removed ``pinning_sites`` argument in :func:`tdgl.solve` (`#10 <https://github.com/loganbvh/py-tdgl/pull/10>`_). Pinning should be implemented using ``disorder_epsilon``.

----

Version 0.1.0 (initial release)
-------------------------------

Release date: 2023-01-04 | `PyPI <https://pypi.org/project/tdgl/0.1.0/>`_ | `GitHub <https://github.com/loganbvh/py-tdgl/releases/tag/v0.1.0>`_

----
