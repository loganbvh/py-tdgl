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

Version 0.2.1
-------------

Release date: 2023-02-07

Changes
=======

* Fix typos in docs (`# 15 <https://github.com/loganbvh/py-tdgl/pull/15>`_).

Version 0.1.1
-------------

Release date: 2023-01-05

Changes
=======

* Removed ``pinning_sites`` argument in :func:`tdgl.solve` (`#10 <https://github.com/loganbvh/py-tdgl/pull/10>`_). Pinning should be implemented using ``disorder_epsilon``.


Version 0.1.0 (initial release)
-------------------------------

Release date: 2023-01-04
