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

``pyTDGL`` requires ``python`` ``3.8``,  ``3.9``, or ``3.10``. We recommend creating a new
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

