************
Installation
************

.. role:: bash(code)
   :language: bash

.. role:: python(code)
  :language: python

``pyTDGL`` requires ``python`` ``3.8``,  ``3.9``, or ``3.10``. We recommend creating a new
`conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
for ``pyTDGL`` to avoid dependency conflicts with other packages. To create and activate a ``conda`` environment called
``tdgl``, run:

.. code-block:: bash

  conda create --name tdgl python=3.9
  conda activate tdgl

Install via ``pip``
-------------------

- From `GitHub <https://github.com/loganbvh/py-tdgl/>`_:

    .. code-block:: bash
    
      pip install git+https://github.com/loganbvh/py-tdgl.git

- From  PyPI, the Python Package index:
    
    Coming soon...

Editable installation
=====================

To install an editable version of ``pyTDGL`` for development, run:

.. code-block:: bash

  git clone https://github.com/loganbvh/py-tdgl.git
  cd py-tdgl
  pip install -e .[dev,docs]

.. seealso::

  :ref:`Contributing to pyTDGL <about/contributing:Contributing>`

