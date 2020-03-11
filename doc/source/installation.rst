.. _installation:

Installation
------------

**PyCDFT** can be installed with **pip**.
First, clone the git repository into a local directory

.. code:: bash

   $ git clone https://github.com/hema-ted/pycdft.git

Then, execute **pip** in the folder containing  **setup.py**

.. code:: bash

   $ pip install -e .

The optional flag -e allows one to install **PyCDFT** in editable mode so one can modify the source code without re-installing the package.

**PyCDFT** depends on the following packages, which will be installed automatically if installed through **pip**:

  - `numpy <https://numpy.org/>`_
  - `scipy <https://scipy.org/>`_
  - `pyFFTW <https://pypi.org/project/pyFFTW/>`_
  - `lxml <https://pypi.org/project/lxml/>`_
  - `ase <https://wiki.fysik.dtu.dk/ase/install.html>`_

In order to use **PyCDFT**, one needs to also install a DFT driver.
Currently, **PyCDFT** supports using Qbox as the DFT driver.
Extension to support other DFT codes may be added in the future.

To install Qbox, please follow the `online instructions <http://qboxcode.org/doc/html/usage/installation.html>`_.

**PyCDFT** has been tested with Python 3.6+ and Qbox 1.67.4+. Note that lower versions of Qbox may not work properly with **PyCDFT**.
