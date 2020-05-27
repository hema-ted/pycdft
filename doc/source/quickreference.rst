.. _quickreference:

Quick Reference
===============

These are quick references for **PyCDFT** usage.

Run PyCDFT interactively
------------------------

Before performing a CDFT calculation, one should always complete the normal ground state DFT calculation first (see Tutorials).

Like most Python packages, **PyCDFT** can be used interactively in a Python terminal or Jupyter notebook, which is ideal for testing purposes.
Example Python commands for using **PyCDFT** are given below.

Besides the Python terminal or Jupyter notebook, one also needs to run the DFT driver in a separate terminal.
For instance, one can use the follow command to execute Qbox in client-server mode as a DFT driver

.. code-block:: bash

   mpirun -np <ntasks> $qb -server qb_cdft.in qb_cdft.out

where qb_cdft.in and qb_cdft.out are input/output file names of Qbox for communicating with **PyCDFT**.

On clusters which use schedulers such as slurm, it may be convenient to execute the DFT driver in an `interactive session <https://rcc.uchicago.edu/docs/using-midway/index.html>`_ .

Run PyCDFT through bash script
------------------------------

For larger jobs, it is recommended to directly execute both **PyCDFT** and the DFT driver in the same job submission script.
An example bash script is

.. code-block:: bash

   export qb="/path/to/qbox"
   mpirun $qb -server qb_cdft.in qb_cdft.out &
   sleep 2
   python -u run_cdft.py > run_cdft.out

where run_cdft.py is a Python script calling **PyCDFT** to perform CDFT calculations (see below for a template).
More example scripts can be found in the /examples directory. Each example comes with a Jupyter notebook that can be converted to a number of formats (e.g., Python script, html, pdf, LaTeX) using

.. code-block:: bash

   jupyter nbconvert --to FORMAT notebook.ipynb

where **FORMAT** takes on values described `here <https://ipython.org/ipython-doc/dev/notebook/nbconvert.html>`_ (e.g., 'script' to convert to Python script).

Template input file
-------------------

A minimum working example for running **PyCDFT** and computing the electron coupling parameter of a He-He+ dimer.

.. literalinclude:: ../../examples/01-he2_coupling/interactive/he2_coupling.py
   :language: python


List of user-defined parameters
-------------------------------
Here we list important user-defined parameters for CDFT calculations.
See the code documentation for more details.

Important parameters for the construction of **CDFTSolver**

   - job: 'scf' for SCF calculation and 'opt' for geometry optimization.
   - optimizer: 'secant', 'bisect', 'brentq', 'brenth' or 'BFGS'. Optimization algorithm used for Langrange multipliers.

Important parameters for the construction of **ChargeConstraint** or **ChargeTransferConstraint**

   - fragment (ChargeConstraint only): a Fragment instance denoting the part of the system to which the constraint applied.
   - donor/acceptor (ChargeTransferConstraint only): a Fragment instance denoting the part of the system serving as the donor/acceptor.
   - N0: target charge number (ChargeConstraint) or charge difference number (ChargeTransferConstraint)
   - N_tol: convergence threshold for N - N0
   - V_brak/V_init: search interval/initial guess for optimizer


FAQ
---

A compilation of helpful hints and common runtime errors. Each item listed has the date of which it was last modified.

1) Things to check for CDFT calculations (05/2020)

   - all files are present or accessible in the directory, i.e., cif structure file, pseudopotentials
   - the cif file and "ground-state" calculation give the same structure
   - the "ground-state" calculation has converged 
   - it is recommended to group atoms by fragment when specifying atom positions
   - PyCDFT is given the correct FFT grid dimensions

2) A note on units (05/2020)

   - PyCDFT uses ASE to read in the structure, which uses Angstroms
   - Qbox uses atomic units (Ry, Bohr)

3) Runtime errors (05/2020)

   Q: I can run a few CDFT iterations of a **CDFTSolver** but **PyCDFT** crashes with (<class 'TypeError'>, TypeError('cannot unpack non-iterable float object')

   A: This could be a sign that the optimizers used in PyCDFT did not converge and threw an error. Adjust inputs to the optimizer and restart the calculation. Further info may be found in the documentation for scipy.optimizer.

Other possible runtime errors in **PyCDFT** are related to an assertion condition failing. If this is the case, check your input files again corresponding to the error given. 
Keep in mind that the **DFTDriver** may also throw runtime errors. 

Debugging
---------

**PyCDFT** provides some functions to inspect intermediate quantities for testing purposes:

.. code-block:: bash

   get_hirsh(CDFTSolver, origin)
   get_hirsh_ct(CDFTSolver, origin)
   get_rho_atom(CDFTSolver, origin)
   get_rho(CDFTSolver, origin)
   get_grad(CDFTSolver, origin)

See **pycdft/pycdft.debug** for details.
