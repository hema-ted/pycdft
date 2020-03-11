pycdft.atomic
=============

This subpackage contains:
  - ``*.spavr`` files, pre-computed spherically-averaged charge densities for an isolated atom
    These charge densities are based on the `ONCV pseudopotentials <http://www.quantum-simulation.org/potentials/sg15_oncv/>`_ (v 1.0, 1.1)
  - ``pp.py``, input file for generating ``*.spavr`` files using post-processing routines in `WEST <http://west-code.org/>`_

There is an assumed 5 Angstrom cutoff of the charge density. All charge densities are sampled in 0.02 Angstrom increments, which is harded coded. 

pycdft.atomic.pp 
-----------------

.. automodule:: pycdft.atomic.pp
   :members:
   :undoc-members:
   :show-inheritance:

