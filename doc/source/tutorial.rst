.. _tutorial:

Tutorial
========

The following tutorials are included in the release, located in the **examples/** folder:

 - 01-he2_coupling: calculation of electronic coupling :math:`H_{ab}` of a He-He+ dimer
 - 02-thiophene: calculation of electronic coupling :math:`H_{ab}` of a stacked thiophene dimer
 - 03-thiophene_rotated: calculation of electronic coupling :math:`H_{ab}` of a stacked thiophene dimer with relative rotation

Here, we show the basic usage of **PyCDFT** with 01-he2_coupling.

Computing electronic coupling of a He-He+ dimer
-----------------------------------------------

This is a minimum working example for computing the electronic coupling of a He-He+ dimer
(two He atoms separated by some distance with one electron removed).

1) After the installation of **PyCDFT** and a DFT driver (**Qbox** is used in this example), perform the ground state DFT calculation with Qbox.
 
.. code-block:: bash

   export qb="/path/to/qbox"
   $mpirun -np <ntasks> $qb < gs.in > gs.out

where /path/to/qbox is the path to the Qbox executable and ntasks denotes the number of MPI processes.

2) In the same directory, execute Qbox in `client-server mode <qboxcode.org/daoc/html/usage/client-server.html>`_.
 
.. code-block:: bash

   $mpirun -np <ntasks> $qb -server qb_cdft.in qb_cdft.out
 
and leave the terminal open throughout the entire calculation, Qbox will response to commands given by **PyCDFT**.

3) In the same directory, open a Python terminal and follow the procedures below.

First, load the atomic structure and construct a **Sample** instance.
In this example, we choose a separation of 3 Angstroms between He atoms, and we used the same FFT grid in Qbox (**n1,n2,n3**). To check the FFT grid used by **Qbox**, simply type **grep np2v gs.out** in the directory containing Qbox output file gs.out.

.. code-block:: python

   from pycdft import *
   from ase.io import read

   # read atomic structure using ASE
   cell = read("./He2_3Ang.cif")

   # construct the Sample instance
   sample = Sample(ase_cell=cell, n1=112, n2=112, n3=112, vspin=1)

Next we set up the CDFT calculations.
To begin with, we construct a QboxDriver instance that provide necessary commands to for **PyCDFT** to communicate with Qbox.
We initialize the QboxDriver by specifying commands used by Qbox in the self-consistent field calculation.
See `Qbox documentation <http://qboxcode.org/doc/html/>`_ for more information. 

.. code-block:: python       

   qboxdriver = QboxDriver(
       sample=sample,
       init_cmd="load gs.xml \n"
           "set xc PBE \n"
           "set wf_dyn PSDA \n"
           "set_atoms_dyn CG \n"
           "set scf_tol 1.0E-8 \n",
       scf_cmd="run 0 50 5",
   )

Then, we construct CDFT solvers (**CDFTSolver**), which orchestrate the entire CDFT calculations.
In order to compute the electronic coupling, we need two CDFT solvers for the initial and final diabatic state, respectively.
After CDFT solvers are constructed, we add constraints to the solvers.
In this example we will apply **ChargeTransferConstraint** to enforce the electron number difference between the two He atoms to be 1.

.. code-block:: python

   solver1 = CDFTSolver(
    job="scf",  # indicate that the calculation is a SCF calculation
    optimizer="brenth", # specifiy the optimizer used for the Lagrangian multiplier
    sample=sample, dft_driver=qboxdriver
   )
   solver2 = solver1.copy()

   V = (2,-2)  # search range for the brenth optimizer

   # add constraint to two solvers
   ChargeTransferConstraint(
       sample=solver1.sample,
       donor=Fragment(solver1.sample, solver1.sample.atoms[0:1]), # based on ordering in cif file
       acceptor=Fragment(solver1.sample, solver1.sample.atoms[1:2]),
       V_brak=V,
       N0=1,       # target number of electrons
       N_tol=1E-6  # numerical tolerance of Hirshfeld weight
   )
   ChargeTransferConstraint(
       sample=solver2.sample, 
       donor=Fragment(solver2.sample, solver2.sample.atoms[0:1]),
       acceptor=Fragment(solver2.sample, solver2.sample.atoms[1:2]),
       V_brak=V,
       N0=-1, 
       N_tol=1E-6
   )

Then, we execute the calculations by calling the **solve** method of **CDFTSolver**

.. code-block:: python
       
   print("---- solver A ------")
   solver1.solve()
   print("---- solver B ------")
   solver2.solve()

Finally, we compute the electronic coupling of the He-He+ dimer based on the two diabatic states obtained from CDFT calculations

.. code-block:: python
       
   compute_elcoupling(solver1, solver2)

The electronic coupling predicted by **PyCDFT** is

.. code-block:: bash

  Hab (H): 0.002145233079196648
  Hab (mH): 2.1452330791966476
  Hab (eV): 0.05837458088794375

Note that if one has a good guess for the Lagrange multipliers in constraint potentials (for instance from previous calculations using smaller kinetic energy cutoff, etc.), it is recommended to use optimizers such as **secant**, which can take a initial guess for the Lagrange multiplier. In this case, the **V_brak** parameter should be replaced by the **V_init** parameter when initializing the constraints.

For the He-He+ dimer separated by 3 Angstrom, a good starting guess is V_init = -0.7 for solver1 and V_init = 0.7 for solver2.

Restarting CDFT calculation for :math:`H_{ab}`
----------------------------------------------

Depending on your computational resources and system, you may want to separately converge each CDFTSolver and save the relevant files for calculating the electronic coupling later.
An example of how to do this in **PyCDFT** is provided in examples/01-he2_coupling/restart_example/.
