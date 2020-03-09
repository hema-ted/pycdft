.. _tutorial:

Tutorial
========

Tutorials to demonstrate how to utilize core features of **PyCDFT**.

The following tutorials are included in the release, located in the **examples/** folder:

 - 01-he2_scf: SCF calculation of a He-He+ dimer
 - 02-he2_coupling: calculation of electronic coupling :math:`H_{ab}` of a He-He+ dimer
 - 03-thiophene: calculation of electronic coupling :math:`H_{ab}` of a stacked thiophene dimer
 - 04-thiophene_rotated: calculation of electronic coupling :math:`H_{ab}` of a stacked thiophene dimer with relative rotation

Here, we show the basic usage with 02-he2_coupling.

.. toctree::
   :maxdepth: 1

   tut_files/3.0_he2_coupling.py

He-He+ dimer coupling
---------------------
       
This is a minimum working example for calculating the electronic coupling in a He-He+ dimer
(i.e., two He atoms separated by some distance with one electron removed). 

1) After compiling the DFT driver and installing PyCDFT, run the ground state calculation.

For Qbox
 
.. code-block:: bash

   export qb="/path/to/executable"
   $qb < gs.in > gs.out

This will generate an **xml** file containing the wavefunctions. 
In this example, it is named **gs.xml**.
Then in the same directory, run `Qbox in server mode <qboxcode.org/daoc/html/usage/client-server.html>`_      
 
.. code-block:: bash

   mpirun -np <ntasks> $qb -server qb_cdft.in qb_cdft.out
 
where qb_cdft.\* are files reserved for client-server mode.

Launch separately an instance of Python and run the following commands.
First, we load the structure using the **read** routine from ASE. 
In this example, we choose a separation of 3 Angstroms between He atoms and initialize a corresponding instance of the **Sample** object.

.. code-block:: python

   from pycdft import *
   from ase.io import read

   print("==================== Initializing Run =========================")
   # load sample geometry
   cell = read("./He2.cif")
   d = 3.0 # Angstroms
   cell.positions[1][2] = d 

   sample = Sample(ase_cell=cell, n1=140, n2=140, n3=140, vspin=1)

The **Sample** object requires input of the FFT grid (**n1,n2,n3**). 
In **Qbox**, this can be found with

.. code-block:: bash

   grep np2v gs.out

Next we load an instance of the DFT driver, which in this example is **Qbox**. 
We initialize the DFT driver with several commands used in the self-consistent field calculation. 
See `Qbox documentation <http://qboxcode.org/doc/html/>`_ for more information. 

.. code-block:: python       
   
   # load DFT driver
   qboxdriver = QboxDriver(
       sample=sample,
       init_cmd="load gs.xml \n" 
           "set xc PBE \n" 
           "set wf_dyn PSDA \n" 
           "set_atoms_dyn CG \n" 
           "set scf_tol 1.0E-8 \n",
           scf_cmd="run 0 50 5",
   )
       
Then, we initialize an instance of the CDFT solver itself.
Since we are not relaxation our structure, the job consists of one self-consistent field calculation and skips the calculation of constrained forces.
We first do a search across a range of values (in this case [-1,1]) for the constraint potential using the **brenth** optimizer.
In order calculate the electronic coupling, we must do a CDFT calculation for the initial and final state.
We set the target number of electrons between the donor He atom and acceptor He atom **N0** to 1. 

.. code-block:: python

   # set up CDFT constraints and solver
   solver1 = CDFTSolver(job="scf", optimizer="brenth",sample=sample, dft_driver=qboxdriver)
   solver2 = solver1.copy()
   V = (-1,1) # range of constraint potentials
       
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

We run the CDFT calculation for the initial and final states.

.. code-block:: python
       
   print("~~~~~~~~~~~~~~~~~~~~ Applying CDFT ~~~~~~~~~~~~~~~~~~~~")
   print("---- solver A ------")
   solver1.solve()
   print("---- solver B ------")
   solver2.solve()

Finally, we may calculate the electronic coupling

.. code-block:: python
       
   print("~~~~~~~~~~~~~~~~~~~~ Calculating coupling ~~~~~~~~~~~~~~~~~~~~")
   compute_elcoupling(solver1, solver2)
   print("==================== JOB DONE =========================")

Once you have a good guess for an initial constraint potential, you may switch the optimizer to **secant**.
Replace **V_brak** with **V_init** when initializing the constraints when switching to the **secant** optimizer.

For the He-He+ dimer at 3 Angstrom separation, a good starting guess is 

.. code-block:: python 

   V = -0.704717721359 

for **solver1**. Due to the symmetry of the system, **solver2** will have a very close constraint potential of the opposite sign.
An example output is included in the examples/ folder.

The outputted electronic coupling is

.. code-block:: bash

  |Hab| (H): 0.002145233079196648
  |Hab| (mH): 2.1452330791966476
  |Hab| (eV): 0.05837458088794375

