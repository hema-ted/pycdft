#!/usr/bin/env python
# coding: utf-8

import subprocess
from pycdft import *
from ase.io import read

cell = read("./He2.cif")
print(r"Initial atomic positions (Ang):")
print(cell.get_positions())
print(cell.get_cell())


# Here we will do a proper run of PyCDFT, this time as an example for how to restart from a previous (converged) calculation.
# We will use the 'secant' optimizer and guess a good value for the constraint potential 
# As before, we initialize the system in the PyCDFT module.

V = -0.703772086888  # this is close to optimal constraint potential, use secant optimizer to refine
    
print("==================== Initializing Run =========================")
# load sample geometry
cell.positions[1][2] = 3.0 
sample = Sample(ase_cell=cell, n1=112, n2=112, n3=112, vspin=1)
print(sample.atoms[1])
    
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
    
# set up CDFT constraints and solver
solver1 = CDFTSolver(job="scf", optimizer="secant",sample=sample, dft_driver=qboxdriver)
solver2 = solver1.copy()
    
# add constraint to two solvers
ChargeTransferConstraint(
    sample=solver1.sample,
    donor=Fragment(solver1.sample, solver1.sample.atoms[0:1]),
    acceptor=Fragment(solver1.sample, solver1.sample.atoms[1:2]),
    V_init = V,
    N0=1,
    N_tol=1E-6
)
ChargeTransferConstraint(
    sample=solver2.sample, 
    donor=Fragment(solver2.sample, solver2.sample.atoms[0:1]),
    acceptor=Fragment(solver2.sample, solver2.sample.atoms[1:2]),
    V_init = -V,
    N0=-1, 
    N_tol=1E-6
)

# And performed constrained DFT
#   Here we rename files from each CDFTSolver, which will be used to 
#      calculate the electronic coupling separately from restart
print("~~~~~~~~~~~~~~~~~~~~ Applying CDFT ~~~~~~~~~~~~~~~~~~~~")
print("---- solver A ------")
solver1.solve()

bashCommand = "mv wfc.xml wfc-1.xml"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

bashCommand = "mv Vc.dat Vc-1.dat"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

print("---- solver B ------")
solver2.solve()

bashCommand = "mv wfc.xml wfc-2.xml"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

bashCommand = "mv Vc.dat Vc-2.dat"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

print("==================== JOB DONE =========================")

