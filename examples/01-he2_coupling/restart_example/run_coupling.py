import sys
import subprocess
from pycdft import *
from ase.io import read

# This file contains commands to restart from previous (converged) calculation of CDFT
# For reference, the inputs from the CDFT run is kept

cell = read("./He2.cif")
print(r"Initial atomic positions (Ang):")
print(cell.get_positions())
print(cell.get_cell())

V = -0.703772086888  # this is close to optimal constraint potential, use secant optimizer to refine
    
print("==================== Initializing Run =========================")
# load sample geometry
cell.positions[1][2] = 3.0 
sample = Sample(ase_cell=cell, n1=112, n2=112, n3=112, vspin=1)
print(sample.atoms[1])
    
# load DFT driver, the commands are not essential but kept for reference
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
solver1 = CDFTSolver(job="scf", optimizer="secant",sample=sample, dft_driver=qboxdriver,lrestart=True)
solver2 = solver1.copy()
    
# add constraint to two solvers; the contraints are not essential but kept for reference
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

# Below are the main routines needed to restart a CDFT calculation
print("~~~~~~~~~~~~~~~~~~~~ Restarting CDFT ~~~~~~~~~~~~~~~~~~~~")
print("---- solver A ------")
solver1.restart("wfc-1.xml",[-4.726619,-0.703904]) # input arguments are name of wfc file, [Ed,Ec] from output
solver1.get_Vc("Vc-1.dat")                         # input argument is name of constraint potential file

print("---- solver B ------")
solver2.restart("wfc-2.xml",[-4.726621,-0.703725])
solver2.get_Vc("Vc-2.dat")


# Finally, we call upon the routines for calculating electronic coupling. 
# An example output is given as coupling_restart.out

print("~~~~~~~~~~~~~~~~~~~~ Calculating coupling ~~~~~~~~~~~~~~~~~~~~")
compute_elcoupling(solver1, solver2,close_dft_driver=False)
    
print("==================== JOB DONE =========================")
