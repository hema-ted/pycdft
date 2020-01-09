# ====== Tutorial: coupling constant for He_2^+ ===============

from pycdft import *
from ase.io import read

cell = read("./He2.cif")
print(r"Initial atomic positions (Ang):")
print(cell.get_positions())
print(cell.get_cell())

# Here we will do a proper run of PyCDFT and include the calculation of the electron coupling.
# Instead of randomly guessing an initial constraint potential $V$, 
# we shall use the `brentq` or `brenth` optimizers to perform a search. 
# For these particular optimizers, we need to specify a range of $V$ where 
# the function space is sampled for both positive and negative values; 
# a range of (-2,2) is typically sufficient.
# 
# As before, we initialize the system in the PyCDFT module.

V = (-2,2)    
print("==================== Initializing Run =========================")
# load sample geometry
cell.positions[1][2] = d 
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
solver1 = CDFTSolver(job="scf", optimizer="brenth",sample=sample, dft_driver=qboxdriver)
solver2 = solver1.copy()
    
# add constraint to two solvers
ChargeTransferConstraint(
    sample=solver1.sample,
    donor=Fragment(solver1.sample, solver1.sample.atoms[0:1]),
    acceptor=Fragment(solver1.sample, solver1.sample.atoms[1:2]),
    V_brak=V,       # <--- initial guess of constraint potential
    N0=1,           # <--- target electron number
    N_tol=1E-6      # <--- convergence tolerance of N0
)
ChargeTransferConstraint(
    sample=solver2.sample, 
    donor=Fragment(solver2.sample, solver2.sample.atoms[0:1]),
    acceptor=Fragment(solver2.sample, solver2.sample.atoms[1:2]),
    V_brak=V,
    N0=-1, 
    N_tol=1E-6
)


# And performed constrained DFT
print("~~~~~~~~~~~~~~~~~~~~ Applying CDFT ~~~~~~~~~~~~~~~~~~~~")
print("---- solver A ------")
solver1.solve()
print("---- solver B ------")
solver2.solve()

# Finally, we call upon the routines for calculating electronic coupling. 
# An example output is given in ./reference
print("~~~~~~~~~~~~~~~~~~~~ Calculating coupling ~~~~~~~~~~~~~~~~~~~~")
compute_elcoupling(solver1, solver2)
    
print("=========== "+ str(d)+" Bohr"+" =======================")
print("==================== JOB DONE =========================")
