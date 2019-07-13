
# coding: utf-8

# ## Prerequisite

# After compiling the DFT driver and installing PyCDFT, run the ground state calculation.
# - - - - - - -
# For Qbox
# 
# ```bash
#  export qb="/path/to/executable"
#  $qb < gs.in > gs.out
# ```
# Then in the same directory, run [Qbox in server mode](qboxcode.org/daoc/html/usage/client-server.html) (using interactive queue)
#     
# 
# ```bash
#  mpirun -np <ntasks> $qb -server qb_cdft.in qb_cdft.out
# ```
# 
# where qb_cdft.\* are files reserved in client-server mode.

# ### Tutorial: coupling constant for He$_2^+$

# In[1]:


from pycdft import *
from ase.io import read


# In[2]:


cell = read("./He2.cif")
print(r"Initial atomic positions (Ang):")
print(cell.get_positions())
print(cell.get_cell())


# In[5]:


d = 3.0
#V = (-1,1)
V = -0.704717721359

print("==================== Initializing Run =========================")
# load sample geometry
cell.positions[1][2] = d 
sample = Sample(ase_cell=cell, n1=140, n2=140, n3=140, vspin=1)
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
    #V_brak=V,
    V_init = -0.704717721359,
    N0=1,
    N_tol=1E-6
)
ChargeTransferConstraint(
    sample=solver2.sample, 
    donor=Fragment(solver2.sample, solver2.sample.atoms[0:1]),
    acceptor=Fragment(solver2.sample, solver2.sample.atoms[1:2]),
    #V_brak=V,
    V_init = 0.704670044631,
    N0=-1, 
    N_tol=1E-6
)



    
print("~~~~~~~~~~~~~~~~~~~~ Applying CDFT ~~~~~~~~~~~~~~~~~~~~")
print("---- solver A ------")
solver1.solve()
print("---- solver B ------")
solver2.solve()
    
print("~~~~~~~~~~~~~~~~~~~~ Calculating coupling ~~~~~~~~~~~~~~~~~~~~")
compute_elcoupling(solver1, solver2)
    
print("=========== "+ str(d)+" Bohr"+" =======================")
print("==================== JOB DONE =========================")

