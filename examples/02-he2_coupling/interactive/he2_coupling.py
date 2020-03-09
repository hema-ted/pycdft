# ## Prerequisite

# After compiling the DFT driver and installing PyCDFT, run the ground state calculation.
# - - - - - - -
# For Qbox
# 
# ```bash
#  export qb="/path/to/executable"
#  $qb < gs.in > gs.out
# ```
# Then in the same directory, run [Qbox in server mode](qboxcode.org/daoc/html/usage/client-server.html) (using interactive queue), e.g., 
#     
# 
# ```bash
#  mpirun -np <ntasks> $qb -server qb_cdft.in qb_cdft.out
# ```
# 
# where ntasks designates the number of tasks/processors and qb_cdft.\* are files reserved in client-server mode.
# 
# Make sure this Jupyter notebook sits in the same directory as the groundstate calculation.
# 
# Using 20 processors on Intel Ivybridge nodes, this example takes less than 10 min.

# ### Tutorial: coupling constant for He$_2^+$

# Here we will do a proper run of PyCDFT and include the calculation of the electron coupling.
# Instead of randomly guessing an initial constraint potential $V$, we shall use the `brentq` or `brenth` optimizers to perform a search. For these particular optimizers, we need to specify a range of $V$ where the function space is sampled for both positive and negative values; a range of (-2,2) is typically sufficient.
# 
# As before, we initialize the system in the PyCDFT module.

from pycdft import *
from ase.io import read

V = (2,-2)
# V = -0.703772086888  # this is close to optimal constraint potential, use secant optimizer to refine
    
print("==================== Initializing Run =========================")
# Read atomic structure
cell = read("./He2_3Ang.cif")
print(r"Initial atomic positions (Ang):")
print(cell.get_positions())
print(cell.get_cell())
cell.positions[1][2] = 3.0

# Construct sample class, set FFT grid
sample = Sample(ase_cell=cell, n1=112, n2=112, n3=112, vspin=1)
print(sample.atoms[1])
    
# Set up the DFT Driver, provide necessary commands to initialize
# the external DFT code (Qbox in this case)
qboxdriver = QboxDriver(
    sample=sample,
    init_cmd="load gs.xml \n" 
        "set xc PBE \n" 
        "set wf_dyn PSDA \n" 
        "set_atoms_dyn CG \n" 
        "set scf_tol 1.0E-8 \n",
        scf_cmd="run 0 50 5",
)
    
# Set up two CDFT solvers for two diabatic states
solver1 = CDFTSolver(job="scf", # Indicate the calculation is SCF calculation
                    optimizer="brenth", # Specifiy the optimizer used for the Lagrangian multiplier
                    sample=sample, 
                    dft_driver=qboxdriver)
solver2 = solver1.copy()
    
# Initialize two constraints that localize the extra +1 charge on each site
# Here we use ChargeTransferConstraint, which constrains the relative electron number
# between two Fragments that represent the donor and acceptor
ChargeTransferConstraint(
    sample=solver1.sample,
    donor=Fragment(solver1.sample, solver1.sample.atoms[0:1]), # Donor Fragment
    acceptor=Fragment(solver1.sample, solver1.sample.atoms[1:2]), # Acceptor Fragment
    V_brak =V, # Search region for the brenth optimizer
    N0=1, # Desired charge to be localized
    N_tol=1E-6 # tolerance to target charge to be localized
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


# Visualize the charge density.

#print ("~~~~~~~~~~~ Debugging ~~~~~~~~~")
#origin=tuple(np.multiply([0,0,0],0.529))
#get_rho(solver1,origin,1)
#get_rho(solver2,origin,2)

