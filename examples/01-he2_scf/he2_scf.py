
# coding: utf-8

# # Tutorial 1: scf cycle in CDFT with He$_2^+$

# ### 0: Prerequisites

# After compiling the DFT driver and installing PyCDFT, run the ground state calculation.
# - - - - - - -
# For Qbox
# 
# Our starting DFT wavefunctions are based on the following input
# 
# ```text
# set cell 20.000000 0.000000 0.000000 0.000000 20.000000 0.000000 0.000000 0.000000 20.000000
# 
# species He He_ONCV_PBE-1.0.xml
# 
# atom He1 He  0.00000000  0.00000000  0.00000000
# atom He2 He  0.00000000  0.00000000  3.00000000
# 
# set net_charge 1
# set nspin 2
# set xc PBE
# set ecut 30
# set wf_dyn JD
# set scf_tol 1.0E-8
# 
# randomize_wf
# run 0 200 5
# save gs.xml
# ```
# We run the Qbox executable 
# 
# ```bash
#  export qb="/path/to/executable/qb"
#  $qb < gs.in > gs.out
# ```
# 
# Then in the same directory (and possibly through a different terminal window), run [Qbox in server mode](qboxcode.org/daoc/html/usage/client-server.html)
# 
# ```bash
#  mpirun -np <ntasks> $qb -server qb_cdft.in qb_cdft.out
# ```
# 
# where qb_cdft.\* are files reserved in client-server mode.
# 
# Make sure the directory is clean of any previous runs

# ### 1: A CDFT cycle

# Now we are in a position to run a self-consistent cycle in CDFT. First we import the module and the structure using a \*cif file

# In[1]:


from pycdft import *
from ase.io import read


# In[2]:


ase_cell = read("./He2.cif")
print("Absolute Coordinates (Ang)")
print(ase_cell.get_positions())


# Instantiate a sample with the imported structure (ASE atoms object) where
# 
#   n1, n2, n3: FFT grid size. This can be found using
#   
# ```bash
#   grep np2v gs.out
# ```
#   vspin: indicates how many constraint potentials; currently for spin-polarized calculations, the same Vc is used

# In[3]:


sample = Sample(ase_cell=ase_cell, n1=80, n2=80, n3=80, vspin=1)


# Next we instantiate an instance of the Constraint class, in this case the charge transfer constraint, which puts a constraint on the different in charge between specific donor and acceptor regions. 
# 
# Donor and acceptor regions are specified by the atoms index. It is advised to bunch donor atoms and acceptor atoms separately. 
# In this case, it is simple, the He atom on the left is the donor and the He atom on the right is the acceptor.
# 
# The KS energy functional with added constraint is then minimized using an optimizer (native to numpy).
# For the `secant` optimizer, an initial guess `V_init` is given.
# The `secant` optimizer is useful for when a good initial guess is given;
# otherwise the `brenth` or `brentq` optimizers are useful for searching the constraint potential within a range of values.
# 
# The constraint is evaluated until the error between the calculated count of electrons is less than the target number `N0` by a tolerance of `N_tol`.

# In[4]:


ChargeTransferConstraint(
    sample=sample,
    donor=Fragment(sample, sample.atoms[0:1]),
    acceptor=Fragment(sample, sample.atoms[1:2]),
    V_init=-1.05,
    N0=2,
    N_tol=1E-3
)


# Here the DFT driver is instantiated. We use Qbox in this instance and provide the basic input parameters and commands.

# In[5]:


qboxdriver = QboxDriver(
    sample=sample,
    init_cmd="load gs.xml\n"
              "set xc PBE\n"
              "set wf_dyn JD\n"
              "set charge_mix_coeff 0.3\n"
              "set scf_tol 1.0E-7\n",
    scf_cmd="run 0 50 5"
)


# Finally, we combine everything into an instance of CDFTSolver and call upon the main solve routine.

# In[ ]:


solver = CDFTSolver(job="scf", optimizer="secant", sample=sample, dft_driver=qboxdriver)
solver.solve()


# ### 2: Examining the output

# We get output like the following below.
# 
# Here we see the output of PyCDFT. For output from Qbox, see qb_cdft.out or the cluster output files.
# In each iteration, PyCDFT prints the DFT energy, the constraint energy and their total. 
# 
# Upon convergence, W = Ed and their difference is a measure for how away you are from convergence.
# 
# In this particular case, we chose a random initial guess and use the secant optimizer. 
# As can be seen constraint potential `V` explodes.
# For a more stable optimization run, do a search over possible `V` using `brenth` or `brentq`.
# 
# We shall do a better optimization and calculate the electronic coupling in the next tutorial, `02-he_coupling`
QboxDriver: setting output path to ./pycdft_outputs/solver1/...
QboxDriver: waiting for Qbox to start...
QboxDriver: initializing Qbox...
===================== Initializing Run ====================
Updating constraint with new structure...
Running optimizer:  secant
=======================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SCF iteration 1
  Ed (DFT energy) = -3.021073
  Ec (constraint energy) = -3.014962
  E (Ed + Ec) = -6.036034
  W (free energy) = -3.021111
  > Constraint #0 (type = charge transfer, N0 = 1, V = -1.050000000000):
    N = 2.871356
    dW/dV = N - N0 = 1.87135565
Elapsed time: 
00h:00m:06.21s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
00h:00m:12.97s
Running optimizer:  secant
=======================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SCF iteration 2
  Ed (DFT energy) = -3.021058
  Ec (constraint energy) = -3.015565
  E (Ed + Ec) = -6.036623
  W (free energy) = -3.021096
  > Constraint #0 (type = charge transfer, N0 = 1, V = -1.050205000000):
    N = 2.871370
    dW/dV = N - N0 = 1.87136982
Elapsed time: 
00h:00m:04.24s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
00h:00m:19.56s
Running optimizer:  secant
=======================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SCF iteration 3
  Ed (DFT energy) = -2.587015
  Ec (constraint energy) = -77.702878
  E (Ed + Ec) = -80.289893
  W (free energy) = -2.588009
  > Constraint #0 (type = charge transfer, N0 = 1, V = 26.014933432129):
    N = -2.986818
    dW/dV = N - N0 = -3.98681848
Elapsed time: 
00h:00m:08.24s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
00h:00m:30.14s
Running optimizer:  secant
=======================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SCF iteration 4
  Ed (DFT energy) = -2.780199
  Ec (constraint energy) = -22.579629
  E (Ed + Ec) = -25.359827
  W (free energy) = -2.780485
  > Constraint #0 (type = charge transfer, N0 = 1, V = 7.595622129883):
    N = -2.972678
    dW/dV = N - N0 = -3.97267844
Elapsed time: 
00h:00m:06.20s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
00h:00m:38.69s
Running optimizer:  secant
=======================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SCF iteration 5
  Ed (DFT energy) = -0.153917
  Ec (constraint energy) = -15475.729276
  E (Ed + Ec) = -15475.883193
  W (free energy) = -0.350200
  > Constraint #0 (type = charge transfer, N0 = 1, V = -5167.356539496155):
    N = 2.994865
    dW/dV = N - N0 = 1.99486456
Elapsed time: 
00h:00m:08.21s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
00h:00m:49.25s
Running optimizer:  secant
=======================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SCF iteration 6
  Ed (DFT energy) = 0.428655
  Ec (constraint energy) = -10301.265948
  E (Ed + Ec) = -10300.837293
  W (free energy) = 0.297605
  > Constraint #0 (type = charge transfer, N0 = 1, V = -3437.443796806851):
    N = 2.996743
    dW/dV = N - N0 = 1.99674279
Elapsed time: 
00h:00m:08.24s