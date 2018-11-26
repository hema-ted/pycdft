import numpy as np
from ase.io.cube import read_cube_data, write_cube

# Compatible for PyCDFT v0.5
#   Helper functions for debugging the forces in PyCDFT
#   The goal is to print the following:
#     1) w_k(r) and grad_w_k(r) for each constraint k; compare with finite diff
#     2) rho_i(r) and grad_rho_i(r) for each atom i; compare with finite diff

def parse(dat,mode):
    """ parsing of e.g., charge density or weights for writing/reading to CUBE format
 
        dat (n1,n2,n3 array)
        mode = 1 from read CUBE -> PyCDFT
             = -1 from PyCDFT -> write CUBE
    """
    assert(np.abs(mode)==1)
    n1,n2,n3 = np.shape(dat)
    if mode==1:
        dat1 = np.roll(dat, n1//2, axis=0)
        dat2 = np.roll(dat1, n2//2, axis=1)
        dat3 = np.roll(dat2, n3//2, axis=2)
    elif mode==-1:
        dat1 = np.roll(dat,int(np.ceil(n3/2)),axis=2)
        dat2 = np.roll(dat1,int(np.ceil(n2/2)),axis=1)
        dat3 = np.roll(dat2,int(np.ceil(n1/2)),axis=0)
    parse_dat = dat3
    return parse_dat

#===================== Hirshfeld and its gradient ====================

def get_hirsh(CDFTSolver,origin):
    """ Extract Hirschfeld weights 
 
        origin: 3-tuple; same as rhor.cube file; origin of volumetric data
                give in Angstroms; gets converted to Bohr in program 

        TOCHECK: proper passing of atoms, so can also get atomic number printed"""

    constraints = CDFTSolver.sample.constraints

    index=1
    for c in constraints:
        ase_cell = CDFTSolver.sample.ase_cell
        atoms = list(Atom(sample=CDFTSolver.sample,ase_atom=atom) for atom in ase_cell)

        # write to cube file for visualizing
        weights_dat = parse(c.w[0],-1)

        filname="hirshr"+str(index)+".cube"
        fileobj=open(filname,"w")

        write_cube(fileobj,atoms,weights_dat,origin=origin)
        fileobj.close()
        print("Generated cube file for Constraint %d"% (index))
        
    print("Completed Hirshfeld weight extraction!")

      
#===================== Charge density and its gradient =========================

def get_rho_atom(CDFTSolver):
    """ Generate charge density for each atom """
    atoms = list(Atom(sample=CDFTSolver.sample,ase_atom=atom) for atom in ase_cell)
    n = CDFTSolver.n1 * CDFTSolver.n2 * CDFTSolver.n3
    omega = CDFTSolver.omega

    index = 1
    for atom in atoms:
        rhoatom_g = CDFTSolver.sample.compute_rhoatom_g(atom)
        rhoatom_r = (n / omega) * np.fft.ifftn(rhoatom_g).real # FT G -> R

        # write to cube file 
        rhoatom_r = parse(rhoatom_r,-1)

        filname="rhoatom_r_"+str(index)+".cube"
        fileobj=open(filname,"w")

        write_cube(fileobj,atom,rhoatom_r,origin=origin)
        fileobj.close()
        print("Generated rhoatom_r cube file for Atom %d"% (index))
        index += 1

def get_rho(CDFTSolver):
    """ Borrowed from implementation of abstract fetch_rhor method for Qbox 
       Generate from Qbox the rhor.cube file """
    vspin = CDFTSolver.sample.vspin
    n1, n2, n3 = CDFTSolver.sample.n1, CDFTSolver.sample.n2, CDFTSolver.sample.n3
    rho_r = np.zeros([vspin, n1, n2, n3])

    for ispin in range(vspin):
        # Qbox generates charge density
        CDFTSolver.dft_driver.run_cmd(cmd="plot -density {} {}".format(
            "-spin {}".format(ispin + 1) if vspin == 2 else "",
            CDFTSolver.dft_driver.rhor_file
        ))

        # quick check
        rhor_raw = read_cube_data(CDFTSolver.dft_driver.rhor_file)[0]
        assert rhor_raw.shape == (n1, n2, n3)
        
    print("Generated charge density cube file!")

#============== Charge and Hirshfeld gradients =============================
def get_grad(CDFTSolver,origin):
    """ Extract gradient of Hirschfeld weights and charge density; both are calculated in calculation
          of grad_w_r
 
        origin: 3-tuple; same as rhor.cube file; origin of volumetric data
                give in Angstroms; gets converted to Bohr in program 

        TOCHECK: proper passing of atoms, so can also get atomic number printed"""

    constraints = CDFTSolver.sample.constraints

    ic = 1
    for c in constraints:
       
        ase_cell = CDFTSolver.sample.ase_cell
        atoms = list(Atom(sample=CDFTSolver.sample,ase_atom=atom) for atom in ase_cell)

        ia = 1
        for atom in atoms: 
   
           w_grad, rho_grad_r = c.debug_w_grad_r(atom)
  
           # write to cube file for visualizing
           w_grad = parse(w_grad[0],-1)
           rho_grad_r = parse(rho_grad_r,-1)
   
           fil1="w_grad_"+str(index)+".cube"
           fil2="rhoatom_grad_"+str(index)+".cube"

           fileobj=open(fil1,"w")
           write_cube(fileobj,atom,w_grad,origin=origin)
           fileobj.close()
           print("Generated Hirshfeld wts cube file for Atom %d,  constraint %d"% (ia, ic))

           fileobj=open(fil2,"w")
           write_cube(fileobj,atom,rho_grad_r,origin=origin)
           fileobj.close()
           print("Generated charge density cube file for Atom %d, constraint %d"% (ia,ic))
           ia += 1
        ic += 1
           
    print("Completed extraction of grad quantities!")
#--------------------------------------------------------
## testing of parse
#rhor=np.arange(60).reshape(3,4,5)
#rev_rhor=parse(rhor,1)
#print(rhor)
#print(rev_rhor)
#print(rhor==parse(rev_rhor,-1))

