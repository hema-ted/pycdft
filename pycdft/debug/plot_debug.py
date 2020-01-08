import numpy as np
from ase.io.cube import read_cube_data, write_cube
from ase import Atom

# get fft
try:
    from pyfftw.interfaces.numpy_fft import fftn, ifftn, rfftn, irfftn
except ImportError:
    from numpy.fft import fftn, ifftn, rfftn, irfftn
from numpy.fft import fftshift, ifftshift


# Compatible for PyCDFT v0.1
#   Helper functions for debugging the forces in PyCDFT
#   The goal is to print the following:
#     1) w_k(r) and grad_w_k(r) for each constraint k; compare with finite diff
#     2) rho_i(r) and grad_rho_i(r) for each atom i; compare with finite diff

def parse(dat,mode):
    """ parsing of e.g., charge density or weights for writing/reading to CUBE format
        
        Attributes: 
           dat (array): volumetric data, n1 x n2 x n3
           mode (int):  1 = from CUBE -> PyCDFT; -1 = from PyCDFT -> write CUBE
    """
    #assert(np.abs(mode)==1)
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

#===================== Hirshfeld weights per constraint ====================

def get_hirsh(CDFTSolver,origin):
    """ Extract Hirschfeld weights for Charge constraint 
 
        Attributes:
           CDFTSolver (CDFTSolver)
           origin (tuple): (x,y,z) coord; same as rhor.cube file; origin of volumetric data;
                give in Angstroms; gets converted to Bohr in program 
    """
    constraints = CDFTSolver.sample.constraints

    index=1
    for c in constraints:
        atoms_iter = CDFTSolver.sample.atoms    # calculating requires pycdft-modified Atom type
        atoms_write = CDFTSolver.sample.ase_cell # writing requires ASE Atom type

        # write to cube file for visualizing
        weights_dat = parse(c.w[0],-1)

        filname="hirshr"+str(index)+".cube"
        fileobj=open(filname,"w")

        write_cube(fileobj,atoms_write,weights_dat,origin=origin)
        fileobj.close()
        print("Generated cube file for Constraint %d"% (index))
        
    print("Completed Hirshfeld weight extraction!")

def get_hirsh_ct(CDFTSolver,origin):
    """ Extract Hirschfeld weights for Charge_Transfer constraint 
 
        Attributes:
           CDFTSolver (CDFTSolver)
           origin (tuple): (x,y,z) coord; same as rhor.cube file; origin of volumetric data;
                give in Angstroms; gets converted to Bohr in program 
    """
    constraints = CDFTSolver.sample.constraints

    index=1
    for c in constraints:
        atoms = CDFTSolver.sample.ase_cell
        # write to cube file for visualizing
        weights_dat = parse(c.w[0],-1)

        filname="hirshr"+str(index)+".cube"
        fileobj=open(filname,"w")

        write_cube(fileobj,atoms,weights_dat,origin=origin)
        fileobj.close()
        print("Generated for Constraint %d"% (index))
        
    print("Completed! Have fun plotting")

      
#===================== Charge density- total and per atom  =========================

def get_rho_atom(CDFTSolver,origin):
    """ Generate charge density for each atom 

        Attributes:
           CDFTSolver (CDFTSolver)
           origin (tuple): (x,y,z) coord; same as rhor.cube file; origin of volumetric data
                give in Angstroms; gets converted to Bohr in program 
    """
    atoms_iter = CDFTSolver.sample.atoms    # calculating requires pycdft-modified Atom type
    atoms_write = CDFTSolver.sample.ase_cell # writing requires ASE Atom type
    n = CDFTSolver.sample.n
    omega = CDFTSolver.sample.omega

    index = 1
    for atom in atoms_iter:
        rhoatom_g = CDFTSolver.sample.compute_rhoatom_g(atom)
        rhoatom_r = (n / omega) * np.fft.ifftn(rhoatom_g).real # FT G -> R
        #rhoatom_r = (n / omega) * ifftn(rhoatom_g).real # FT G -> R

        # write to cube file 
        rhoatom_r = parse(rhoatom_r,-1)

        filname="rhoatom_r_"+str(index)+".cube"
        fileobj=open(filname,"w")

        write_cube(fileobj,atoms_write,rhoatom_r,origin=origin)
        fileobj.close()
        print("Generated rhoatom_r cube file for Atom %d"% (index))
        index += 1
    print("Completed get_rho_atom")

def get_rho(CDFTSolver):
    """ Borrowed from implementation of abstract ``fetch_rhor`` method for Qbox 
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
        #assert rhor_raw.shape == (n1, n2, n3)
        
    print("Generated charge density cube file!")

#============== Charge and Hirshfeld gradients =============================
def get_grad(CDFTSolver,origin):
    r""" Extract gradient of Hirschfeld weights and charge density; both are calculated in calculation
         of :math:`\nabla w(\bf{r})`
 
         Attributes:
            CDFTSolver (CDFTSolver)
            origin (tuple): (x,y,z) coord; same as rhor.cube file; origin of volumetric data;
                give in Angstroms; gets converted to Bohr in program 
    """
    constraints = CDFTSolver.sample.constraints

    ic = 1
    for c in constraints:
       
        atoms_iter = CDFTSolver.sample.atoms    # calculating requires pycdft-modified Atom type
        atoms_write = CDFTSolver.sample.ase_cell # writing requires ASE Atom type

        ia = 1
        for atom in atoms_iter: 
   
            #w_grad, rho_grad_r,w_grad_part,rhopro_r = c.debug_w_grad_r(atom)
            w_grad, rho_grad_r = c.debug_w_grad_r(atom)
  
            # write to cube file for visualizing
            # separate file for each cartesian direction
            for icart in range(3):               
                w_grad_tmp = parse(w_grad[icart][0],-1)
                rho_grad_r_tmp = parse(rho_grad_r[icart],-1)
   
                fil1="w_grad_atom"+str(ia)+"_c"+str(ic)+"_i"+str(icart)+".cube"
                fil2="rhoatom_grad_atom"+str(ia)+"_c"+str(ic)+"_i"+str(icart)+".cube"

                fileobj=open(fil1,"w")
                write_cube(fileobj,atoms_write,w_grad_tmp,origin=origin)
                fileobj.close()
                print("Generated Hirshfeld wts cube file for Atom %d,  constraint %d, dir %d"% (ia, ic, icart))

                fileobj=open(fil2,"w")
                write_cube(fileobj,atoms_write,rho_grad_r_tmp,origin=origin)
                fileobj.close()

            ia += 1
        ic += 1
        print("Generated charge density cube file for Atom %d, constraint %d, dir %d"% (ia,ic, icart))

    print("Completed extraction of grad quantities!")

#--------------------------------------------------------
