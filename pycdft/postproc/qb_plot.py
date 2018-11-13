import numpy as np
from ase.io.cube import read_cube_data, write_cube
import matplotlib.pyplot as plt
#import ase.visualize.mlab

# Compatible for PyCDFT v0.5
# plotting functions for Qbox interface

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
      
def get_rho(CDFTSolver):
    """ Borrowed from implementation of abstract fetch_rhor method for Qbox 
       Generate from Qbox the rhor.cube file """
    vspin = self.sample.vspin
    n1, n2, n3 = self.sample.n1, self.sample.n2, self.sample.n3
    self.sample.rho_r = np.zeros([vspin, n1, n2, n3])

    for ispin in range(vspin):
        # Qbox generates charge density
        self.run_cmd(cmd="plot -density {} {}".format(
            "-spin {}".format(ispin + 1) if vspin == 2 else "",
            self.rhor_file
        ))

        # quick check
        rhor_raw = read_cube_data(self.rhor_file)[0]
        assert rhor_raw.shape == (n1, n2, n3)

def get_hirsh_ct(CDFTSolver,origin):
    """ Extract Hirschfeld weights for plotting  
        For ChargeTransfer constraint only 
 
        origin: 3-tuple; same as rhor.cube file; origin of volumetric data
                in Angstroms"""
    constraints = self.sample.constraints
 
    index=1
    for c in constraints:
       # write to cube file for visualizing
       # donor and acceptor atoms
       atoms = c.atoms
       weights_dat = parse(c.w,-1)
       
       filname="hirshr"+str(index)+".cube"
       fileobj=open(filname,"w")
 
       write_cube(fileobj,atoms,weights_dat,origin=origin)
       fileobj.close()
   
       index+=1
    
    return weights


#--------------------------------------------------------
## testing of parse
#rhor=np.arange(60).reshape(3,4,5)
#rev_rhor=parse(rhor,1)
#print(rhor)
#print(rev_rhor)
#print(rhor==parse(rev_rhor,-1))

