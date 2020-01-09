import numpy as np
from scipy.linalg import fractional_matrix_power
from pycdft.cdft import CDFTSolver
from pycdft.common.ft import FFTGrid, ftrr
from pycdft.common.units import hartree_to_ev, hartree_to_millihartree
import time 

def compute_elcoupling(solver1: CDFTSolver, solver2: CDFTSolver,debug=True):
    """ Compute electronic coupling in mH between two KS wavefunctions.
        For theory on implementation, see:
        1) Oberhofer2010; dx.doi.org/10.1063/1.3507878
        2) Kaduk2012; dx.doi.org/10.1021/cr200148b
        3) Goldey2017; dx.doi.org/10.1021/acs.jctc.7b00088
  
        only @ Gamma point, so quantities are real; but keep conjugate operations for now
        
        Attributes:
           solver1, solver2 (CDFTSolver): instances of solver
  
        Internal Parameters:
           O (array): overlap matrix of KS orbitals, norb x norb
           S (array): overlap matrix of diabatic states, 2x2 
           W (array): weight function matrix, 2x2
           H (array): diabatic Hamiltonian matrix, 2x2
           Hsymm (array): symmetrized diabatic Hamiltonian matrix, 2x2
    """
    try:
       start_time = time.time()
       assert solver1.sample.vspin == solver2.sample.vspin
       vspin = solver1.sample.vspin
      
       if vspin != 1:
           raise NotImplementedError
   
       wfc1 = solver1.sample.wfc
       wfc2 = solver2.sample.wfc
   
       assert wfc1.nspin == wfc2.nspin
       assert wfc1.nkpt == wfc2.nkpt
       assert np.all(wfc1.nbnd == wfc2.nbnd)
       nspin, nkpt, nbnd, norb = wfc1.nspin, wfc1.nkpt, wfc1.nbnd, wfc1.norb
   
       # Gamma point only
       if nspin not in [1, 2] or nkpt != 1:
           raise NotImplementedError
    except:
       print("Error in elcoupling! Check your solvers!")
       solver1.dft_driver.exit()
       sys.exit()

    sample = solver1.sample
    # density grid
    n1, n2, n3 = sample.n1, sample.n2, sample.n3
    n = n1 * n2 * n3
    # wavefunction grid
    wgrid = wfc1.wgrid
    m1, m2, m3 = wgrid.n1, wgrid.n2, wgrid.n3
    m = m1 * m2 * m3
    omega = sample.omega

    #----------------------------------------------
    print("")
    if debug:
      print(""); print(" Below is a breakdown of components that go into calculating H_ab" )
    print(" npin: %d, nkpt: %d \n nbnd (per spin channel): %s, norb: %d"%(nspin,nkpt,nbnd,norb))
  
    # To calculate the coupling, we need S, W, and H
    print("")
    # S matrix
    O = cdft_get_O(wfc1,wfc2,omega,m)
    S,Odet = cdft_get_S(O)
    print("DONE: S") 
    timer(start_time,time.time())

    # W matrix 
    # TODO: averaging for Hermitian H in *get_H; remove here ?
    # constraint potential matrix element Vab = <psi_a| (V_a + V_b)/2 |psi_b>
    # where V_i = \sum V w_j, i.e., the constraint lagrange multiplier is included 
    Vc_dense = 0.5 * (solver1.Vc_tot + solver2.Vc_tot)[0, ...]
    Vc = ftrr(Vc_dense, source=FFTGrid(n1, n2, n3), dest=FFTGrid(m1, m2, m3)).real
    W,C = cdft_get_W(wfc1,wfc2,Vc,O,omega,m)
    print("DONE: W")
    timer(start_time,time.time())
    
    # H matrix 
    H = cdft_get_H(solver1,solver2,S,W)
    print("DONE: H")
    timer(start_time,time.time())

    # H matrix between orthogonal diabatic states 
    Hsymm = cdft_get_Hsymm(H,S)
 
    # debug output
    print("")
    if debug:
       print("O matrix"); print(O); print("|O|:",Odet); print("")
       print("S matrix"); print(S); print("")
       print("W matrix"); print(W); print("")
       print("--> Cofactor:"); print(C); print("")
       print("H matrix"); print(H); print("")
       print("H ortho. Lowdin"); print(Hsymm); print("")

    # final output
    print("~~~~~~~~~~~~~~~~~ Electronic Coupling ~~~~~~~~~~~~~~~~~~")
    print(" See Oberhofer & Blumberger 2010: dx.doi.org/10.1063/1.3507878 ")
    print("")
    print("|Hab| (H):", np.abs(Hsymm[0, 1]))
    print("|Hab| (mH):", np.abs(Hsymm[0, 1] * hartree_to_millihartree))
    print("|Hab| (eV):", np.abs(Hsymm[0, 1] * hartree_to_ev))
    print(""); print("Elapsed time for Electronic Coupling:")
    timer(start_time,time.time())

    solver1.dft_driver.exit()

def cdft_get_O(wfc1,wfc2,omega,m):
    r""" construct orbital overlap matrix 
     
      :math:`{\bf O}_{ij} = \langle \phi^j_b | \phi^i_a \rangle`
     
      where :math:`\phi^i_a` corresponds to an orthogonal spin orbital *i* in the KS determinant of state *a*

      see Eq. 20 in Oberhofer2010, which assumes plane wave basis

    """

    nspin, nkpt, nbnd, norb = wfc1.nspin, wfc1.nkpt, wfc1.nbnd, wfc1.norb
    # Otot, containing spin up and down
    O = np.zeros([norb, norb])
    for ispin in range(nspin):
        for ibnd, jbnd in np.ndindex(nbnd[ispin, 0], nbnd[ispin, 0]):
            i = wfc1.skb2idx(ispin, 0, ibnd)
            j = wfc2.skb2idx(ispin, 0, jbnd) # before was wfc1 
            O[i, j] = (omega / m) * np.sum(np.conjugate(wfc1.psi_r[i]) * wfc2.psi_r[j])
 
    return O

def cdft_get_S(O):
    r""" build overlap matrix S
 
      :math:`{\bf S}_{ab} = \langle \psi_a | \psi_b \rangle \det{{\bf O}}`
  
      see Eq 20 in Oberhofer2010
    """
    S = np.zeros([2,2]) # 2x2 state overlap matrix S
    Odet = np.linalg.det(O)
   
    S[0,0] = S[1,1] = 1.0
    S[1,0] = Odet # S_BA
    S[0,1] = np.conjugate(Odet) # Eq. 12, Oberhofer2010 # S_AB
 
    return S, Odet

def cdft_get_W(wfc1,wfc2,Vc,O,omega,m): 
    r""" build W matrix 

        :math:`W_{ba} = \langle \psi_b | \sum_i w({\bf r}_i)| \psi_a \rangle = \sum_i \sum_j \langle \phi_A^i | w({\bf r}) | \phi_B^j \rangle {\bf C}_{ij}`

        see Eqs. 21, 24, 25 in Oberhofer2010
    """

    nspin, nkpt, nbnd, norb = wfc1.nspin, wfc1.nkpt, wfc1.nbnd, wfc1.norb
    P12 = np.zeros([norb, norb]); #P21 = np.zeros([norb, norb]); 
    #P11 = np.zeros([norb, norb]); P22 = np.zeros([norb, norb]); 
    W = np.zeros([2,2])

    # see Eq. 25 in Oberhofer2010
    # cofactor matrix C
    Odet = np.linalg.det(O)
    Oinv = np.linalg.inv(O)
    CT = Oinv @ (Odet*np.eye(norb))
    C= CT.T

    # constraint potential matrix P, for finding V_a*W_ab, see Eq. 22 in Oberhofer2010
    # in our notation: Vab = V_a*W_ab
    for ispin in range(nspin):
       for ibnd, jbnd in np.ndindex(nbnd[ispin, 0], nbnd[ispin, 0]):
           i = wfc1.skb2idx(ispin, 0, ibnd)
           j = wfc2.skb2idx(ispin, 0, jbnd)
   
           p = (omega / m) * np.sum(np.conjugate(wfc1.psi_r[i]) * Vc * wfc2.psi_r[j])
           P12[i, j] = p # orbital overlaps, <\phi_A | w | \phi_B>
   
           #p = (omega / m) * np.sum(np.conjugate(wfc2.psi_r[i]) * Vc * wfc1.psi_r[j])
           #P21[i, j] = p # orbital overlaps, <\phi_B | w | \phi_A>
   
           ## not need on-diagonal W, omit for speed up?
           #p = (omega / m) * np.sum(np.conjugate(wfc1.psi_r[i]) * Vc * wfc1.psi_r[j])
           #P11[i, j] = p # orbital overlaps, <\phi_A | w | \phi_A>
   
           #p = (omega / m) * np.sum(np.conjugate(wfc2.psi_r[i]) * Vc * wfc2.psi_r[j])
           #P22[i, j] = p # orbital overlaps, <\phi_B | w | \phi_B>
         
    Vab = np.trace(P12 @ C) # \sum_ij = Tr(A_ij * B_ij) 
    Vba = np.conjugate(Vab) # np.trace(P21 @ C)

    #Vaa = np.trace(P11 @ C)
    #Vbb = np.trace(P22 @ C) 
   
    ## on-diagonal elements not used
    #W[0,0] = Vaa
    #W[1,1] = Vbb
    W[0,1] = Vab
    W[1,0] = Vba

    return W, C

def cdft_get_H(solver1,solver2,S,W):
    r""" 
     H matrix between nonorthogonal diabatic states 
     
     (Eq. 9-11 in Oberhofer2010, Eq. 5 in Goldey2017);
     see also p 344 of Kaduk2012

       :math:`H_{aa} = \langle \psi_a|H_{KS}|\psi_a \rangle`

       :math:`H_{ab} = F_b S_{ab} - V_b  W_{ab}`

       :math:`H_{ba} = F_a  S_{ba} - V_a  W_{ba}`

       :math:`F_a = \langle \psi_a|H_{KS} + V \cdot w | \psi_a \rangle`

    """ 
    H = np.zeros([2, 2])
    H[0, 0] = solver1.sample.Ed
    H[1, 1] = solver2.sample.Ed
    Fa = solver1.sample.Ed + solver1.sample.Ec
    Fb = solver2.sample.Ed + solver2.sample.Ec

    # to make H hermitian
    # H_ab -> 1/2(H_ab + H_ba)
    H[0, 1] = 0.5 * (Fb * S[0, 1] + Fa * S[1,0]) - 0.5 * (W[0,1]+W[1,0])
    H[1,0] = np.conjugate(H[0,1])
   
    return H

def cdft_get_Hsymm(H,S):
    """ 
       get orthogonal diabatic H matrix using Lowdin diagonalization
    """ 
    Ssqrtinv = fractional_matrix_power(S, -1 / 2)
    Hsymm = Ssqrtinv @ H @ Ssqrtinv
    print("H matrix between orthogonal diabatic states using Lowdin diagonalization:")
    print(Hsymm)
  
    return Hsymm

def timer(start,end):
        hours, rem = divmod(end-start,3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),seconds))
