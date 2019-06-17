import numpy as np
from scipy.linalg import fractional_matrix_power
from pycdft.cdft import CDFTSolver
from pycdft.common.ft import FFTGrid, ftrr
from pycdft.common.units import hartree_to_ev, hartree_to_millihartree

def compute_elcoupling(solver1: CDFTSolver, solver2: CDFTSolver):
    """ Compute electronic coupling in mH between two KS wavefunctions.
   
        Compatible for symmetric and non-symmetric systems
        TODO: implementation of spin  
 
        Notes on implementation, see:
        1) Oberhofer & Blumberger 2010; dx.doi.org/10.1063/1.3507878
        2) Kaduk, et al 2012; dx.doi.org/10.1021/cr200148b, esp. p 344, Eq. 51 
        3) Goldey, et al 2017; dx.doi.org/10.1021/acs.jctc.7b00088
  
    """

    assert solver1.sample.vspin == solver2.sample.vspin
    vspin = solver1.sample.vspin
   
    # TODO: more than one constraint
    if vspin != 1:
        raise NotImplementedError

    wfc1 = solver1.sample.wfc
    wfc2 = solver2.sample.wfc

    assert wfc1.nspin == wfc2.nspin
    assert wfc1.nkpt == wfc2.nkpt
    assert np.all(wfc1.nbnd == wfc2.nbnd)

    # Gamma point only
    if nspin not in [1, 2] or nkpt != 1:
        raise NotImplementedError

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
  
    # To calculate the coupling, we need S, W, and H
    print(" See Oberhofer & Blumberger 2010: dx.doi.org/10.1063/1.3507878 ")
    print(" Below is a breakdown of components that go into calculating H_ab" )
    # S matrix
    O = cdft_get_O(wfc1,wfc2,omega,m)
    S = cdft_get_S(wfc1,wfc2)

    # W matrix 
    # constraint potential matrix element Vab = <psi_a| (V_a + V_b)/2 |psi_b>
    # where V_i = \sum V w_j, i.e., the constraint lagrange multiplier is included 
    Vc_dense = 0.5 * (solver1.Vc_tot + solver2.Vc_tot)[0, ...]
    Vc = ftrr(Vc_dense, source=FFTGrid(n1, n2, n3), dest=FFTGrid(m1, m2, m3)).real
    W = cdft_get_W(wfc1,wfc2,Vc,O,omega,m)
    
    # H matrix 
    H = cdft_get_H(solver1,solver2,S,W)

    # H matrix between orthogonal diabatic states 
    Hsymm = cdft_get_Hsymm(H,S)
 
    # final output
    print("|Hab| (H):", np.abs(Hsymm[0, 1]))
    print("|Hab| (mH):", np.abs(Hsymm[0, 1] * hartree_to_millihartree))
    print("|Hab| (eV):", np.abs(Hsymm[0, 1] * hartree_to_ev))

def cdft_get_O(wfc1,wfc2,omega,m):
    """ Overlap matrix 
      For plane waves, see Eq. 20 in Oberhofer & Blumberger 2010"""

    nspin, nkpt, nbnd, norb = wfc1.nspin, wfc1.nkpt, wfc1.nbnd, wfc1.norb
    O = np.zeros([norb, norb])
    for ispin in range(nspin):
        for ibnd, jbnd in np.ndindex(nbnd[ispin, 0], nbnd[ispin, 0]):
            i = wfc1.skb2idx(ispin, 0, ibnd)
            j = wfc2.skb2idx(ispin, 0, jbnd) # before was wfc1 
            O[i, j] = (omega / m) * np.sum(np.conjugate(wfc1.psi_r[i]) * wfc2.psi_r[j])
 
    #print("O matrix:")
    #print(O)
    return O

def cdft_get_S(O,nspin):
    """ build overlap matrix S
      O  is orbital overlap matrix 
 
      nspin = 1: S is 2x2 matrix
      nspin = 2: S is 2x2x2 matrix (S_ij, spin channel)"""

    Odet = np.linalg.det(O)
    print("|O|:", Odet)

    if nspin == 1:
       # 2x2 state overlap matrix S
       S = np.eye(2)
       S[1,0] = Odet # S_BA
       S[0,1] = np.conjugate(Odet) # Eq. 12, Oberhofer2010 # S_AB
    elif nspin == 2:
       pass
    print("S matrix containing (plane wave) orbital overlaps:")
    print(S)
   
    return S

def cdft_get_W(wfc1,wfc2,Vc,O,omega,m): 
    """ build W matrix """

    nspin, nkpt, nbnd, norb = wfc1.nspin, wfc1.nkpt, wfc1.nbnd, wfc1.norb

    # see Eq. 25 in Oberhofer2010
    # cofactor matrix C
    Odet = np.linalg.det(O)
    Oinv = np.linalg.inv(O)
    CT = Oinv @ (Odet*np.eye(norb))
    C = CT.T

    # constraint potential matrix P, for finding V_a*W_ab, see Eq. 22 in Oberhofer2010
    # in our notation: Vab = V_a*W_ab
    P = np.zeros([norb, norb])
    for ispin in range(nspin):
        for ibnd, jbnd in np.ndindex(nbnd[ispin, 0], nbnd[ispin, 0]):
            i = wfc1.skb2idx(ispin, 0, ibnd)
            j = wfc2.skb2idx(ispin, 0, jbnd)
            p = (omega / m) * np.sum(np.conjugate(wfc1.psi_r[i]) * Vc * wfc2.psi_r[j])
            # print(p)
            P[i, j] = p # orbital overlaps, <\phi_B | w | \phi_a>
    print("P (<phi_m|w(r)|phi_n>):", P)
    print("C (cofactor matrix):", C)
    print("P @ C", P @ C)
    Vab = np.trace(P @ C) #<- need check if off diagonal elements important 
    Vba = np.conjugate(Vab)

    W = np.zeros([2,2])
    # TODO: set W[0,0], W[1,1]
    W[0,1] = Vab
    W [1,0] = Vba

    return W

def cdft_get_H(solver1,solver2,S,W):
    """ 
     H matrix between nonorthogonal diabatic states 
     (Eq. 9-11 in Oberhofer2010, Eq. 5 in Goldey2017)
     see also p 344 of Kaduk2012
    
     H_aa = <\psi_a|H_KS|\psi_a>
     H_ab = F_b * S_ab - V_b * W_ab
     H_ba = F_a * S_ba - V_a * W_ba

     F_a = <\psi_a|H_KS + V*w | \psi_a>
    """ 

    H = np.zeros([2, 2])
    H[0, 0] = solver1.sample.Ed
    H[1, 1] = solver2.sample.Ed
    Fa = solver1.sample.Ed + solver1.sample.Ec
    Fb = solver2.sample.Ed + solver2.sample.Ec

    # to make H hermitian
    # H_ab -> 1/2(H_ab + H_ba)
    H[0, 1] = 0.5 * (Fb * S[0, 1] + Fa * S[1,0]) - 0.5 * (Vab + Vba)
    H[1,0] = np.conjugate(H[0,1])
    print("H matrix between nonorthogonal diabatic states:")
    print(H)
   
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
