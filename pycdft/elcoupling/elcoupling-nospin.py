import numpy as np
from scipy.linalg import fractional_matrix_power
from pycdft.cdft import CDFTSolver
from pycdft.common.ft import FFTGrid, ftrr
from pycdft.common.units import hartree_to_ev, hartree_to_millihartree


def compute_elcoupling_symm_nospin(solver1: CDFTSolver, solver2: CDFTSolver):
    """ Compute electronic coupling in mH between two KS wavefunctions.
        There is implicit symmetrization on the matrix elements.
        Works only for symmetric systems
   
        Notes on implementation, see:
        1) Oberhofer & Blumberger 2010; dx.doi.org/10.1063/1.3507878
        2) Kaduk, et al 2012; dx.doi.org/10.1021/cr200148b, esp. p 344, Eq. 51 
        3) Goldey, et al 2017; dx.doi.org/10.1021/acs.jctc.7b00088
  
    """

    print(" See Oberhofer & Blumberger 2010: dx.doi.org/10.1063/1.3507878 ")
    print(" Below is a breakdown of components that go into calculating H_ab" )
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


    ### build overlap matrix S
    # orbital overlap matrix O
    O = np.zeros([norb, norb])
    for ispin in range(nspin):
        for ibnd, jbnd in np.ndindex(nbnd[ispin, 0], nbnd[ispin, 0]):
            i = wfc1.skb2idx(ispin, 0, ibnd)
            j = wfc1.skb2idx(ispin, 0, jbnd)
            O[i, j] = (omega / m) * np.sum(wfc1.psi_r[i] * wfc2.psi_r[j])
    Odet = np.linalg.det(O)
    Oinv = np.linalg.inv(O)
    print("O matrix:")
    print(O)
    print("|O|:", Odet)

    # see Eq. 25 in Oberhofer2010
    # cofactor matrix C
    C = Odet * Oinv.T

    # 2x2 state overlap matrix S
    S = np.eye(2)
    S[0, 1] = S[1, 0] = Odet
    print("S matrix containing (plane wave) orbital overlaps:")
    print(S)

    # constraint potential matrix element Vab = <psi_a| (V_a + V_b)/2 |psi_b>
    # V_i = \sum V w_j
    Vc_dense = 0.5 * (solver1.Vc_tot + solver2.Vc_tot)[0, ...]
    Vc = ftrr(Vc_dense, source=FFTGrid(n1, n2, n3), dest=FFTGrid(m1, m2, m3)).real

    # constraint potential matrix P, for finding W_ab aka Vab, see Eq. 22 in Oberhofer2010
    P = np.zeros([norb, norb])
    for ispin in range(nspin):
        for ibnd, jbnd in np.ndindex(nbnd[ispin, 0], nbnd[ispin, 0]):
            i = wfc1.skb2idx(ispin, 0, ibnd)
            j = wfc1.skb2idx(ispin, 0, jbnd)
            p = (omega / m) * np.sum(wfc1.psi_r[i] * Vc * wfc2.psi_r[j])
            # print(p)
            P[i, j] = p # orbital overlaps, <\phi_B | w | \phi_a>
    print("P (<phi_m|w(r)|phi_n>):", P)
    print("C (cofactor matrix):", C)
    print("P @ C", P @ C)
    Vab = np.trace(P @ C) #<- need check if off diagonal elements important 

    # H matrix between nonorthogonal diabatic states (Eq. 9-11 in Oberhofer2010, Eq. 5 in Goldey2017)
    # see also p 344 of Kaduk2012
    H = np.zeros([2, 2])
    H[0, 0] = solver1.sample.Ed
    H[1, 1] = solver2.sample.Ed
    Fa = solver1.sample.Ed + solver1.sample.Ec
    Fb = solver2.sample.Ed + solver2.sample.Ec
    H[0, 1] = H[1, 0] = 0.5 * (Fa + Fb) * S[0, 1] - Vab
    print("H matrix between nonorthogonal diabatic states:")
    print(H)

    # H matrix between orthogonal diabatic states using Lowdin diagonalization
    Ssqrtinv = fractional_matrix_power(S, -1 / 2)
    Hsymm = Ssqrtinv @ H @ Ssqrtinv
    print("H matrix between orthogonal diabatic states using Lowdin diagonalization:")
    print(Hsymm)

    print("|Hab| (H):", np.abs(Hsymm[0, 1]))
    print("|Hab| (mH):", np.abs(Hsymm[0, 1] * hartree_to_millihartree))
    print("|Hab| (eV):", np.abs(Hsymm[0, 1] * hartree_to_ev))

def compute_elcoupling_nospin(solver1: CDFTSolver, solver2: CDFTSolver):
    """ Compute electronic coupling in mH between two KS wavefunctions.
        Non-symmetric systems
   
        Notes on implementation, see:
        1) Oberhofer & Blumberger 2010; dx.doi.org/10.1063/1.3507878
        2) Kaduk, et al 2012; dx.doi.org/10.1021/cr200148b, esp. p 344, Eq. 51 
        3) Goldey, et al 2017; dx.doi.org/10.1021/acs.jctc.7b00088
  
    """

    print(" See Oberhofer & Blumberger 2010: dx.doi.org/10.1063/1.3507878 ")
    print(" Below is a breakdown of components that go into calculating H_ab" )
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

    ### build overlap matrix S
    # orbital overlap matrix O
    O = np.zeros([norb, norb])
    for ispin in range(nspin):
        for ibnd, jbnd in np.ndindex(nbnd[ispin, 0], nbnd[ispin, 0]):
            i = wfc1.skb2idx(ispin, 0, ibnd)
            j = wfc2.skb2idx(ispin, 0, jbnd) # before was wfc1 
            O[i, j] = (omega / m) * np.sum(np.conjugate(wfc1.psi_r[i]) * wfc2.psi_r[j])
    Odet = np.linalg.det(O)
    Oinv = np.linalg.inv(O)
    print("O matrix:")
    print(O)
    print("|O|:", Odet)

    # see Eq. 25 in Oberhofer2010
    # cofactor matrix C
    #C = Odet * Oinv.T #<-- need proper matrix mult and transpose; 
    CT = Oinv @ (Odet*np.eye(norb))
    C = CT.T

    # 2x2 state overlap matrix S
    S = np.eye(2)
    S[1,0] = Odet # S_BA
    S[0,1] = np.conjugate(Odet) # Eq. 12, Oberhofer2010 # S_AB
    print("S matrix containing (plane wave) orbital overlaps:")
    print(S)

    # constraint potential matrix element Vab = <psi_a| (V_a + V_b)/2 |psi_b>
    # V_i = \sum V w_j
    Vc_dense = 0.5 * (solver1.Vc_tot + solver2.Vc_tot)[0, ...]
    Vc = ftrr(Vc_dense, source=FFTGrid(n1, n2, n3), dest=FFTGrid(m1, m2, m3)).real

    # constraint potential matrix P, for finding W_ab aka Vab, see Eq. 22 in Oberhofer2010
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
    Vba = np.conjugate(Vab) #<- temporary
    W = np.eye(2)
    #W[0,0] = solver1.sample.constraints.N
    #W[1,1] = solver2.sample.constraints.N
    #W[0,1] = W [1,0] = Wab
    #print("W (weight function matrix):", W)

    # H matrix between nonorthogonal diabatic states (Eq. 9-11 in Oberhofer2010, Eq. 5 in Goldey2017)
    # see also p 344 of Kaduk2012
    H = np.zeros([2, 2])
    H[0, 0] = solver1.sample.Ed
    H[1, 1] = solver2.sample.Ed
    Fa = solver1.sample.Ed + solver1.sample.Ec
    Fb = solver2.sample.Ed + solver2.sample.Ec
    H[0, 1] = 0.5 * (Fb * S[0, 1] + Fa * S[1,0]) - 0.5 * (Vab + Vba)
    H[1,0] = np.conjugate(H[0,1])
    print("H matrix between nonorthogonal diabatic states:")
    print(H)

    # H matrix between orthogonal diabatic states using Lowdin diagonalization
    Ssqrtinv = fractional_matrix_power(S, -1 / 2)
    Hsymm = Ssqrtinv @ H @ Ssqrtinv
    print("H matrix between orthogonal diabatic states using Lowdin diagonalization:")
    print(Hsymm)

    print("|Hab| (H):", np.abs(Hsymm[0, 1]))
    print("|Hab| (mH):", np.abs(Hsymm[0, 1] * hartree_to_millihartree))
    print("|Hab| (eV):", np.abs(Hsymm[0, 1] * hartree_to_ev))
