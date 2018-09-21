import numpy as np
from pycdft.cdft import CDFTSolver
from pycdft.common.ft import FFTGrid, ftrr


def compute_elcoupling(solver1: CDFTSolver, solver2: CDFTSolver):
    """ Compute electronic coupling in mH between two KS wavefunctions."""
    assert solver1.sample.nspin == solver2.sample.nspin
    vspin = solver1.sample.nspin
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
    dgrid = FFTGrid(n1, n2, n3)
    # wavefunction grid
    wgrid = wfc1.wgrid
    m1, m2, m3 = wgrid.n1, wgrid.n2, wgrid.n3
    m = m1 * m2 * m3
    omega = sample.omega

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

    # 2x2 state overlap matrix S
    S = np.eye(2)
    S[0, 1] = S[1, 0] = Odet
    print("S matrix:")
    print(S)

    # constraint potential matrix element Vab = <psi_a| (V_a + V_b)/2 |psi_b>
    vc = ftrr(0.5 * (solver1.Vc_tot + solver2.Vc_tot)[0, ...], dgrid, wgrid)

    # cofactor matrix C
    C = Odet * Oinv.T
    print("C:", C)

    P = np.zeros([norb, norb])
    for ispin in range(nspin):
        for ibnd, jbnd in np.ndindex(nbnd[ispin, 0], nbnd[ispin, 0]):
            i = wfc1.skb2idx(ispin, 0, ibnd)
            j = wfc1.skb2idx(ispin, 0, jbnd)
            p = (omega / m) * np.sum(wfc1.psi_r[i] * vc * wfc2.psi_r[j])
            print(p)
            P[i, j] = p
    print("P:", P)
    Vab = np.trace(P @ C)
    print("Vab:", Vab)

    # H matrix (Eq. 9-11 in CP2K paper, Eq. 5 in QE paper)
    H = np.zeros([2, 2])
    H[0, 0] = solver1.sample.Ed
    H[1, 1] = solver2.sample.Ed
    Fa = solver1.sample.W
    Fb = solver2.sample.W
    H[0, 1] = H[1, 0] = 0.5 * (Fa + Fb) * S[0, 1] - Vab
    print("H matrix:")
    print(H)

    Hab = 1 / (1 - S[0, 1]**2) * (H[0, 1] - S[0, 1] * (H[0, 0] + H[1, 1]) / 2)
    # print(H[0, 1], S[0, 1] * (H[0, 0] + H[1, 1]) / 2, 1 / (1 - S[0, 1]**2))

    print("Hab (mH):", Hab * 1E3)
