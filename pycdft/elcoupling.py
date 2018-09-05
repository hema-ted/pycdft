import numpy as np
from pycdft.cdft import CDFTSolver
from pycdft.common.ft import FFTGrid, ftrr


def compute_elcoupling(solver1: CDFTSolver, solver2: CDFTSolver):
    """ Compute electronic coupling between two KS wavefunctions."""
    wfc1 = solver1.sample.wfc
    wfc2 = solver2.sample.wfc

    if np.any(np.array([wfc1.nspin, wfc1.nkpt, wfc2.nspin, wfc2.nkpt]) != 1):
        raise NotImplementedError

    assert wfc1.norb == wfc2.norb
    norb = wfc1.norb

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
    for i, j in np.ndindex(norb, norb):
        O[i, j] = (omega / m) * np.sum(wfc1[norb] * wfc2[norb])
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
    vc = ftrr(0.5 * (solver1.Vc_tot + solver2.Vc_tot), dgrid, wgrid)
    # cofactor matrix C
    C = Odet * Oinv.T
    P = np.zeros([norb, norb])
    for i, j in np.ndindex(norb, norb):
        P[i, j] = (omega / m) * np.sum(wfc1[norb] * vc * wfc2[norb])
    Vab = np.trace(P @ C)
    print("Vab:", Vab)

    # H matrix (Eq. 9-11 in CP2K paper, Eq. 5 in QE paper)
    H = np.zeros(2, 2)
    H[0, 0] = solver1.sample.Edft_bare
    H[1, 1] = solver2.sample.Edft_bare
    Fa = solver1.sample.Efree
    Fb = solver2.sample.Efree
    H[0, 1] = H[1, 0] = 0.5 * (Fa + Fb) * S[0, 1] - Vab
    print("H matrix:")
    print(H)

    Hab = 1 / (1 - S[0, 1]**2) * (H[0, 1] - S[0, 1] * (H[0, 0] + H[1, 1]) / 2)
    return Hab
