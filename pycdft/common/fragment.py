import numpy as np
from numpy.fft import fftfreq, fftn, ifftn
from ase.io.cube import read_cube_data
from .atom import Atom
from .sample import Sample


class Fragment(object):
    """ A part of the system to which constraints may apply.

    Attributes:
        sample (Sample): sample.
        atoms (list of Atom): list of atoms belonging to the fragment.
        w (numpy array, shape = [n1, n2, n3]: Hirshfeld weight function.
    """

    _eps = 0.0001  # cutoff of Hirshfeld weight when the density approaches zero

    def __init__(self, sample: Sample, atoms: list, atomic_density_files: dict):
        self.sample = sample
        self.atoms = atoms
        self.natoms = len(self.atoms)
        #
        # fftgrid = self.sample.fftgrid
        # n1, n2, n3 = fftgrid.n1, fftgrid.n2, fftgrid.n3
        # self.w = np.zeros([n1, n2, n3])
        #
        # self.rhor_atomic_dict = {}
        # self.rhog_atomic_dict = {}
        # for species, f in atomic_density_files.items():
        #     rhor = read_cube_data(f)[0]
        #     assert rhor.shape == (n1, n2, n3)
        #     rhor1 = np.roll(rhor, n1 // 2, axis=0)
        #     rhor2 = np.roll(rhor1, n2 // 2, axis=1)
        #     rhor3 = np.roll(rhor2, n3 // 2, axis=2)
        #     self.rhor_atomic_dict[species] = rhor3
        #     self.rhog_atomic_dict[species] = fftn(rhor3)

        self.rhopro = None

    # def update(self):
    #     """Update weight function w with new atomic coordinates."""
    #     fftgrid = self.sample.fftgrid
    #     n1, n2, n3 = fftgrid.n1, fftgrid.n2, fftgrid.n3
    #     self.rho_pro = np.zeros([n1, n2, n3])
    #     self.rho_pro_tot = np.zeros([n1, n2, n3])
    #
    #     for atom in self.sample.cell.atoms:
    #         rhor = self.compute_rhor_atomic(atom)
    #         self.rho_pro_tot += rhor
    #         if atom in self.atoms:
    #             self.rho_pro += rhor
    #
    #     self.w[:, ...] = self.rho_pro / self.rho_pro_tot
    #     self.w[:, self.rho_pro < self._eps] = 0.0
    #
    # def compute_rhor_atomic(self, atom):
    #     """ Compute charge density for an atom with specific coordinate in cell. """
    #     assert isinstance(atom, Atom)
    #     r = atom.abs_coord
    #     species = atom.symbol
    #     rhog0 = self.rhog_atomic_dict[species]
    #
    #     G1, G2, G3 = atom.cell.G
    #     n1, n2, n3 = rhog0.shape
    #     freqlist1 = fftfreq(n1, d=1/n1)
    #     freqlist2 = fftfreq(n2, d=1/n2)
    #     freqlist3 = fftfreq(n3, d=1/n3)
    #
    #     eigr = np.zeros([n1, n2, n3], dtype=complex)
    #     # TODO: tidy up the following code
    #     """
    #     for h, k, l in np.ndindex(n1, n2, n3):
    #         G = freqlist1[h] * G1 + freqlist2[k] * G2 + freqlist3[l] * G3
    #         eigr[h, k, l] = np.exp(- 1j * G @ r)  # (r - r0)
    #     """
    #     G1, G2, G3 = G1 @ r, G2 @ r, G3 @ r
    #     look_up_table_h = np.exp(-1j * (freqlist1 * G1))
    #     look_up_table_k = np.exp(-1j * (freqlist2 * G2))
    #     look_up_table_l = np.exp(-1j * (freqlist3 * G3))
    #     eigr = np.kron(look_up_table_h, look_up_table_k)
    #     eigr = np.kron(eigr, look_up_table_l)
    #     eigr = np.reshape(eigr, newshape=(n1, n2, n3))
    #
    #     rhog = rhog0 * eigr  # translated charge density
    #     rhor = ifftn(rhog)
    #
    #     return rhor.real

    def compute_w_grad(self, atom):
        fftgrid = self.sample.fftgrid
        n1, n2, n3 = fftgrid.n1, fftgrid.n2, fftgrid.n3

        r = atom.abs_coord

        # Compute the fourier transform term
        t2 = np.zeros([3, n1, n2, n3])
        G1, G2, G3 = atom.cell.G
        freqlist1 = fftfreq(n1, d=1 / n1)
        freqlist2 = fftfreq(n2, d=1 / n2)
        freqlist3 = fftfreq(n3, d=1 / n3)


        # TODO : tidy up the code
        """
        ig = np.zeros([3, n1, n2, n3], dtype=complex)
        eigr = np.zeros([n1, n2, n3], dtype=complex)
        for h, k, l in np.ndindex(n1, n2, n3):
            G = freqlist1[h] * G1 + freqlist2[k] * G2 + freqlist3[l] * G3
            eigr[h, k, l] = np.exp(- 1j * G @ r)

            ig[0, h, k, l] = 1j * G[0]
            ig[1, h, k, l] = 1j * G[1]
            ig[2, h, k, l] = 1j * G[2]
        """
        ig0 = np.reshape(np.kron(G1, freqlist1), newshape=(len(G1), n1))
        ig1 = np.reshape(np.kron(G2, freqlist2), newshape=(len(G2), n2))
        ig2 = np.reshape(np.kron(G3, freqlist3), newshape=(len(G3), n3))
        ig = np.array([np.reshape(1j * (np.kron(np.kron(ig0[0], ig1[0]), ig2[0])), newshape=(n1, n2, n3)),
                       np.reshape(1j * (np.kron(np.kron(ig0[1], ig1[1]), ig2[1])), newshape=(n1, n2, n3)),
                       np.reshape(1j * (np.kron(np.kron(ig0[2], ig1[2]), ig2[2])), newshape=(n1, n2, n3))])

        look_up_table_h = np.exp(-1j * (freqlist1 * (G1 @ r)))
        look_up_table_k = np.exp(-1j * (freqlist2 * (G2 @ r)))
        look_up_table_l = np.exp(-1j * (freqlist3 * (G3 @ r)))

        eigr = np.kron(look_up_table_h, look_up_table_k)
        eigr = np.kron(eigr, look_up_table_l)
        eigr = np.reshape(eigr, newshape=(n1, n2, n3))

        for i in range(0, 3):
            t2[i] = ifftn(ig[i] * eigr * self.rhog_atomic_dict[atom.symbol]).real

        # Compute the first term
        if atom in self.atoms:
            t1 = (1 - self.w[0]) / self.rho_pro_tot
        else:
            t1 = (-self.w[0]) / self.rho_pro_tot

        return t1 * t2