import numpy as np
from numpy.fft import *
from .atom import Atom
from .sample import Sample


class Fragment(object):
    """ A part of the system to which constraints may apply.

    Attributes:
        sample (Sample): sample.
        atoms (list of Atom): list of atoms belonging to the fragment.
        w (numpy array, shape = [n1, n2, n3]: Hirshfeld weight function.
    """

    def __init__(self, name: str, sample: Sample, atoms: list):
        self.name = name
        self.sample = sample
        self.atoms = atoms
        self.natoms = len(self.atoms)
        self.rhopro_r = None
        self.sample.fragments.append(self)

    def compute_w_grad(self, atom, weight):
        n1, n2, n3 = self.sample.n1, self.sample.n2, self.sample.n3

        r = atom.abs_coord

        # Compute the fourier transform term
        t2 = np.zeros([3, n1, n2, n3])
        G1, G2, G3 = atom.sample.G
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
            t2[i] = ifftn(ig[i] * eigr * self.sample.rhoatom_g[atom.symbol]).real

        # Compute the first term
        if atom in self.atoms:
            t1 = (1 - weight) / self.sample.rhopro_tot_r
        else:
            t1 = - weight / self.sample.rhopro_tot_r

        return t1 * t2
