import numpy as np
from numpy.fft import *
from pycdft.common.atom import Atom
from pycdft.common.sample import Sample


class Fragment(object):
    """ A part of the system to which constraints may apply.

    Attributes:
        sample (Sample): sample.
        atoms (list of Atom): list of atoms belonging to the fragment.
        w (numpy array, shape = [n1, n2, n3]: Hirshfeld weight function.
    """

    def __init__(self, sample: Sample, atoms: list, name: str=""):
        self.name = name
        self.sample = sample
        self.atoms = atoms
        self.natoms = len(self.atoms)
        self.rhopro_r = None
        self.sample.fragments.append(self)
