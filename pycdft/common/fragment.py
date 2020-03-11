from pycdft.common.atom import Atom
from pycdft.common.sample import Sample


class Fragment(object):
    """ 
    A part of the system to which constraints may apply.

    Attributes:
        sample (Sample): sample.
        atoms (list of Atom): list of atoms belonging to the fragment.
        natoms (int): number of atoms in fragment
        rhopro_r (array): real space promolecule charge density
    """

    def __init__(self, sample: Sample, atoms: list, name: str = ""):
        self.name = name
        self.sample = sample
        self.atoms = atoms
        self.natoms = len(self.atoms)
        self.rhopro_r = None
        self.sample.fragments.append(self)
