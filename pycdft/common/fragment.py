from __future__ import absolute_import, division, print_function

from .atom import Atom
from .sample import Sample


class Fragment(object):
    def __init__(self, sample, atoms):
        """ A part of the system on which the constraint acts.

        Args:
            sample (Sample): sample.
            atoms (list of Atom): list of atoms belonging to the fragment.
        """
        assert isinstance(sample, Sample)
        assert all(isinstance(atom, Atom) for atom in atoms)

        self.sample = sample
        self.atoms = atoms


    @property
    def natoms(self):
        return len(self.atoms)
