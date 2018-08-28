from copy import deepcopy
from random import randint
from subprocess import Popen
import numpy as np
from ase import Atoms
from ase.io import read
from .pp import SG15PP
from .units import angstrom_to_bohr, bohr_to_angstrom
from .atom import Atom


def _tmp_file():
    return "/tmp/cell{}".format(randint(1000, 9999))


class Cell(object):
    """ A cell that is defined by lattice constants and contains a list of atoms.

    A cell can be initialized by and exported to an ASE Atoms object.
    A cell can also be exported to several useful formats such as Qbox and Quantum Espresso formats.

    All internal quantities below are in atomic unit.

    Attributes:
        R (float 3x3 array): primitive lattice vectors (each row is one vector).
        G (float 3x3 array): reciprocal lattice vectors (each row is one vector).
        omega(float): cell volume.
        atoms(list of Atoms): list of atoms.
    """

    def __init__(self, ase_cell=None, R=None):
        """
        Args:
            ase_cell (ASE Atoms or str): input ASE cell, or name of a (ASE-supported) file.
            R (float array): real space lattice constants, used to construct an empty cell.
        """
        if ase_cell is not None:
            if isinstance(ase_cell, str):
                ase_cell = read(ase_cell)
            else:
                assert isinstance(ase_cell, Atoms)
            if np.all(ase_cell.get_cell() == np.zeros([3, 3])):
                self.R = R
            else:
                self.R = ase_cell.get_cell() * angstrom_to_bohr

            self._atoms = list(Atom(cell=self, ase_atom=atom) for atom in ase_cell)

        else:
            self.R = R
            self._atoms = list()

        self.distance_matrix = None

    def compute_distance_matrix(self):
        self.distance_matrix = None
        d = np.zeros([self.natoms, self.natoms])
        for i, ai in enumerate(self.atoms):
            for j, aj in enumerate(self.atoms):
                d[i, j] = np.linalg.norm(ai - aj)
        self.distance_matrix = d

    @property
    def isperiodic(self):
        return not bool(self.R is None)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        if R is None:
            self._R = self._G = self._omega = None
            return
        elif isinstance(R, int) or isinstance(R, float):
            self._R = np.eye(3) * R
        else:
            assert R.shape == (3, 3)
            self._R = R.copy()
        self._G = 2 * np.pi * np.linalg.inv(self._R).T
        assert np.all(np.isclose(np.dot(self._R, self._G.T), 2 * np.pi * np.eye(3)))
        self._omega = np.linalg.det(self._R)

    @property
    def G(self):
        return self._G

    @property
    def omega(self):
        return self._omega

    @property
    def atoms(self):
        return self._atoms

    @property
    def natoms(self):
        return len(self.atoms)

    @property
    def species(self):
        return sorted(set([atom.symbol for atom in self.atoms]))

    @property
    def nspecies(self):
        return len(self.species)

    @property
    def ase_cell(self):
        """ Get an ASE Atoms object of current cell."""
        ase_cell = Atoms(cell=self.R * bohr_to_angstrom if self.isperiodic else None)
        for atom in self.atoms:
            ase_cell.append(atom.ase_atom)
        return ase_cell

    def __repr__(self):
        return "Cell \"{}\" natoms={}\nLattice constants:\n{}\nReciprocal lattice constants:\n{}\n".format(
            self.ase_cell.get_chemical_formula(), self.natoms, self.R, self.G
        )

    def __str__(self):
        return self.__repr__()

    def add_atom(self, **kwargs):
        """ Add an atom."""
        self.atoms.append(Atom(cell=self, **kwargs))

    def show(self):
        """ Visualize the cell by VESTA."""
        fmt = "cif" if self.isperiodic else "xyz"
        fname = "{}.{}".format(_tmp_file(), fmt)
        self.save(fname)
        Popen(["vesta", fname])
        Popen(["sleep 120 && rm {}".format(fname)], shell=True)

    def save(self, fname):
        """ Save the cell to file."""
        self.ase_cell.write(fname)

    def export(self, fmt="qb", pseudos=None):
        """ Export the cell to various formats. Currently support Qbox and Quantum Espresso."""
        output = ""

        if fmt.lower() == "qb":
            output += "set cell {}\n".format(
                " ".join("{:06f}".format(self.R[i, j]) for i, j in np.ndindex(3, 3))
            )
            output += "\n"

            for species in self.species:
                output += "species {} {}\n".format(
                    species, pseudos[species] if pseudos else SG15PP[species]["xmlfile"]
                )
            output += "\n"

            for iatom, atom in enumerate(self.atoms):
                x, y, z = atom.abs_coord
                output += "atom {}{} {}  {:.8f}  {:.8f}  {:.8f}\n".format(
                    atom.symbol, iatom + 1, atom.symbol, x, y, z
                )

        else:
            raise NotImplementedError

        return output

    def copy(self):
        """ Get a copy of current cell. """
        return deepcopy(self)

    def nel(self, pseudos="SG15"):
        """ Compute # of electrons according to certain pseudopotential family."""
        if pseudos == "SG15":
            nel = 0
            for symbol in self.ase_cell.get_chemical_symbols():
                nel += SG15PP[symbol]["nel"]
        else:
            raise NotImplementedError
        return nel

    def get_maxforce(self):
        maxforce = 0
        for atom in self.atoms:
            f = np.linalg.norm(atom.Ftotal)
            if f > maxforce:
                maxforce = f
                maxforceatom = atom

        return maxforce, maxforceatom
