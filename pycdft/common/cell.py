from __future__ import absolute_import, division, print_function

import numpy as np
from ase import Atoms
from ase.io import read
from six import string_types

from .units import angstrom_to_bohr, bohr_to_angstrom
from .atom import Atom
from .pp import SG15PP


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
            R (float array): direct lattice vectors, used to construct an empty cell.
        """
        if ase_cell is not None:
            if isinstance(ase_cell, string_types):
                ase_cell = read(ase_cell)
            else:
                assert isinstance(ase_cell, Atoms)
            self._R = ase_cell.get_cell() * angstrom_to_bohr
            self._compute_parameters()
            self._atoms = list(Atom(cell=self, ase_atom=atom) for atom in ase_cell)

        else:
            assert R.shape == (3, 3)
            self._R = R.copy()
            self._atoms = list()
            self._compute_parameters()

    def _compute_parameters(self):
        self._G = 2 * np.pi * np.linalg.inv(self._R).T
        assert np.all(np.isclose(np.dot(self._R, self._G.T), 2 * np.pi * np.eye(3)))
        self._omega = np.linalg.det(self._R)

    def get_forces(self):
        F = np.zeros([self.natoms, 3])
        for i, atom in enumerate(self.atoms):
            F[i] = atom.force
        return F

    def get_fc(self):
        F = np.zeros([self.natoms, 3])
        for i, atom in enumerate(self.atoms):
            F[i] = atom.fc
        return F

    def get_fdft(self):
        F = np.zeros([self.natoms, 3])
        for i, atom in enumerate(self.atoms):
            F[i] = atom.fdft
        return F

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        self._R = value
        self._compute_parameters()

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
    def nspecies(self):
        return len(set([atom.symbol for atom in self.atoms]))

    @property
    def ase_cell(self):
        """ Get an ASE Atoms object of current cell."""
        ase_cell = Atoms(cell=self.R * bohr_to_angstrom)
        for atom in self.atoms:
            ase_cell.append(atom.ase_atom)
        return ase_cell

    def __repr__(self):
        return "Cell \"{}\" natoms={}\nLattice constants:\n{}\nReciprocal lattice constants:\n{}\n".format(
            self.ase_cell.get_chemical_formula(), self.natoms, self.R, self.G
        )

    def __str__(self):
        return self.__repr__()

    def export(self, fmt="qb", pseudos=None):
        """ Export the cell to various formats. Currently support Qbox and Quantum Espresso."""
        ase_cell = self.ase_cell
        cell = ase_cell.get_cell()
        symbols = ase_cell.get_chemical_symbols()

        output = ""

        if fmt.lower() == "qe":
            output += "CELL_PARAMETERS angstrom\n"
            for i in range(3):
                output += "  ".join("{:06f}".format(cell[i, j]) for j in range(3)) + "\n"

            output += "ATOMIC_SPECIES\n"
            for symbol in set(symbols):
                output += "{}  {}  {}\n".format(
                    symbol, SG15PP[symbol]["mass"], pseudos[symbol] if pseudos else SG15PP[symbol]["upffile"]
                )

            positions = ase_cell.get_scaled_positions()
            output += "ATOMIC_POSITIONS crystal\n"
            for symbol, position in sorted(zip(symbols, positions), key=lambda x: x[0]):
                output += "  ".join([symbol] + ["{:06f}".format(position[j])
                                                for j in range(3)]) + "\n"

        if fmt.lower() == "qb":
            positions = ase_cell.get_positions()
            cell *= angstrom_to_bohr
            positions *= angstrom_to_bohr
            output += "set cell {}\n".format(
                " ".join("{:06f}".format(cell[i, j]) for i, j in np.ndindex(3, 3))
            )
            output += "\n"

            for symbol in set(symbols):
                output += "species {} {}\n".format(
                    symbol, pseudos[symbol] if pseudos else SG15PP[symbol]["xmlfile"]
                )
            output += "\n"

            #for iatom, (symbol, position) in enumerate(
            #        sorted(zip(symbols, positions), key=lambda x: x[0])):
            for iatom, (symbol, position) in enumerate(zip(symbols, positions)):
                output += ("atom {}{} {}  ".format(symbol, iatom + 1, symbol) +
                           "  ".join("{:06f}".format(position[j]) for j in range(3)) + "\n")


        if fmt.lower() == "vmd":
            positions = ase_cell.get_positions()
            output += "{:d}\n\n".format(len(positions))

            for iatom, (symbol, position) in enumerate(zip(symbols, positions)):
                output += ("{} ".format(symbol) +
                           "  ".join("{:06f}".format(position[j]) for j in range(3)) + "\n")

        if fmt.lower() == "force" or fmt.lower() == 'forces':
            forces = self.get_forces()
            output += "{:d}\n\n".format(len(forces))

            for iatom, (symbol, force) in enumerate(zip(symbols, forces)):
                output += ("{} ".format(symbol) +
                           "  ".join("{:06f}".format(force[j]) for j in range(3)) + "\n")


        if fmt.lower() == "fc":
            forces = self.get_fc()
            output += "{:d}\n\n".format(len(forces))

            for iatom, (symbol, force) in enumerate(zip(symbols, forces)):
                output += ("{} ".format(symbol) +
                           "  ".join("{:06f}".format(force[j]) for j in range(3)) + "\n")

        if fmt == "fdft":
            forces = self.get_fdft()
            output += "{:d}\n\n".format(len(forces))

            for iatom, (symbol, force) in enumerate(zip(symbols, forces)):
                output += ("{} ".format(symbol) +
                           "  ".join("{:06f}".format(force[j]) for j in range(3)) + "\n")

        return output
