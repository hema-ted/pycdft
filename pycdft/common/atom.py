from __future__ import absolute_import, division, print_function

import numpy as np
from ase import Atom as ASEAtom
from six import string_types

from .units import angstrom_to_bohr, bohr_to_angstrom
from .pp import SG15PP


class Atom(object):
    """ An atom in a specific cell.

    An atom can be initialized by and exported to an ASE Atom object.

    All internal quantities below are in atomic unit.

    Attributes:
        cell(Cell): the cell where the atom resides.
        symbol(str): chemical symbol of the atom.
        abs_coord(float array): absolute coordinate.
        cry_coord(float array): crystal coordinate.

    Extra attributes are welcomed to be attached to an atom.
    """

    _extra_attr_to_print = ["velocity", "force", "freezed", "fc", "fdft"]

    def __init__(self, cell, ase_atom=None, symbol=None, cry_coord=None, abs_coord=None, **kwargs):
        self.cell = cell

        if ase_atom is not None:
            assert isinstance(ase_atom, ASEAtom)
            self.symbol = ase_atom.symbol
            self.abs_coord = ase_atom.position * angstrom_to_bohr
        else:
            assert isinstance(symbol, string_types)
            assert bool(cry_coord is None) != bool(abs_coord is None)
            self.symbol = symbol
            if cry_coord is not None:
                self.cry_coord = np.array(cry_coord)
            else:
                self.abs_coord = np.array(abs_coord)

        for i in range(3):
            if not 0 <= self.cry_coord[i] < 1:
                self.cry_coord[i] = self.cry_coord[i] % 1

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        info = "Atom {}. Cry coord ({}). Abs coord ({}). ".format(
            self.symbol,
            ", ".join("{:.6f}".format(coord) for coord in self.cry_coord),
            ", ".join("{:.6f}".format(coord) for coord in self.abs_coord)
        )
        if any(hasattr(self, attr) for attr in self._extra_attr_to_print):
            info += " Extra attributes: {}".format(
                "; ".join("{}: {}".format(
                    attr, getattr(self, attr)
                )
                          for attr in self._extra_attr_to_print if hasattr(self, attr)
                          )
            )
        return info

    def __str__(self):
        return self.__repr__()

    @property
    def abs_coord(self):
        return np.dot(self.cry_coord, self.cell.R)

    @abs_coord.setter
    def abs_coord(self, value):
        self.cry_coord = np.dot(value, self.cell.G.T / (2 * np.pi))

    @property
    def ase_atom(self):
        return ASEAtom(symbol=self.symbol, position=self.abs_coord * bohr_to_angstrom)

    def nel(self, pseudos="SG15"):
        """ Compute # of electrons according to certain pseudopotential family."""
        if pseudos == "SG15":
            return SG15PP[self.symbol]["nel"]
        else:
            raise NotImplementedError
