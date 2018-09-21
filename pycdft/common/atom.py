import numpy as np
from ase import Atom as ASEAtom
from pycdft.common.units import angstrom_to_bohr, bohr_to_angstrom


class Atom(object):
    """ An atom in a specific cell.

    An atom can be initialized by and exported to an ASE Atom object.

    All physical quantities are in atomic unit.

    Attributes:
        sample (Sample): the sample where the atom lives.
        symbol (str): chemical symbol of the atom.
        abs_coord (np.ndarray, shape = [3]): absolute coordinate.
        cry_coord (np.ndarray, shape = [3]): crystal coordinate.

    Extra attributes are welcomed to be attached to an atom.
    """

    _extra_attr_to_print = []

    def __init__(self, sample, ase_atom):
        self.sample = sample
        self.symbol = ase_atom.symbol
        self.abs_coord = ase_atom.position * angstrom_to_bohr

    @property
    def cry_coord(self):
        return self.abs_coord @ self.sample.G.T / (2 * np.pi)

    @cry_coord.setter
    def cry_coord(self, cry_coord):
        self.abs_coord = cry_coord @ self.sample.R

    @property
    def ase_atom(self):
        return ASEAtom(symbol=self.symbol.capitalize(),
                       position=self.abs_coord * bohr_to_angstrom)

    def __repr__(self):
        rep = "Atom {}. Abs coord ({}). ".format(
                self.symbol,
                ", ".join("{:.3f}".format(coord) for coord in self.abs_coord)
            )
        if any(hasattr(self, attr) for attr in self._extra_attr_to_print):
            rep += "; ".join("{} = {}".format(
                        attr, getattr(self, attr)
                    )
                    for attr in self._extra_attr_to_print if hasattr(self, attr)
                )
        return rep

    def __str__(self):
        return self.__repr__()
