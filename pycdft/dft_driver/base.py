from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod


class DFTDriver(object):
    __metaclass__ = ABCMeta

    def __init__(self, sample):
        self.sample = sample

    @abstractmethod
    def set_Vc(self, Vc):
        """ Set the constraint potential Vc in DFT code.

        Given constraint potential Vc as an array of shape [nspin, n1, n2, n3],
        this method send the constraint potential to the DFT code.

        Args:
            Vc: the constraint potential.
        """
        pass

    @abstractmethod
    def run_scf(self):
        """ Let the DFT code perform SCF calculation under the constraint.

        Returns when SCF calculation is finished.
        """
        pass

    @abstractmethod
    def run_opt(self):
        """ Let the DFT code to run one structure relaxation step."""
        pass

    @abstractmethod
    def fetch_rhor(self):
        """ Fetch the charge density from the DFT code, write to self.sample.rhor."""
        pass

    @abstractmethod
    def fetch_force(self):
        """ Fetch the force from the DFT code."""
        pass

    @abstractmethod
    def set_force(self, Fc):
        """ Set the force in the DFT code."""
        pass

    @abstractmethod
    def fetch_structure(self):
        """ Fetch the structure from the DFT code, write to self.sample.cell."""
        pass
