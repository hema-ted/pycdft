from abc import ABCMeta, abstractmethod
from pycdft.common.sample import Sample


class DFTDriver(object):
    """ DFT driver.

    Attributes:
        sample (Sample): the whole system for which CDFT calculation is performed.
        istep (int): current geometry optimization step.
        icscf (int): current constrained SCF step.
        output_path (str): output file path.
    """

    __metaclass__ = ABCMeta

    def __init__(self, sample: Sample):
        self.sample = sample
        self.istep = None
        self.icscf = None
        self.output_path = None

    @abstractmethod
    def reset(self, output_path):
        """ Reset the DFT code."""
        pass

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
        """ Order the DFT code to perform SCF calculation under the constraint.

        Returns when SCF calculation is finished.
        """
        pass

    @abstractmethod
    def run_opt(self):
        """ Order the DFT code to run one structure relaxation step."""
        pass

    @abstractmethod
    def get_rho_r(self):
        """ Fetch the charge density from the DFT code, write to self.sample.rhor."""
        pass

    @abstractmethod
    def get_force(self):
        """ Fetch the DFT force from the DFT code."""
        pass

    @abstractmethod
    def set_Fc(self):
        """ Set the constraint force in the DFT code."""
        pass

    @abstractmethod
    def get_structure(self):
        """ Fetch the structure from the DFT code, write to self.sample."""
        pass

    @abstractmethod
    def get_wfc(self):
        """ Fetch the wavefunction from the DFT code."""
        pass
