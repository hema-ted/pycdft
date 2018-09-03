from abc import ABCMeta, abstractmethod
import numpy as np
from pycdft.common.sample import Sample
from pycdft.optimizer import Optimizer


class Constraint(object):
    """ Constraint.

    Attributes:
        sample (Sample): the whole system.
        N (float): the electron number or electron number difference for this constraint
                   computed from the charge density of the current DFT step.
        N0 (float): the target value for the electron number or electron number difference,
                   N = N0 at convergence.
        optimizer (Optimizer): the optimizer for free energy.
        V (float): Lagrangian multiplier associate with the constraint.
        V_old (float): Lagrangian multiplier at the previous CDFT step.
        Vc (np.ndarray, shape = [nspin, n1, n2, n3]): constraint potential.
        weight (np.ndarray, shape = [nspin (optional), n1, n2, n3]): weight function.
        Ntol (float): convergence threshold for N - N0 (= dW/dV).
    """

    __metaclass__ = ABCMeta
    type = None

    @abstractmethod
    def __init__(self, sample: Sample, N0, optimizer: Optimizer,
                 V_init=0.0, Ntol=1.0E-2):
        """
        Args:
            V_init (float): initial guess for V.
        """
        self.sample = sample
        self.N0 = N0
        self.optimizer = optimizer
        self.V = self.V_old = V_init

        self.Ntol = Ntol

        self.weight = None
        self.N = None
        self.Vc = None
        self.Fc = None
        self.is_converged = None

        self.sample.constraints.append(self)

    @property
    def dW_by_dV(self):
        """ The derivative of free energy with respect to V.
        dW/dV = \int dr w_i(r) n(r) - N0 = N - N0"""
        return self.N - self.N0

    def update(self):
        """ Update the constraint with new value for V."""
        print("type: {}, N0 = {}, V = {}".format(self.type, self.N0, self.V))
        self.update_N()
        print("N = {}".format(self.N))
        print("dW/dV = {}".format(self.dW_by_dV))

        # Obtained a new V value from optimizer.
        V_new = self.optimizer.update(self.dW_by_dV, self.V, self.sample.W)
        print("V = {}".format(V_new))

        # Update constraint info.
        self.V_old = self.V
        self.V = V_new
        self.update_Vc()
        self.is_converged = bool(abs(self.N - self.N0) < self.Ntol)

    def update_structure(self):
        """ Update the constraint with new structure."""
        print("Updating constraint with new structure")
        self.update_weight()
        self.update_N()
        self.update_Vc()
        self.is_converged = False

    @abstractmethod
    def update_weight(self)-> None:
        """ Update the weight with new structure. """
        pass

    @abstractmethod
    def update_N(self) -> float:
        """ Update the electron number or electron number difference. """
        pass

    @abstractmethod
    def update_Vc(self)-> np.ndarray:
        """ Update constraint potential. """
        pass

    @abstractmethod
    def update_Fc(self)-> np.ndarray:
        """ Update constraint force. """
        pass
