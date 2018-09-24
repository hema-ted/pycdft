from abc import ABCMeta, abstractmethod
import numpy as np
from pycdft.common.sample import Sample


class Constraint(object):
    """ Constraint.

    Attributes:
        sample (Sample): the whole system.
        N (float): the electron number or electron number difference for this constraint
                   computed from the charge density of the current DFT step.
        N0 (float): the target value for the electron number or electron number difference,
                   N = N0 at convergence.
        V (float): Lagrangian multiplier associate with the constraint.
        V_init (float): Initial guess or bracket for V, used for certain optimization algorithms.
        V_brak (2-tuple of float): Search bracket for V, used for certain optimization algorithms.
        Vc (np.ndarray, shape = [vspin, n1, n2, n3]): constraint potential.
        w (np.ndarray, shape = [vspin, n1, n2, n3]): weight function.
        N_tol (float): convergence threshold for N - N0 (= dW/dV).
    """

    __metaclass__ = ABCMeta
    type = None

    @abstractmethod
    def __init__(self, sample: Sample, N0, V_init=None, V_brak=None, N_tol=None):
        """
        Args:
            V_init (float): initial guess for V.
        """
        self.sample = sample
        self.N0 = N0
        self.V_init = V_init
        self.V_brak = V_brak
        self.N_tol = N_tol

        self.V = None
        self.w = None
        self.N = None
        self.Vc = None
        self.Fc = None

        self.sample.constraints.append(self)

    @property
    def dW_by_dV(self):
        """ The derivative of free energy with respect to V.
        dW/dV = \int dr w_i(r) n(r) - N0 = N - N0"""
        return self.N - self.N0

    @property
    def is_converged(self):
        return bool(abs(self.N - self.N0) < self.N_tol)

    def update_structure(self):
        """ Update the constraint with new structure."""
        print("Updating constraint with new structure...")
        self.update_w()
        self.update_N()

    @abstractmethod
    def update_w(self):
        """ Update the weight with new structure. """
        pass

    def update_N(self):
        """ Update the electron number or electron number difference. """
        omega = self.sample.omega
        n = self.sample.n1 * self.sample.n2 * self.sample.n3
        rho_r = self.sample.rho_r
        self.N = (omega / n) * np.sum(self.w * rho_r)

    def update_Vc(self):
        """ Update constraint potential. """
        self.Vc = self.V * self.w

    def update_Fc(self):
        """ Update constraint force. """
        omega = self.sample.omega
        n = self.sample.n
        self.Fc = np.zeros([self.sample.natoms, 3])

        for iatom, atom in enumerate(self.sample.atoms):
            w_grad = self.compute_w_grad_r(atom)
            self.Fc[iatom] = - self.V * (omega / n) * np.einsum(
                "sijk,asijk->a", self.sample.rho_r, w_grad
            )

    @abstractmethod
    def compute_w_grad_r(self, atom):
        pass
