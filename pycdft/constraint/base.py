from abc import ABCMeta, abstractmethod
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
        Vc (np.array, shape == [nspin, n1, n2, n3]): constraint potential.
        Ntol (float): convergence threshold for N.
        Vtol (float): convergence threshold for V.
        dVtol (float): convergence threshold for dW/dV.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, sample: Sample, N0, optimizer: Optimizer,
                 V_init=0.0, Ntol=None, Vtol=None, dVtol=1.0E-2):
        """
        Args:
            V_init (float): initial guess for V.
        """
        self.sample = sample
        self.N0 = N0
        self.optimizer = optimizer
        self.V = self.V_old = V_init

        if all((Ntol is None, Vtol is None, dVtol is None)):
            raise ValueError("At least one of Ntol, Vtol or dVtol has to be specified.")
        self.Ntol = Ntol
        self.Vtol = Vtol
        self.dVtol = dVtol

        self.type = None
        self.Vc = None
        self.N = None
        self.is_converged = False

    @property
    def dW_by_dV(self):
        """ The derivative of free energy with respect to V.
        dW/dV = \int dr w_i(r) n(r) - N0 = N - N0"""
        return self.N - self.N0

    def update_V(self):
        """ Update the constraint with new value for V."""
        print("type: {}, N0 = {}, V = {}".format(self.type, self.N0, self.V))
        self.N = self.compute_N()
        print("N = {}".format(self.N))
        print("dW/dV = {}".format(self.dW_by_dV))

        # Obtained a new V value from optimizer.
        V_new = self.optimizer.update(self.dW_by_dV, self.V, self.sample.W)
        print("updated V = {}".format(V_new))

        # Update constraint info.
        self.V_old = self.V
        self.V = V_new
        self.Vc = self.compute_Vc()
        self.is_converged = self.check_convergence()

    def check_convergence(self):
        """ Check convergence for N, V and dW/dV."""
        if self.Ntol is None:
            Nconv = "-"
        else:
            if abs(self.N - self.N0) < self.Ntol:
                Nconv = "yes"
            else:
                Nconv = "no"

        if self.Vtol is not None:
            Vconv = "-"
        else:
            if abs(self.V - self.V_old) < self.Vtol:
                Vconv = "yes"
            else:
                Vconv = "no"

        if self.dVtol is not None:
            dVconv = "-"
        else:
            if abs(self.dW_by_dV) < self.dVtol:
                dVconv = "yes"
            else:
                dVconv = "no"

        print("convergence: N ({}), V ({}), dW/dV ({}).".format(Nconv, Vconv, dVconv))
        return not any(conv == "no" for conv in [Nconv, Vconv, dVconv])

    @abstractmethod
    def update_structure(self):
        """ Update the constraint with new structure. """
        pass

    @abstractmethod
    def compute_N(self) -> float:
        """ Update the electron number or electron number difference. """
        pass

    @abstractmethod
    def compute_Vc(self):
        """"""
        pass

    @abstractmethod
    def compute_Fc(self):
        """ Compute constraint force. """
        pass
