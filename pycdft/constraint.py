from __future__ import absolute_import, division, print_function

import numpy as np
from .common.sample import Sample
from .optimizer import Optimizer
from .weight import Weight


class Constraint(object):
    """ Constraint.

    Attributes:
        sample (Sample): the whole system.
        N (float): the electron number or electron number difference for this constraint
                   computed from the charge density of the current DFT step.
        N0 (float): the target value for the electron number or electron number difference,
                   N = N0 at convergence.
        weight (Weight): weight function.
        optimizer (Optimizer): the optimizer for free energy.
        V (float): Lagrangian multiplier associate with the constraint.
        V_old (float): Lagrangian multiplier at the previous CDFT step.
        Vc (np.array, shape == [nspin, n1, n2, n3]): constraint potential.
        Ntol (float): convergence threshold for N.
        Vtol (float): convergence threshold for V.
        dVtol (float): convergence threshold for dW/dV.
    """

    def __init__(self, N0, weight: Weight, optimizer: Optimizer,
                 V_init=0.0, Ntol=None, Vtol=None, dVtol=1.0E-2):
        """
        Args:
            V_init (float): initial guess for V.
        """
        self.N0 = N0
        self.weight = weight
        self.sample = self.weight.sample
        self.optimizer = optimizer
        self.V = self.V_old = V_init

        if all((Ntol is None, Vtol is None, dVtol is None)):
            raise ValueError("At least one of Ntol, Vtol or dVtol should be specified.")
        self.Ntol = Ntol
        self.Vtol = Vtol
        self.dVtol = dVtol

        self.Vc = None
        self.N = None
        self.dW_by_dV = None
        self.is_converged = False

        self.update_structure()

    def update_structure(self):
        """ Update the constraint with new structure. """
        self.weight.update()
        self.is_converged = False

    def update_V(self, W):
        """ Update the constraint with new value for V."""
        print("Updating constraint (type: {}, N0 = {})...".format(self.weight.weight_type, self.N0))

        omega = self.sample.cell.omega
        n123 = self.sample.fftgrid.N
        rhor = self.sample.rhor

        self.N = np.sum(rhor) * omega / n123
        print("N = {}".format(self.N))

        # Compute the derivative of free energy with respect to V.
        # dW/dV = \int dr w_i(r) n(r) - N
        self.dW_by_dV = np.sum(self.weight.w * rhor) * omega / n123 - self.N0
        print("dW/dV = {}".format(self.dW_by_dV))

        # Obtained a new V value from optimizer.
        V_new = self.optimizer.update(self.dW_by_dV, self.V, W)

        # Update constraint info.
        self.V_old = self.V
        self.V = V_new
        self.Vc = self.V * self.weight.w

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

    # def compute_Fc(self):
    #     """ Compute constraint force. """
    #     n123 = self.sample.fftgrid.N
    #     omega = self.sample.cell.omega
    #     rhor = np.sum(self.sample.rhor, axis=0)
    #
    #     Fc = np.zeros([self.sample.cell.natoms, 3])
    #
    #     for iatom, atom in enumerate(self.sample.cell.atoms):
    #         w_grad = self.weight.compute_w_grad(atom)
    #         Fc[iatom] = self.V * np.einsum("ijk,aijk->a", rhor, w_grad) * omega / n123
    #
    #     return Fc
