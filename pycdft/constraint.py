from __future__ import absolute_import, division, print_function

import numpy as np
from .weight import Weight
from .common.fragment import Fragment
from .optimizer import Optimizer
from .weight import HirshfeldWeight


class Constraint(object):
    """ Constraint.

    Attributes:
        sample (Sample): whole system.
        fragment (Fragment): the fragment of the system where the constraint is defined.
        N (float): constraint for the electron number on the fragment.

        weight (str): only support "Hirshfeld" now.

        optimizer (Optimizer): the optimizer for free energy.
        V (float): Lagrangian multiplier associate with the constraint.
        V_old (float): Lagrangian multiplier at the previous CDFT step.

        Vc (np.array, shape == [nspin, n1, n2, n3]): constraint potential.

        Vtol (float): convergence threshold for V.
        Ntol (float): convergence threshold for N.

    """

    def __init__(self, fragment, N, weight, optimizer, atomic_density_files,
                 V0=0.0, Vtol=1.0E-2):
        """
        Args:
            V0 (float): initial guess for V.
        """
        assert isinstance(fragment, Fragment)
        assert isinstance(N * 1.0, float)
        assert isinstance(optimizer, Optimizer)

        self.sample = fragment.sample
        self.fragment = fragment
        self.N = N

        if weight == "Hirshfeld":
            self.weight = HirshfeldWeight(self.fragment, atomic_density_files)
        else:
            raise ValueError

        self.optimizer = optimizer
        #self.optimizer.setup(x0=V0)
        self.V = self.V_old = V0

        self.Vtol = Vtol

        self.Vc = None
        self.dW_by_dV = None
        self.update_R(0)

    def update_R(self, istep):
        """ Update the constraint with new structure. """
        self.weight.update(istep)
        self.Vc = self.V * self.weight.w

    def update_V(self, W):
        """ Update the constraint with new value for V."""
        omega = self.sample.cell.omega
        n123 = self.sample.fftgrid.N
        rhor = self.sample.rhor

        # Compute the derivative of free energy with respect to V.
        # dW/dV = \int dr w_i(r) n(r) - N
        self.dW_by_dV = np.sum(self.weight.w * rhor) * omega / n123 - self.N
        print("Integrated rhor = {}".format(np.sum(rhor) * omega / n123))
        print("Integrated w*rhor = {}".format(np.sum(self.weight.w * rhor) * omega / n123))
        print("N = {}".format(self.N))
        print("dW/dV = {}".format(self.dW_by_dV))

        # Obtained a new V value from optimizer.
        V_new = self.optimizer.update(self.dW_by_dV, self.V, W)

        # Update constraint info.
        self.V_old = self.V
        self.V = V_new
        self.Vc = self.V * self.weight.w

    def is_converged(self):
        """ Check convergence."""
        return abs(self.dW_by_dV) < self.Vtol
        #return abs(self.V - self.V_old) < self.Vtol

    def compute_pre_Fc(self):
        """ Compute constraint force. """
        n123 = self.sample.fftgrid.N
        omega = self.sample.cell.omega
        rhor = np.sum(self.sample.rhor, axis=0)

        Fc = np.zeros([self.sample.cell.natoms, 3])

        for iatom, atom in enumerate(self.sample.cell.atoms):
            w_grad = self.weight.compute_w_grad(atom)
            Fc[iatom] = self.V * np.einsum("ijk,aijk->a", rhor, w_grad) * omega / n123

        return Fc
