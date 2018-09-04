import numpy as np
from pycdft.common.sample import Sample
from pycdft.common.fragment import Fragment
from pycdft.optimizer import Optimizer
from .base import Constraint


class ChargeConstraint(Constraint):
    """ Constraint on the absolute electron number of a fragment.

    Extra attributes:
        fragment (Fragment): a fragment of the whole system.
    """

    type = "charge"
    _eps = 0.0001  # cutoff of Hirshfeld weight when the density approaches zero

    def __init__(self, sample: Sample, fragment: Fragment, N0, optimizer: Optimizer,
                 V_init=0.0, N_tol=1.0E-3):
        super(ChargeConstraint, self).__init__(sample, N0, optimizer, V_init=V_init, N_tol=N_tol)
        self.fragment = fragment

    def update_w(self):
        w = self.fragment.rhopro_r / self.sample.rhopro_tot_r
        w[self.sample.rhopro_tot_r < self._eps] = 0.0
        if self.sample.nspin == 1:
            self.w = w[None, ...]
        else:
            self.w = np.append(w, w, axis=0).reshape(2, *w.shape)

        self.is_converged = False

    def update_Fc(self):
        omega = self.sample.omega
        n123 = self.sample.n123
        rhor = np.sum(self.sample.rho_r, axis=0)
        self.Fc = np.zeros([self.sample.natoms, 3])

        for iatom, atom in enumerate(self.sample.atoms):
            w_grad = self.fragment.compute_w_grad(atom, self.w)
            self.Fc[iatom] = (omega * n123) * self.V * np.einsum("ijk,aijk->a", rhor, w_grad)
