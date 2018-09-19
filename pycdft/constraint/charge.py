import numpy as np
from pycdft.common.sample import Sample
from pycdft.common.fragment import Fragment
from pycdft.constraint.base import Constraint


class ChargeConstraint(Constraint):
    """ Constraint on the absolute electron number of a fragment.

    Extra attributes:
        fragment (Fragment): a fragment of the whole system.
    """

    type = "charge"
    _eps = 0.0001  # cutoff of Hirshfeld weight when the density approaches zero

    def __init__(self, sample: Sample, fragment: Fragment, N0: float,
                 V_init=0, V_brak=(-1, 1), N_tol=1.0E-3):
        super(ChargeConstraint, self).__init__(
            sample, N0, V_init=V_init, V_brak=V_brak, N_tol=N_tol
        )
        self.fragment = fragment

    def update_w(self):
        w = self.fragment.rhopro_r / self.sample.rhopro_tot_r
        w[self.sample.rhopro_tot_r < self._eps] = 0.0
        if self.sample.nspin == 1:
            self.w = w[None, ...]
        else:
            self.w = np.append(w, w, axis=0).reshape(2, *w.shape)

    def compute_w_grad_r(self, atom):
        delta = 1 if atom in self.fragment.atoms else 0
        rho_grad_r = self.sample.compute_rhoatom_grad_r(atom)
        return (delta - self.w) * rho_grad_r / self.sample.rhopro_tot_r
