import numpy as np
from pycdft.common.sample import Sample
from pycdft.common.fragment import Fragment
from pycdft.constraint.base import Constraint


class ChargeConstraint(Constraint):
    r""" Constraint on the absolute electron number of a fragment.

    For fragment F, Hirshfeld weight is defined as 
    :math:`w_\text{F}({\bf r}) =  \frac{\sum_{I \in F} \rho_I}{\sum_I \rho_I}`


    Extra attributes:
         fragment (Fragment): a fragment of the whole system.
    """

    type = "charge"

    def __init__(self, sample: Sample, fragment: Fragment, N0: float,
                 V_init=0, V_brak=(-1, 1), N_tol=1.0E-3, eps=1e-6):
        super(ChargeConstraint, self).__init__(
            sample, N0, V_init=V_init, V_brak=V_brak, N_tol=N_tol,
        )
        self.fragment = fragment
        self.eps = eps
        print(f"Constraint: type = {self.type}, N_tol = {self.N_tol:.5f}, eps = {self.eps:.2E}")

    def update_w(self):
        w = self.fragment.rhopro_r / self.sample.rhopro_tot_r
        w[self.sample.rhopro_tot_r < self.eps] = 0.0
        if self.sample.vspin == 1:
            self.w = w[None, ...]
        else:
            self.w = np.append(w, w, axis=0).reshape(2, *w.shape)

    def compute_w_grad_r(self, atom):
        delta = 1 if atom in self.fragment.atoms else 0
        rho_grad_r = self.sample.compute_rhoatom_grad_r(atom)
        w_grad = np.einsum(
            "sijk,aijk,ijk->asijk", delta - self.w, rho_grad_r, 1/self.sample.rhopro_tot_r
        )
        for icart, ispin in np.ndindex(3, self.sample.vspin):
            w_grad[icart, ispin][self.sample.rhopro_tot_r < self.eps] = 0.0
        return w_grad

    # added for debugging forces 
    def debug_w_grad_r(self, atom):
        delta = 1 if atom in self.fragment.atoms else 0
        rho_grad_r = self.sample.compute_rhoatom_grad_r(atom)
        w_grad = np.einsum(
            "sijk,aijk,ijk->asijk", delta - self.w, rho_grad_r, 1/self.sample.rhopro_tot_r
        )
#        w_grad_part = np.einsum(
#            "sijk,ijk->sijk", delta - self.w, 1/self.sample.rhopro_tot_r
#        )
        for icart, ispin in np.ndindex(3, self.sample.vspin):
            w_grad[icart, ispin][self.sample.rhopro_tot_r < self.eps] = 0.0
        return w_grad, rho_grad_r
