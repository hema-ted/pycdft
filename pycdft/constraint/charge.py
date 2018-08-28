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

    def __init__(self, sample: Sample, fragment: Fragment, N0, optimizer: Optimizer,
                 V_init=0.0, Ntol=None, Vtol=None, dVtol=1.0E-2):
        super(ChargeConstraint, self).__init__(sample, N0, optimizer, V_init=V_init,
                                               Ntol=Ntol, Vtol=Vtol, dVtol=dVtol)
        self.fragment = fragment
        self.update_structure()

    def update_structure(self):
        self.fragment.update()
        self.is_converged = False

    def compute_N(self) -> float:
        omega = self.sample.cell.omega
        n123 = self.sample.fftgrid.N
        rhor = self.sample.rhor
        return (omega / n123) * np.sum(np.einsum("ijk,sijk->s", self.fragment.w, rhor))

    def compute_Vc(self):
        nspin = self.sample.nspin
        w = self.fragment.w
        return self.V * np.append(w, w, axis=0).reshape(nspin, *w.shape)

    def compute_Fc(self):
        n123 = self.sample.fftgrid.N
        omega = self.sample.cell.omega
        rhor = np.sum(self.sample.rhor, axis=0)

        Fc = np.zeros([self.sample.cell.natoms, 3])

        for iatom, atom in enumerate(self.sample.cell.atoms):
            w_grad = self.fragment.compute_w_grad(atom)
            Fc[iatom] = self.V * np.einsum("ijk,aijk->a", rhor, w_grad) * omega / n123

        return Fc