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

    _eps = 0.0001  # cutoff of Hirshfeld weight when the density approaches zero

    def __init__(self, sample: Sample, fragment: Fragment, N0, optimizer: Optimizer,
                 V_init=0.0, Ntol=None, Vtol=None, dVtol=1.0E-2):
        super(ChargeConstraint, self).__init__(sample, N0, optimizer, V_init=V_init,
                                               Ntol=Ntol, Vtol=Vtol, dVtol=dVtol)
        self.fragment = fragment
        self.update_structure()

    def update_structure(self):
        self.weight = self.fragment.rhopro / self.sample.cell.rhopro_tot
        self.weight[self.sample.cell.rhopro_tot < self._eps] = 0.0
        self.is_converged = False

    def compute_Fc(self):
        n123 = self.sample.fftgrid.N
        omega = self.sample.cell.omega
        rhor = np.sum(self.sample.rhor, axis=0)

        Fc = np.zeros([self.sample.cell.natoms, 3])

        for iatom, atom in enumerate(self.sample.cell.atoms):
            w_grad = self.fragment.compute_w_grad(atom)
            Fc[iatom] = self.V * np.einsum("ijk,aijk->a", rhor, w_grad) * omega / n123

        return Fc
