import numpy as np
from pycdft.common.sample import Sample
from pycdft.common.fragment import Fragment
from pycdft.optimizer import Optimizer
from .base import Constraint


class ChargeTransferConstraint(Constraint):
    """ Constraint on electron number difference between a donor and an acceptor fragment

    Extra attributes:
        donor (Fragment): donor fragment.
        acceptor (Fragment): acceptor fragment.
    """

    def __init__(self, sample: Sample, donor: Fragment, acceptor: Fragment, N0,
                 optimizer: Optimizer, V_init=0.0, Ntol=None, Vtol=None, dVtol=1.0E-2):
        super(ChargeTransferConstraint, self).__init__(sample, N0, optimizer, V_init=V_init,
                                                       Ntol=Ntol, Vtol=Vtol, dVtol=dVtol)
        self.donor = donor
        self.acceptor = acceptor
        self.update_structure()

    def update_structure(self):
        self.donor.update()
        self.acceptor.update()
        self.is_converged = False

    def compute_N(self) -> float:
        omega = self.sample.cell.omega
        n123 = self.sample.fftgrid.N
        rhor = self.sample.rhor
        return np.sum((self.donor.w - self.acceptor.w) * rhor) * omega / n123

    def compute_Vc(self):
        return self.V * (self.donor.w - self.acceptor.w)

    def compute_Fc(self):
        n123 = self.sample.fftgrid.N
        omega = self.sample.cell.omega
        rhor = np.sum(self.sample.rhor, axis=0)

        Fc = np.zeros([self.sample.cell.natoms, 3])

        for iatom, atom in enumerate(self.sample.cell.atoms):
            w_grad = self.donor.compute_w_grad(atom) - self.acceptor.compute_w_grad(atom)
            Fc[iatom] = self.V * np.einsum("ijk,aijk->a", rhor, w_grad) * omega / n123

        return Fc
