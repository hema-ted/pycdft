import numpy as np
import os
import shutil
from copy import deepcopy
import scipy.optimize
from pycdft.common import Sample
from pycdft.constraint import Constraint
from pycdft.dft_driver import DFTDriver


class CDFTSCFConverged(Exception):
    pass


class CDFTSolver:
    """ Constrained DFT solver.

    Attributes:
        job (str): "scf" or "relax".
        sample (Sample): the whole system for which CDFT calculation is performed.
        constraints (list of Constraint): constraints on the system.
        dft_driver (DFTDriver): the interface to DFT code (e.g. Qbox or PW).
        maxiter (int): maximum number of CDFT iterations.
        maxstep (int): maximum relaxation steps.
        F_tol (float): force threshold for relaxation.
        Vc_tot (float array, shape == [nspin, n1, n2, n3]): total constraint potential
            as a sum of all constraints defined on all fragments.
    """

    _archive_folder = 'outputs'

    def __init__(self, job: str, sample: Sample, dft_driver: DFTDriver,
                 optimizer: str="secant", maxiter: int=1000, maxstep: int=100,
                 F_tol: float=1.0E-2):

        self.job = job
        self.sample = sample
        self.constraints = self.sample.constraints
        self.dft_driver = dft_driver
        self.optimizer = optimizer
        self.maxiter = maxiter
        self.maxstep = maxstep
        self.F_tol = F_tol
        self.Vc_tot = None
        self.itscf = None

        if os.path.exists(self._archive_folder):
            shutil.rmtree(self._archive_folder)
            os.makedirs(self._archive_folder)
        else:
            os.makedirs(self._archive_folder)

    def solve(self):
        """ Solve CDFT SCF or relax problem."""
        self.dft_driver.get_rho_r()
        if self.job == "scf":
            self.solve_scf()
        elif self.job == "relax":
            self.solve_relax()
        else:
            raise ValueError
        self.dft_driver.get_wfc()

    def solve_scf(self):
        """ Iteratively solve the CDFT problem.

        An outer loop (implemented below) is performed to maximize the free energy w.r.t.
        Lagrangian multipliers for all constrains, an inner loop (casted to a KS problem and
        outsourced to DFT code) is performed to minimize the free energy w.r.t. charge density.
        """

        self.sample.update_weights()

        if self.optimizer in ["secant", "bisect", "brentq", "brenth"]:
            assert len(self.constraints) == 1

        self.itscf = 0
        try:
            if self.optimizer == "secant":
                self.constraints[0].V_init = scipy.optimize.newton(
                    func=self.solve_scf_for_dW_by_dV,
                    x0=self.constraints[0].V_init,
                    maxiter=self.maxiter
                )
            elif self.optimizer == "bisect":
                self.constraints[0].V_init, info = scipy.optimize.bisect(
                    f=self.solve_scf_for_dW_by_dV,
                    a=self.constraints[0].V_brak[0],
                    b=self.constraints[0].V_brak[1],
                    maxiter=self.maxiter
                )
            elif self.optimizer == "brentq":
                self.constraints[0].V_init, info = scipy.optimize.brentq(
                    f=self.solve_scf_for_dW_by_dV,
                    a=self.constraints[0].V_brak[0],
                    b=self.constraints[0].V_brak[1],
                    maxiter=self.maxiter
                )
            elif self.optimizer == "brenth":
                self.constraints[0].V_init, info = scipy.optimize.brenth(
                    f=self.solve_scf_for_dW_by_dV,
                    a=self.constraints[0].V_brak[0],
                    b=self.constraints[0].V_brak[1],
                    maxiter=self.maxiter
                )
            elif self.optimizer in ["BFGS"]:
                res = scipy.optimize.minimize(
                    method=self.optimizer,
                    fun=self.solve_scf_with_new_V,
                    x0=np.array(list(c.V_init for c in self.constraints)),
                    jac=True,
                    options={"maxiter": self.maxiter}
                )
                for c, V in zip(self.constraints, res.x):
                    c.V_init = V
            else:
                raise ValueError

        except CDFTSCFConverged:
            print("CDFTSolver: convergence achieved!")
        # else:
        #     print("CDFTSolver: convergence NOT achieved after {} iterations.".format(self.maxiter))

        if self.job == "relax":
            self.dft_driver.get_force()

    def solve_scf_with_new_V(self, Vs):
        """ Given V for all constraints, solve KS problem."""
        self.itscf += 1

        # Update constraints
        for c, V in zip(self.constraints, Vs):
            c.V = V
            c.update_Vc()

        # Compute the total constraint potential Vc.
        self.Vc_tot = np.sum(c.Vc for c in self.constraints)

        # Impose the constraint potential Vc to DFT code.
        self.dft_driver.set_Vc(self.Vc_tot)

        # Order DFT code to perform SCF calculation under the constraint potential Vc.
        # After dft driver run_scf command should read etotal and force
        self.dft_driver.run_scf()
        self.dft_driver.get_rho_r()
        self.sample.W = self.sample.Edft_total - np.sum(c.V * c.N for c in self.constraints)

        # Print intermediate results
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Iter {}:".format(self.itscf))
        print("W (free energy) = {}".format(self.sample.W))
        print("E (DFT KS energy) = {}".format(self.sample.Edft_bare))
        print("Constraint info:")
        for i, c in enumerate(self.constraints):
            c.update_N()
            print("Constraint {} (type = {}, N0 = {}, V = {}):".format(
                i, c.type, c.N0, c.V
            ))
            print("N = {}".format(c.N))
            print("dW/dV = N - N0 = {}".format(c.dW_by_dV))

        if all(c.is_converged for c in self.constraints):
            raise CDFTSCFConverged

        # return the negative of W and dW/dV to be used by scipy minimizers
        return -self.sample.W, np.array(list(-c.dW_by_dV for c in self.constraints))

    def solve_scf_for_dW_by_dV(self, V):
        """ Wrapper function for solve_scf_with_new_V returning dW/dV."""
        return self.solve_scf_with_new_V([V])[1][0]

    def solve_relax(self):
        """ Relax the structure under constraint.

        A force from constraint potential is added to DFT force during relaxation.
        """

        for istep in range(1, self.maxstep + 1):
            # run SCF to converge electronic structure
            self.solve_scf()

            maxforce, maxforceatom = self.sample.get_maxforce()
            print("Maximum force = {} au, on {}".format(maxforce, maxforceatom))
            if maxforce < self.F_tol:
                print("CDFTSolver: force convergence achieved!")
                break

            # add constraint force to DFT force
            Fc_total = np.zeros([self.sample.natoms, 3])
            for c in self.constraints:
                Fc_total += c.update_Fc()

            # add constraint force to DFT force
            self.dft_driver.set_Fc(Fc_total)

            # run optimization
            self.dft_driver.run_opt()

            # parse updated coordinates
            self.dft_driver.get_structure()

            # update weights and constraint potentials
            for c in self.constraints:
                c.update_R(istep)

            print("================================")
            print("Structure updated")
            print("================================")

        else:
            print("CDFTSolver: relaxation NOT achieved after {} steps.".format(self.maxstep))

    def copy(self):
        return deepcopy(self)
