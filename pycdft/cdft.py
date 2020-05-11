import numpy as np
import os
import shutil
from copy import deepcopy
import time
import sys
import io
import base64
import scipy.optimize
from pycdft.common import Sample, timer
from pycdft.constraint import Constraint
from pycdft.dft_driver import DFTDriver

class CDFTSCFConverged(Exception):
    """ Test for scf convergence. """
    pass


class CDFTSolver:
    """ Constrained DFT solver.

    All quantities are expressed in Hartree atomic units.

    Attributes:
        job (str): "scf" for SCF calculations or "opt" for geometry optimization.
        sample (:class:`Sample`): the whole system for which CDFT calculation is performed.
        constraints (list of :class:`Constraint`): constraints on the system.
        dft_driver (:class:`DFTDriver`): the interface to DFT code (e.g., Qbox or PWscf).
        optimizer (str): optimization strategy for constrained Hamiltonian, default = "secant"
        maxcscf (int): maximum number of CDFT iterations, default =1000.
        maxstep (int): maximum geometry optimization steps, default = 100.
        F_tol (float): force threshold for optimization, default = 1e-02.

    Internal Parameters:
        Vc_tot (float array, shape == [vspin, n1, n2, n3]): total constraint potential
            as a sum of all constraints defined on all fragments.
    """

    nsolver = 0

    def __init__(self, job: str, sample: Sample, dft_driver: DFTDriver,
                 optimizer: str = "secant", maxcscf: int = 1000, maxstep: int = 100,
                 F_tol: float = 1.0E-2, lrestart: bool=False):

        self.job = job
        self.sample = sample
        self.constraints = self.sample.constraints
        self.dft_driver = dft_driver
        self.optimizer = optimizer
        self.maxcscf = maxcscf
        self.maxstep = maxstep
        self.F_tol = F_tol
        self.Vc_tot = None
        self.itscf = None
        self.start_time = time.time()
        self.lrestart = lrestart

        if not self.lrestart:
           # make output folder, keeping any previous runs
           CDFTSolver.nsolver += 1
           self.isolver = CDFTSolver.nsolver
           self.output_path = "./pycdft_outputs/solver{}/".format(self.isolver)
           if os.path.exists(self.output_path):
               shutil.rmtree(self.output_path)
               os.makedirs(self.output_path)
           else:
               os.makedirs(self.output_path)
           self.dft_driver.reset(self.output_path)

    def solve(self):
        """ Solve CDFT SCF or optimization problem."""
        print("===================== Initializing Run =====================")
        if self.job not in ['scf', 'opt']:
            raise ValueError

        try:
            self.dft_driver.get_rho_r()
            if self.job == "scf":
                self.solve_scf()
            else:
                self.solve_opt()
            self.dft_driver.get_wfc()
        except:
            print(sys.exc_info())
            print("Terminating DFT driver...")
            self.dft_driver.exit() 

    def solve_scf(self):
        """ Iteratively solve the CDFT problem.

        An outer loop (implemented below) is performed to maximize the free energy w.r.t.
        Lagrangian multipliers for all constrains, an inner loop (casted to a KS problem and
        outsourced to DFT code) is performed to minimize the free energy w.r.t. charge density.

        Internal Parameters:
             secant: requires V_init
             bisect, brentq, brenth: requires V_brak

        Note:
             For methods requiring V_brak = [a,b], the objective function f must be continuous
             and f(a) and f(b) must have opposite signs. 
             see scipy.optimize documentation
           
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
                    maxiter=self.maxcscf
                )
            elif self.optimizer == "bisect":
                self.constraints[0].V_init, info = scipy.optimize.bisect(
                    f=self.solve_scf_for_dW_by_dV,
                    a=self.constraints[0].V_brak[0],
                    b=self.constraints[0].V_brak[1],
                    maxiter=self.maxcscf
                )
            elif self.optimizer == "brentq":
                self.constraints[0].V_init, info = scipy.optimize.brentq(
                    f=self.solve_scf_for_dW_by_dV,
                    a=self.constraints[0].V_brak[0],
                    b=self.constraints[0].V_brak[1],
                    maxiter=self.maxcscf
                )
            elif self.optimizer == "brenth":
                self.constraints[0].V_init, info = scipy.optimize.brenth(
                    f=self.solve_scf_for_dW_by_dV,
                    a=self.constraints[0].V_brak[0],
                    b=self.constraints[0].V_brak[1],
                    maxiter=self.maxcscf
                )
            elif self.optimizer in ["BFGS"]:
                res = scipy.optimize.minimize(
                    method=self.optimizer,
                    fun=self.solve_scf_with_new_V,
                    x0=np.array(list(c.V_init for c in self.constraints)),
                    jac=True,
                    options={"maxiter": self.maxcscf}
                )
                for c, V in zip(self.constraints, res.x):
                    c.V_init = V
            else:
                raise ValueError

        except CDFTSCFConverged:
            print("*Constrained SCF converged!*\n")
        else:
            print("*Constrained SCF NOT converged after {} iterations!*".format(self.maxcscf))

    def solve_scf_with_new_V(self, Vs):
        """ Given V for all constraints, solve the KS problem."""
        self.itscf += 1

        print("Running optimizer: ", self.optimizer)

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
        start_dft = time.time()
        self.dft_driver.run_scf()
        self.dft_driver.get_rho_r()
        end_dft = time.time()

        for i, c in enumerate(self.constraints):
            c.update_N()
        self.sample.W = self.sample.Ed + self.sample.Ec - np.sum(c.N * c.V for c in self.constraints)

        # Print intermediate results
        print("=======================================")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("SCF iteration {}".format(self.itscf))
        print("  Ed (DFT energy) = {:.6f}".format(self.sample.Ed))
        print("  Ec (constraint energy) = {:.6f}".format(self.sample.Ec))
        print("  E (Ed + Ec) = {:.6f}".format(self.sample.Ed + self.sample.Ec))
        print("  W (free energy) = {:.6f}".format(self.sample.W))
        for i, c in enumerate(self.constraints):
            print("  > Constraint #{} (type = {}, N0 = {}, V = {:.6f}):".format(
                i, c.type, c.N0, c.V
            ))
            print("    N = {:.6f}".format(c.N))
            print("    dW/dV = N - N0 = {:.6f}".format(c.dW_by_dV))
        print("SCF time cycle:")
        timer(start_dft, end_dft)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
        print("Elapsed time: ")
        timer(self.start_time, time.time())

        if all(c.is_converged for c in self.constraints):
            raise CDFTSCFConverged

        # return the negative of W and dW/dV to be used by scipy minimizers
        return -self.sample.W, np.array(list(-c.dW_by_dV for c in self.constraints))

    def solve_scf_for_dW_by_dV(self, V):
        """ Wrapper function for solve_scf_with_new_V returning dW/dV."""
        return self.solve_scf_with_new_V([V])[1][0]

    def solve_opt(self):
        """ Relax the structure under constraint.

        A force from constraint potential is added to DFT force during optimization.
        """

        for istep in range(1, self.maxstep + 1):
            print("======================================")
            print("Geometry optimization step {}".format(istep))
            print("======================================")

            # run SCF to converge electronic structure
            self.solve_scf()

            # get DFT force
            self.dft_driver.get_force()

            # compute constraint force
            for c in self.constraints:
                c.update_Fc()
            self.sample.Fc = np.sum(c.Fc for c in self.constraints)

            self.sample.Fw = self.sample.Fd + self.sample.Fc
            Fwnorm = np.linalg.norm(self.sample.Fw, axis=1)
            maxforce = np.max(Fwnorm)
            imaxforce = np.argmax(Fwnorm)
            print("Maximum force = {:.6f} au, on {}th atom ({}).  Fw = {:.6f}, {:.6f}, {:.6f}".format(
                maxforce, imaxforce+1, self.sample.atoms[imaxforce].symbol, *self.sample.Fw[imaxforce]
            ))
            print("--")
            print("  Fd = {:.6f}, {:.6f}, {:.6f}; Fc = {:.6f}, {:.6f}, {:.6f}".format(
                *self.sample.Fd[imaxforce], *self.sample.Fc[imaxforce]
            ))
 
            # print all forces
            for i in np.arange(len(self.sample.atoms)):
                print("{}th atom ({})  Fd = {:.6f}, {:.6f}, {:.6f}; Fc = {:.6f}, {:.6f}, {:.6f}".format(
                 i + 1, self.sample.atoms[i].symbol, *self.sample.Fd[i], *self.sample.Fc[i]
                ))
            if maxforce < self.F_tol:
                print("\n**Constrained optimization converged!**\n")
                break

            # add constraint force to DFT force
            self.dft_driver.set_Fc()

            # run optimization
            self.dft_driver.run_opt()

            # parse updated coordinates
            self.dft_driver.get_structure()

            # update weights and constraint potentials
            for c in self.constraints:
                c.update_structure()

        else:
            print("\n**Constrained optimization NOT achieved after {} steps!**\n".format(self.maxstep))

    def copy(self):
        """ Generate a deepcopy of the current CDFTSolver instance."""
        return deepcopy(self)

    def restart(self, wfcfile, dft_energies):
        """ Restart from previous CDFT run.

        Args:
            wfcfile (str): name of wfc file
            dft_energies: [Ed, Ec] corresponding to Vc
        """
        print("=================== Restarting run =======================")
        try:
            self.dft_driver.restart_wfc(wfcfile, dft_energies)
  
        except:
            print(" Restart failed! ")
            e = sys.exc_info()
            print(e)
            print("Quitting DFT driver")
            self.dft_driver.exit()

    def get_Vc(self, Vc_file):
        """ Read Vc in cube format from Qbox; for restarting calculations for elcoupling"""
        sample = self.dft_driver.sample
        vspin = sample.vspin
        if vspin == 2:
            raise NotImplementedError("Spin-dependent Vext not implemented in Qbox yet.")
        n1, n2, n3 = sample.n1, sample.n2, sample.n3

        with io.open(Vc_file,'r',encoding='utf8') as f:
           data = f.readlines()
    
        # exclude header
        data = data[10:-2]
        data = ''.join(i for i in data).strip()
    
        # decode Vc.dat file
        data = data.encode("utf-8")
        data = base64.decodebytes(data)
    
        Vc = np.frombuffer(data, dtype=np.float64)
        Vc_tot = Vc.reshape(n3,n2,n1,vspin).T

        assert isinstance(Vc, np.ndarray) and Vc_tot.shape == (vspin, n1, n2, n3)
        self.Vc_tot = Vc_tot
        print(" uploaded Vc_tot")
