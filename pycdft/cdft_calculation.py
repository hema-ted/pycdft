from __future__ import absolute_import, division, print_function

from sys import exit
import numpy as np
import os
import shutil

from .common import Sample
from .constraint import Constraint
from .dft_driver import DFTDriver


class CDFTCalculation:
    """ Constrained DFT calculation.

    Attributes:
        job (str): "scf" or "relax".
        sample (Sample): the sample on which cDFT calculation is performed.
        constraints (list of Constraint instances): a list of constraints on the system.
        dft_driver (DFTDriver): the interface to DFT code (e.g. Qbox or PW).
        maxiter (int): maximum number of CDFT iterations.
        maxstep (int): maximum relaxation steps.
        Ftol (float): force threshold for relaxation.
        Vc_tot (float array, shape == [nspin, n1, n2, n3]): total constraint potential
            as a sum of all constraints defined on all fragments.
    """

    def __init__(self, job, sample, constraints, dft_driver,
                 maxiter=1000, maxstep=100, Ftol=1.0E-2):
        assert job in ["scf", "relax"]
        assert isinstance(sample, Sample)
        assert all(isinstance(c, Constraint) for c in constraints)
        assert isinstance(dft_driver, DFTDriver)

        self.job = job
        self.sample = sample
        self.constraints = constraints
        self.dft_driver = dft_driver
        self.maxiter = maxiter
        self.maxstep = maxstep
        self.Ftol = Ftol
        self.W = None
        self.Ftot = None
        self.Fc = None
        self.Fdft = None


        with open('forces.xyz', 'w') as f:
            f.write("")

        with open('fc.xyz', 'w') as f:
            f.write("")

        with open('fdft.xyz', 'w') as f:
            f.write("")

        self._output_file = None
        self._archive_folder = 'outputs'
        if os.path.exists(self._archive_folder):
            shutil.rmtree(self._archive_folder)
            os.makedirs(self._archive_folder)
        else:
            os.makedirs(self._archive_folder)


        nspin = sample.nspin
        n1, n2, n3 = sample.fftgrid.n1, sample.fftgrid.n2, sample.fftgrid.n3
        self.Vc_tot = np.zeros([nspin, n1, n2, n3])


    def solve(self):
        """ Solve CDFT SCF or relax problem."""
        if self.job == "scf":
            self.solve_scf()
        elif self.job == "relax":
            self.solve_relax()
        else:
            raise ValueError

    def solve_relax(self):
        """ Relax the structure with CDFT.

        A force from constraint potential is added to DFT force during relaxation.
        """
        with open('vmd.xyz', 'w') as f:
            f.write("")
        self.record_vmd()


        for istep in range(1, self.maxstep + 1):
            # run SCF to converge electronic structure
            self.solve_scf()

            for i in range(1, len(self.constraints) + 1):
                self._output_file = 'c' + str(i)
                shutil.copyfile(self._output_file,
                                "{}/{}_{}.out".format(self._archive_folder, self._output_file, istep))

            if np.all(np.absolute(self.Ftot) < self.Ftol):
                print("CDFTCalculation: force convergence achieved!")
                break

            # add constraint force to DFT force
            self.dft_driver.set_force(self.Fc)

            # run optimization
            self.dft_driver.run_opt()

            # parse updated coordinates
            self.dft_driver.fetch_structure()

            # update weights and constraint potentials
            for c in self.constraints:
                c.update_R(istep)

            self.record_vmd()

            print("================================")
            print("Structure updated")
            print("================================")


        else:
            print("CDFTCalculation: relaxation NOT achieved after {} steps.".format(self.maxstep))
            #exit()

        self.record_force()


    def solve_scf(self):
        """ Iteratively solve the CDFT problem.

        An outer loop (implemented below) is performed to maximize the free energy w.r.t.
        Lagrangian multipliers for all constrains, an inner loop (casted to a KS problem and
        outsourced to a DFT code) is performed to minimize the free energy w.r.t. charge density.
        """
        for c in self.constraints:
            c.optimizer.setup()

        for i in range(1, len(self.constraints)+1):
            with open("c" + str(i), 'w') as file:
                file.write("")


        for iiter in range(1, self.maxiter + 1):

            # Compute the total constraint potential Vc.
            self.Vc_tot = np.sum(c.Vc for c in self.constraints)
            #self.Vc_tot = np.sum(c.V * c.weight.w for c in self.constraints)

            # Impose the constraint potential Vc to DFT code.
            self.dft_driver.set_Vc(self.Vc_tot)
            # Let DFT code perform SCF calculation under the constraint potential Vc.
            # dft driver run_scf command should read etotal and force
            self.dft_driver.run_scf()

            self.W = self.sample.Etotal - np.sum(c.V * c.N for c in self.constraints)

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Iter {}:".format(iiter))
            for i, c in enumerate(self.constraints, 1):
                print("V = {} (constraint {})".format(c.V, i))
            print("Etotal = {}".format(self.sample.Etotal))
            print("W = {}".format(self.W))

            for i, c in enumerate(self.constraints, 1):
                with open("c" + str(i), 'a') as file:
                    file.write("{:06f} {:06f}\n".format(c.V, self.W))

            # Parse the resulting charge density.
            self.dft_driver.fetch_rhor()

            # Update the Lagrangian multipliers for all constraints.
            for i, c in enumerate(self.constraints, 1):
                c.update_V(self.W)

            if all(c.is_converged for c in self.constraints):
                # Check if convergence achieved
                print("CDFTCalculation: convergence achieved!")
                self.record_force()
                return

        else:
            print("CDFTCalculation: convergence NOT achieved after {} iterations.".format(self.maxiter))
            self.record_force()
            #exit()


    def compute_Fc(self):
        Fc = np.zeros([self.sample.cell.natoms, 3])
        for c in self.constraints:
            Fc += c.compute_Fc()

        return Fc


    def record_force(self):
        # read DFT force
        self.Fdft = self.dft_driver.fetch_force()
        # compute constraint force
        self.Fc = self.compute_Fc()
        # compute total force and check convergence
        self.Ftot = self.Fdft + self.Fc


        for i in range(self.sample.cell.natoms):
            self.sample.cell.atoms[i].force = self.Ftot[i]

        with open('forces.xyz', 'a') as f:
            f.write(self.sample.cell.export('forces'))



        for i in range(self.sample.cell.natoms):
            self.sample.cell.atoms[i].fc = self.Fc[i]

        with open('fc.xyz', 'a') as f:
            f.write(self.sample.cell.export('fc'))



        for i in range(self.sample.cell.natoms):
            self.sample.cell.atoms[i].fdft = self.Fdft[i]

        with open('fdft.xyz', 'a') as f:
            f.write(self.sample.cell.export('fdft'))


    def record_vmd(self):
        with open('vmd.xyz', 'a') as f:
            f.write(self.sample.cell.export('vmd'))
