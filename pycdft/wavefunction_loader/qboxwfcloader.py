from __future__ import absolute_import, division, print_function

import base64
import os
from glob import glob
import numpy as np
from lxml import etree
from mpi4py import MPI
from ase import Atoms, Atom
from ..common.units import bohr_to_angstrom
from ..common.ft import FFTGrid
from ..common.wfc import Wavefunction
from .text import parse_many_values
from .wfcloader import WavefunctionParser
from ..common.cell import Cell

_mpiroot = MPI.COMM_WORLD.Get_rank() == 0


class QboxWavefunctionParser(WavefunctionParser):
    def __init__(self, path="."):
        super(QboxWavefunctionParser, self).__init__(path)

    def parse(self):
        super(QboxWavefunctionParser, self).parse()

        if os.path.isfile(self.path):
            self.xmlfile = self.path
        else:
            xmllist = sorted(glob("{}/*xml".format(self.path)), key=lambda f: os.path.getsize(f))
            if len(xmllist) == 0:
                raise IOError("No xml file found in current directory: {}".format(os.getcwd()))
            elif len(xmllist) == 1:
                self.xmlfile = xmllist[0]
            else:
                self.xmlfile = xmllist[-1]
                if _mpiroot:
                    print("More than one xml files found: {}".format(xmllist))
                    print("Assume wavefunction is in the largest xml file: {} ({} MB)".format(
                        self.xmlfile, os.path.getsize(self.xmlfile) / 1024 ** 2
                    ))

        if _mpiroot:
            print("Reading wavefunction from file {}".format(self.xmlfile))

        iterxml = etree.iterparse(self.xmlfile, huge_tree=True, events=("start", "end"))

        nkpt = 1
        ikpt = 0  # k points are not supported yet

        for event, leaf in iterxml:
            if event == "end" and leaf.tag == "unit_cell":
                R1 = np.fromstring(leaf.attrib["a"], sep=" ", dtype=np.float_) * bohr_to_angstrom
                R2 = np.fromstring(leaf.attrib["b"], sep=" ", dtype=np.float_) * bohr_to_angstrom
                R3 = np.fromstring(leaf.attrib["c"], sep=" ", dtype=np.float_) * bohr_to_angstrom
                lattice = np.array([R1, R2, R3])
                ase_cell = Atoms(cell=lattice, pbc=True)

            if event == "end" and leaf.tag == "atom":
                species = leaf.attrib["species"]
                position = np.array(parse_many_values(3, float, leaf.find("position").text))
                ase_cell.append(Atom(symbol=species, position=position * bohr_to_angstrom))

            if event == "start" and leaf.tag == "wavefunction":
                nspin = int(leaf.attrib["nspin"])
                nbnd = np.zeros((nspin, nkpt))
                occs = np.zeros((nspin, nkpt), dtype=object)

            if event == "end" and leaf.tag == "grid":
                n1, n2, n3 = int(leaf.attrib["nx"]), int(leaf.attrib["ny"]), int(leaf.attrib["nz"])

            if event == "start" and leaf.tag == "slater_determinant":
                ispin = 0
                if nspin == 2 and leaf.attrib["spin"] == "down":
                    ispin = 1

            if event == "end" and leaf.tag == "density_matrix":
                occ = np.fromstring(leaf.text, sep=" ", dtype=np.float_)
                nbnd[ispin, ikpt] = len(occ)
                occs[ispin, ikpt] = occ

            if event == "end" and leaf.tag == "grid_function":
                leaf.clear()

            if event == "start" and leaf.tag == "wavefunction_velocity":
                break

        cell = Cell(ase_cell)
        wgrid = FFTGrid(n1, n2, n3)

        occs_ = np.zeros((nspin, nkpt, max(len(occs[ispin, ikpt])
                                           for ispin, ikpt in np.ndindex(nspin, nkpt))))
        for ispin, ikpt in np.ndindex(nspin, nkpt):
            occs_[ispin, ikpt, 0:len(occs[ispin, ikpt])] = occs[ispin, ikpt]

        self.wfc = Wavefunction(cell=cell, wgrid=wgrid, dgrid=wgrid,
                                nspin=nspin, nkpt=nkpt, nbnd=nbnd, occ=occs_,
                                gamma=True, gvecs=None)

        iterxml = etree.iterparse(self.xmlfile, huge_tree=True, events=("start", "end"))

        for event, leaf in iterxml:
            ispin = 0
            ikpt = 0
            if event == "start" and leaf.tag == "slater_determinant":
                if self.wfc.nspin == 2 and leaf.attrib["spin"] == "down":
                    ispin = 1
                ibnd = 0

            if event == "end" and leaf.tag == "grid_function":
                if leaf.attrib["encoding"] == "base64":
                    psir = np.frombuffer(
                        base64.b64decode(leaf.text), dtype=np.float64
                    ).reshape(self.wfc.wgrid.n3, self.wfc.wgrid.n2, self.wfc.wgrid.n1).T
                elif leaf.attrib["encoding"] == "text":
                    psir = np.fromstring(leaf.text, sep=" ", dtype=np.float64).reshape(
                        self.wfc.wgrid.n3, self.wfc.wgrid.n2, self.wfc.wgrid.n1
                    ).T
                else:
                    raise ValueError
                self.wfc.psir[ispin, ikpt, ibnd] = psir
                ibnd += 1
                leaf.clear()

            if event == "start" and leaf.tag == "wavefunction_velocity":
                break
