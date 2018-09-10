import os
import re
import shutil
import time
import base64
import numpy as np
from lxml import etree
from ase.io.cube import read_cube_data, write_cube
from pycdft.common.ft import FFTGrid
from pycdft.common.wfc import Wavefunction
from pycdft.dft_driver.base import DFTDriver


class QboxLockfileError(Exception):
    pass


class QboxDriver(DFTDriver):
    """ DFT driver.

    Extra attributes:
        init_cmd (str): initialization command for Qbox.
        scf_cmd (str): command for running constrained SCF.
        opt_cmd (str): command for running geometry optimization.
    """

    sleep_seconds = 2
    max_sleep_seconds = 3600 * 6

    input_file = "qb_cdft.in"
    lock_file = "{}.lock".format(input_file)
    output_file = "qb_cdft.out"
    Vc_file = "Vc.cube"
    rhor_file = "rhor.cube"
    wfc_file = "wfc.xml"
    wfc_cmd = "save {}".format(wfc_file)

    def __init__(self, sample, init_cmd, scf_cmd, opt_cmd=None):
        super(QboxDriver, self).__init__(sample)
        self.init_cmd = init_cmd
        self.scf_cmd = scf_cmd
        self.opt_cmd = opt_cmd

        self.scf_xml = None
        self.opt_xml = None

    def reset(self, output_path):
        self.istep = self.icscf = 1
        print("QboxDriver: setting output path to {}...".format(output_path))
        self.output_path = output_path
        print("QboxDriver: waiting for Qbox to start...")
        self.wait_for_lock_file()
        print("QboxDriver: initializing Qbox...")
        self.run_cmd(self.init_cmd)

    def wait_for_lock_file(self):
        """ Wait for Qbox lock file to appear."""
        nsec = 0
        while not os.path.isfile(self.lock_file):
            time.sleep(self.sleep_seconds)
            nsec += self.sleep_seconds
            if nsec > self.max_sleep_seconds:
                raise QboxLockfileError

    def run_cmd(self, cmd):
        """ Order Qbox to run given command. """
        open(self.input_file, "w").write(cmd + "\n")
        os.remove(self.lock_file)
        self.wait_for_lock_file()

    def set_Vc(self, Vc):
        """ Write Vc in cube format, then send set vext command to Qbox."""
        nspin = self.sample.nspin
        if nspin == 2:
            raise NotImplementedError("Spin-dependent Vext not implemented in Qbox yet.")
        n1, n2, n3 = self.sample.n1, self.sample.n2, self.sample.n3
        assert isinstance(Vc, np.ndarray) and Vc.shape == (nspin, n1, n2, n3)

        ase_cell = self.sample.ase_cell
        write_cube(open(self.Vc_file, "w"), atoms=ase_cell, data=Vc[0])

        self.run_cmd("set vext {}".format(self.Vc_file))

    def copy_output(self):
        """ Copy Qbox output file to self.output_path. """
        shutil.copyfile(self.output_file, "{}/step{}-scf{}.out".format(
            self.output_path, self.istep, self.icscf
        ))

    def run_scf(self):
        """ Run SCF calculation in Qbox."""
        self.run_cmd(self.scf_cmd)
        self.copy_output()
        self.icscf += 1
        self.scf_xml = etree.parse(self.output_file).getroot()
        self.sample.Edft_total = float(self.scf_xml.findall("iteration/etotal")[-1].text)
        self.sample.Edft_bare = self.sample.Edft_total - float(self.scf_xml.findall("iteration/eext")[-1].text)

    def run_opt(self):
        """ Run geometry optimization in Qbox."""
        self.run_cmd(self.opt_cmd)
        self.copy_output()
        self.opt_xml = etree.parse(self.output_file).getroot()

    def get_rho_r(self):
        """ Implement abstract fetch_rhor method for Qbox.

        Send plot charge density commands to Qbox, then parse charge density.
        """
        nspin = self.sample.nspin
        n1, n2, n3 = self.sample.n1, self.sample.n2, self.sample.n3
        self.sample.rho_r = np.zeros([nspin, n1, n2, n3])

        for ispin in range(nspin):
            self.run_cmd(cmd="plot -density {} {}".format(
                "-spin {}".format(ispin + 1) if nspin == 2 else "",
                self.rhor_file
            ))

            rhor_raw = read_cube_data(self.rhor_file)[0]
            assert rhor_raw.shape == (n1, n2, n3)

            rhor1 = np.roll(rhor_raw, n1//2, axis=0)
            rhor2 = np.roll(rhor1, n2//2, axis=1)
            rhor3 = np.roll(rhor2, n3//2, axis=2)
            self.sample.rho_r[ispin] = rhor3

    def get_force(self):
        """ Implement abstract fetch_force method for Qbox."""
        # parse from self.scf_xml
        Fdft = np.zeros([self.sample.natoms, 3])

        for i in self.scf_xml.findall('iteration')[-1:]:
            for atoms in i.findall('atomset'):
                for atom in atoms.findall('atom'):

                    m = re.match(r"([a-zA-Z]+)([0-9]+)", atom.attrib['name'])
                    symbol, index = m.group(1), int(m.group(2))
                    assert self.sample.atoms[index-1].symbol == symbol

                    a = atom.findall('force')
                    f = np.array(a[0].text.split()).astype(np.float)

                    Fdft[index-1] = f

        return Fdft

    def set_Fc(self, Fc):
        """ Implement abstract set_force method for Qbox."""
        for i in range(self.sample.natoms):
            symbol = self.sample.atoms[i].symbol
            self.run_cmd(cmd="extforce delete f{}{}".format(symbol, i + 1))

        for i in range(self.sample.natoms):
            symbol = self.sample.atoms[i].symbol
            qb_sym = symbol + str(i+1)
            self.run_cmd(cmd="extforce define f{} {} {:06f} {:06f} {:06f}".format(qb_sym, qb_sym, Fc[i][0], Fc[i][1], Fc[i][2]))

    def get_structure(self):
        """ Implement abstract fetch_structure method for Qbox."""
        # parse from self.scf_xml
        for i in self.opt_xml.findall('iteration')[-1:]:
            for atoms in i.findall('atomset'):
                for atom in atoms.findall('atom'):
                    m = re.match(r"([a-zA-Z]+)([0-9]+)", atom.attrib['name'])
                    symbol, index = m.group(1), int(m.group(2))
                    assert self.sample.atoms[index - 1].symbol == symbol

                    a = atom.findall('position')
                    p = np.array(a[0].text.split()).astype(np.float)

                    self.sample.atoms[index - 1].abs_coord = p

    def clean(self):
        """ Clean qb_cdft.in qb_cdft.out qb_cdft.in.lock"""
        self.run_cmd("save wf.xml")

        if os.path.exists(self.input_file):
            os.remove(self.input_file)

        if os.path.exists(self.lock_file):
            os.remove(self.lock_file)

        _output_file = self.input_file.split('.')[0] + '.out'
        if os.path.exists(_output_file):
            os.remove(_output_file)

    def get_wfc(self):
        """ Parse wavefunction from Qbox."""

        self.run_cmd(self.wfc_cmd)
        wfcfile = self.wfc_file

        iterxml = etree.iterparse(wfcfile, huge_tree=True, events=("start", "end"))

        nkpt = 1
        ikpt = 0  # k points are not supported yet

        for event, leaf in iterxml:
            if event == "start" and leaf.tag == "wavefunction":
                nspin = int(leaf.attrib["nspin"])
                assert nspin >= self.sample.nspin
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

        wgrid = FFTGrid(n1, n2, n3)

        occs_ = np.zeros((nspin, nkpt, max(len(occs[ispin, ikpt])
                                           for ispin, ikpt in np.ndindex(nspin, nkpt))))
        for ispin, ikpt in np.ndindex(nspin, nkpt):
            occs_[ispin, ikpt, 0:len(occs[ispin, ikpt])] = occs[ispin, ikpt]

        wfc = Wavefunction(sample=self.sample, wgrid=wgrid, dgrid=wgrid,
                           nspin=nspin, nkpt=nkpt, nbnd=nbnd, occ=occs_, gamma=True)

        iterxml = etree.iterparse(wfcfile, huge_tree=True, events=("start", "end"))

        ispin = 0
        ikpt = 0
        for event, leaf in iterxml:

            if event == "start" and leaf.tag == "slater_determinant":
                if wfc.nspin == 2 and leaf.attrib["spin"] == "down":
                    ispin = 1
                ibnd = 0

            if event == "end" and leaf.tag == "grid_function":
                if leaf.attrib["encoding"] == "base64":
                    psir = np.frombuffer(
                        base64.b64decode(leaf.text), dtype=np.float64
                    ).reshape(wfc.wgrid.n3, wfc.wgrid.n2, wfc.wgrid.n1).T
                elif leaf.attrib["encoding"] == "text":
                    psir = np.fromstring(leaf.text, sep=" ", dtype=np.float64).reshape(
                        wfc.wgrid.n3, wfc.wgrid.n2, wfc.wgrid.n1
                    ).T
                else:
                    raise ValueError
                wfc.psi_r[ispin, ikpt, ibnd] = psir
                ibnd += 1
                leaf.clear()

            if event == "start" and leaf.tag == "wavefunction_velocity":
                break

        self.sample.wfc = wfc
