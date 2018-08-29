import os
import re
import shutil
import time
import base64
import numpy as np
from lxml import etree
from ase.io.cube import read_cube_data, write_cube
from .base import DFTDriver
from ..common.ft import FFTGrid
from ..common.wfc import Wavefunction


class QboxLockfileError(Exception):
    pass


class QboxDriver(DFTDriver):

    _sleep_seconds = 2
    _max_sleep_seconds = 3600 * 6

    _Vc_file = "Vc.cube"
    _rhor_file = "rhor.cube"
    _wfc_file = "wfc.xml"
    _wfc_cmd = "save {}".format(_wfc_file)

    _archive_folder = 'qbox_outputs'

    def __init__(self, sample, init_cmd, scf_cmd, opt_cmd, input_file="qb_cdft.in"):
        """Initialize QboxDriver.

        Control commands for SCF calculations (e.g. "run 0 100 5") needs to
        be specified by scf_cmd.
        """
        super(QboxDriver, self).__init__(sample)
        self._opt_cmd = opt_cmd
        self._init_cmd = init_cmd
        self._scf_cmd = scf_cmd
        self._input_file = input_file
        self._output_file = self._input_file.split('.')[0] + '.out'
        self._lock_file = "{}.lock".format(input_file)

        self.iter = 0

        self.scf_xml = None
        self.opt_xml = None

        if os.path.exists(self._archive_folder):
            shutil.rmtree(self._archive_folder)
            os.makedirs(self._archive_folder)
        else:
            os.makedirs(self._archive_folder)

        # Initialize Qbox
        print("QboxDriver: waiting for Qbox to start...")
        self.wait_for_lockfile()
        print("QboxDriver: initializing Qbox...")
        self.run_cmd(self._init_cmd)

    def wait_for_lockfile(self):
        """ Wait for Qbox lock file to appear."""
        nsec = 0
        while not os.path.isfile(self._lock_file):
            time.sleep(self._sleep_seconds)
            nsec += self._sleep_seconds
            if nsec > self._max_sleep_seconds:
                raise QboxLockfileError

    def run_cmd(self, cmd):
        """ Let Qbox run given command. """
        open(self._input_file, "w").write(cmd + "\n")
        os.remove(self._lock_file)
        self.wait_for_lockfile()

    def set_Vc(self, Vc):
        """ Write Vc in cube format, then send set vext command to Qbox."""
        nspin = self.sample.nspin
        if nspin == 2:
            raise NotImplementedError("Spin-dependent Vext not implemented in Qbox yet.")
        fftgrid = self.sample.fftgrid
        n1, n2, n3 = fftgrid.n1, fftgrid.n2, fftgrid.n3
        assert isinstance(Vc, np.ndarray) and Vc.shape == (nspin, n1, n2, n3)

        ase_cell = self.sample.ase_cell
        write_cube(open(self._Vc_file, "w"), atoms=ase_cell, data=Vc[0])

        self.run_cmd("set vext {}".format(self._Vc_file))

    def run_scf(self):
        """ Run SCF calculation in Qbox."""
        self.run_cmd(self._scf_cmd)
        self.iter += 1
        shutil.copyfile(self._output_file,
                        "{}/iter{}.out".format(self._archive_folder, self.iter))
        self.scf_xml = etree.parse(self._output_file).getroot()
        for i in self.scf_xml.findall('iteration'):
            self.sample.Etotal = float((i.findall('etotal'))[0].text)

    def run_opt(self):
        """ Run geometry relaxation in Qbox."""
        self.run_cmd(self._opt_cmd)
        shutil.copyfile(self._output_file,
                        "{}/iter{}_opt.out".format(self._archive_folder, self.iter))
        self.opt_xml = etree.parse(self._output_file).getroot()

    def get_rho_r(self):
        """ Implement abstract fetch_rhor method for Qbox.

        Send plot charge density commands to Qbox, then parse charge density.
        """
        nspin = self.sample.nspin

        for ispin in range(nspin):
            self.run_cmd(cmd="plot -density {} {}".format(
                "-spin {}".format(ispin + 1) if nspin == 2 else "",
                self._rhor_file
            ))

            rhor_raw = read_cube_data(self._rhor_file)[0]
            n1, n2, n3 = rhor_raw.shape

            rhor1 = np.roll(rhor_raw, n1//2, axis=0)
            rhor2 = np.roll(rhor1, n2//2, axis=1)
            rhor3 = np.roll(rhor2, n3//2, axis=2)
            self.sample.rhor[ispin] = rhor3

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

        if os.path.exists(self._input_file):
            os.remove(self._input_file)

        if os.path.exists(self._lock_file):
            os.remove(self._lock_file)

        _output_file = self._input_file.split('.')[0] + '.out'
        if os.path.exists(_output_file):
            os.remove(_output_file)

    def get_wfc(self):
        """ Parse wavefunction from Qbox."""

        self.run_cmd(self._wfc_cmd)
        wfcfile = self._wfc_file

        iterxml = etree.iterparse(wfcfile, huge_tree=True, events=("start", "end"))

        nkpt = 1
        ikpt = 0  # k points are not supported yet

        for event, leaf in iterxml:
            if event == "start" and leaf.tag == "wavefunction":
                nspin = int(leaf.attrib["nspin"])
                assert nspin == self.sample.nspin
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
                           nspin=nspin, nkpt=nkpt, nbnd=nbnd, occ=occs_,
                           gamma=True, gvecs=None)

        for event, leaf in iterxml:
            ispin = 0
            ikpt = 0
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
                wfc.psir[ispin, ikpt, ibnd] = psir
                ibnd += 1
                leaf.clear()

            if event == "start" and leaf.tag == "wavefunction_velocity":
                break

        return wfc
