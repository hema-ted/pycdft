import os
import re
import shutil
import time
import base64
import numpy as np
from lxml import etree
from ase.io.cube import read_cube_data
from pycdft.common.ft import FFTGrid
from pycdft.common.wfc import Wavefunction
from pycdft.dft_driver.base import DFTDriver


class QboxLockfileError(Exception):
    pass


class QboxDriver(DFTDriver):
    """ DFT driver for Qbox (v 1.69.0 +, post-XML conversion)

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
    Vc_file = "Vc.dat"
    rhor_file = "rhor.cube"
    wfc_file = "wfc.xml"
    wfc_cmd = "save {}".format(wfc_file)

    f3d_template = """\
<?xml version="1.0" encoding="UTF-8"?>
<fpmd:function3d xmlns:fpmd="http://www.quantum-simulation.org/ns/fpmd/fpmd-1.0"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 xsi:schemaLocation="http://www.quantum-simulation.org/ns/fpmd/fpmd-1.0 function3d.xsd"
 name="delta_v">
<domain a="{R00:.5f}  {R01:.5f}  {R02:.5f}"
        b="{R10:.5f}  {R11:.5f}  {R12:.5f}"
        c="{R20:.5f}  {R21:.5f}  {R22:.5f}"/>
<grid nx="{nx}" ny="{ny}" nz="{nz}"/>
<grid_function type="double" nx="{nx}" ny="{ny}" nz="{nz}" encoding="base64">
{data}
</grid_function>
</fpmd:function3d>
"""

    def __init__(self, sample, init_cmd, scf_cmd, opt_cmd="run 1 0 0"):
        super(QboxDriver, self).__init__(sample)
        self.init_cmd = init_cmd
        self.scf_cmd = scf_cmd
        self.opt_cmd = opt_cmd

        self.scf_xml = None
        self.opt_xml = None

    def reset(self, output_path):
        self.istep = self.icscf = 0
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
        time.sleep(self.sleep_seconds)

    def run_cmd(self, cmd):
        """ Order Qbox to run given command. """
        open(self.input_file, "w").write(cmd + "\n")
        os.remove(self.lock_file)
        self.wait_for_lock_file()

    def set_Vc(self, Vc):
        """ Write Vc in cube format, then send set vext command to Qbox."""
        vspin = self.sample.vspin
        if vspin == 2:
            raise NotImplementedError("Spin-dependent Vext not implemented in Qbox yet.")
        n1, n2, n3 = self.sample.n1, self.sample.n2, self.sample.n3
        assert isinstance(Vc, np.ndarray) and Vc.shape == (vspin, n1, n2, n3)

        data = base64.encodebytes(Vc.T.tobytes()).strip()  # reverse order of x, y, z direction
        R = self.sample.R
        f = self.f3d_template.format(
            R00=R[0, 0], R01=R[0, 1], R02=R[0, 2],
            R10=R[1, 0], R11=R[1, 1], R12=R[1, 2],
            R20=R[2, 0], R21=R[2, 1], R22=R[2, 2],
            nx=n1, ny=n2, nz=n3,
            data=data.decode("utf-8")
        )
        open(self.Vc_file, "w").write(f)

        self.run_cmd("set vext {}".format(self.Vc_file))

    def copy_output(self):
        """ Copy Qbox output file to self.output_path. """
        shutil.copyfile(self.output_file, "{}/step{}-scf{}.out".format(
            self.output_path, self.istep, self.icscf
        ))

    def run_scf(self):
        """ Run SCF calculation in Qbox."""
        self.run_cmd(self.scf_cmd)
        self.scf_xml = etree.parse(self.output_file).getroot()
        etotal = float(self.scf_xml.findall("iteration/etotal")[-1].text)
        eext = float(self.scf_xml.findall("iteration/eext")[-1].text)
        self.sample.Ec = eext
        self.sample.Ed = etotal - eext

        self.icscf += 1
        self.copy_output()

    def run_opt(self):
        """ Run geometry optimization in Qbox."""
        self.run_cmd(self.opt_cmd)
        self.opt_xml = etree.parse(self.output_file).getroot()

        self.istep += 1
        self.icscf = 0
        self.copy_output()

    def get_rho_r(self):
        """ Implement abstract fetch_rhor method for Qbox.

        Send plot charge density commands to Qbox, then parse charge density.
        """
        vspin = self.sample.vspin
        n1, n2, n3 = self.sample.n1, self.sample.n2, self.sample.n3
        self.sample.rho_r = np.zeros([vspin, n1, n2, n3])

        for ispin in range(vspin):
            # Qbox generates charge density
            self.run_cmd(cmd="plot -density {} {}".format(
                "-spin {}".format(ispin + 1) if vspin == 2 else "",
                self.rhor_file
            ))

            rhor_raw = read_cube_data(self.rhor_file)[0]
            assert rhor_raw.shape == (n1, n2, n3)

            # 
            rhor1 = np.roll(rhor_raw, n1//2, axis=0)
            rhor2 = np.roll(rhor1, n2//2, axis=1)
            rhor3 = np.roll(rhor2, n3//2, axis=2)
            self.sample.rho_r[ispin] = rhor3

    def get_force(self):
        """ Implement abstract fetch_force method for Qbox."""
        # parse from self.scf_xml
        self.sample.Fd = np.zeros([self.sample.natoms, 3])

        for i in self.scf_xml.findall('iteration')[-1:]:
            for atoms in i.findall('atomset'):
                for atom in atoms.findall('atom'):

                    m = re.match(r"([a-zA-Z]+)([0-9]+)", atom.attrib['name'])
                    symbol, index = m.group(1), int(m.group(2))
                    assert self.sample.atoms[index-1].symbol == symbol

                    a = atom.findall('force')
                    f = np.array(a[0].text.split()).astype(np.float)

                    self.sample.Fd[index-1] = f

    def set_Fc(self):
        """ Implement abstract set_force method for Qbox."""
        cmd = ""
        for i in range(self.sample.natoms):
            symbol = self.sample.atoms[i].symbol
            cmd += "extforce delete f{}{}\n".format(symbol, i + 1)

        Fc = self.sample.Fc
        for i in range(self.sample.natoms):
            symbol = self.sample.atoms[i].symbol
            qb_sym = symbol + str(i+1)
            cmd += "extforce define f{} {} {:06f} {:06f} {:06f}\n".format(
                qb_sym, qb_sym, Fc[i][0], Fc[i][1], Fc[i][2]
            )
        self.run_cmd(cmd)

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

    def parse_wfc_from_file(self, wfcfile):
        """ Parse wavefunction from an XML file."""
        iterxml = etree.iterparse(wfcfile, huge_tree=True, events=("start", "end"))

        nkpt = 1
        ikpt = 0  # k points are not supported yet

        for event, leaf in iterxml:
            if event == "start" and leaf.tag == "wavefunction":
                nspin = int(leaf.attrib["nspin"])
                assert nspin >= self.sample.vspin
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
        return wfc

    def get_wfc(self):
        """ Parse wavefunction from Qbox."""
        self.run_cmd(self.wfc_cmd)
        self.sample.wfc = self.parse_wfc_from_file(self.wfc_file)

    def restart_wfc(self, wfcfile, dft_energies):
        """ Load DFT energies and wavefunction."""
        
        # set up total energies used in calculation of electronic coupling
        self.sample.Ed = dft_energies[0]
        self.sample.Ec = dft_energies[1]
 
        self.sample.wfc = self.parse_wfc_from_file(wfcfile)
        print("QboxDriver: loaded wfc from file for restart.")

    def exit(self):
        """ Quit DFT driver """
        open(self.input_file, "w").write("quit" + "\n")
        os.remove(self.lock_file)
        print("Qbox session ended")
