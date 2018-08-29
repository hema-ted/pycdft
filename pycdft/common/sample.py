from random import randint
from subprocess import Popen
import numpy as np
from ase import Atoms
from ase.io.cube import read_cube_data
from numpy.fft import *
from .pp import SG15PP
from .units import angstrom_to_bohr, bohr_to_angstrom
from .atom import Atom


class Sample(object):
    """ The physical system to be simulated.

    All physical quantities are in atomic unit.

    Attributes:
        R (np.ndarray, shape = [3, 3]): the real space lattice vectors of the system.
        G (np.ndarray, shape = [3, 3]): the reciprocal space lattice vectors of the system.
        omega (float): cell volume.
        atoms (list of Atoms): list of atoms.
        fragments (list of Fragments): list of fragments defined on the sample.
        rhopro_tot_r (np.ndarray, shape = [n1, n2, n3]): total promolecule density.
        nspin (int): number of spin channels (1 or 2).
        n1, n2, n3 (int): FFT grid for charge density, weight function and constraint potential.
        Edft (float): DFT total energy.
        W (float): free energy.
    """
    def __init__(self, ase_cell: Atoms, nspin: int, n1: int, n2: int, n3: int,
                 atomic_density_files: dict=None):

        # define cell
        self.R = ase_cell.get_cell() * angstrom_to_bohr
        self.G = 2 * np.pi * np.linalg.inv(self.R).T
        assert np.all(np.isclose(np.dot(self.R, self.G.T), 2 * np.pi * np.eye(3)))
        self.omega = np.linalg.det(self.R)
        self.atoms = list(Atom(cell=self, ase_atom=atom) for atom in ase_cell)
        self.natoms = len(self.atoms)
        self.species = sorted(set([atom.symbol for atom in self.atoms]))
        self.nspecies = len(self.species)

        # define fragments
        self.fragments = []

        # define nspin and FFT grid
        self.nspin = nspin
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.n123 = n1 * n2 * n3

        # define charge density and promolecule charge densities
        self.rho_r = None
        self.rhopro_tot_r = None

        # compute atomic density for all species
        self.rhoatom_g = {}
        if atomic_density_files is not None:
            for species, f in atomic_density_files.items():
                rho_r = read_cube_data(f)[0]
                assert rho_r.shape == (n1, n2, n3)
                rho_r1 = np.roll(rho_r, n1//2, axis=0)
                rho_r2 = np.roll(rho_r1, n2//2, axis=1)
                rho_r3 = np.roll(rho_r2, n3//2, axis=2)
                self.rhoatom_g[species] = fftn(rho_r3)
        else:
            raise NotImplementedError

        # define energies and forces
        self.Edft = None
        self.W = None
        self.Ftotal = None
        self.Fdft = None
        self.Fc = None

    def update_rhopro(self):
        """ Update promolecule densities. """
        self.rhopro_tot_r = np.zeros([self.n1, self.n2, self.n3], dtype=np.complex_)
        for f in self.fragments:
            f.rhopro_r = np.zeros([self.n1, self.n2, self.n3], dtype=np.complex_)

        # Compute promolecule densities in G space
        for atom in self.atoms:
            rhog = self.compute_rhoatom_g(atom)
            self.rhopro_tot_r += rhog
            for f in self.fragments:
                if atom in f:
                    f.rhopro_r += rhog

        # FT -> R space
        self.rhopro_tot_r = np.fft.ifftn(self.rhopro_tot_r).real
        for f in self.fragments:
            f.rhopro_r = np.fft.ifftn(f.rhopro_r).real

    def compute_rhoatom_g(self, atom):
        """ Compute charge density for an atom with specific coordinate in cell. """
        rhog0 = self.rhoatom_g[atom.symbol]

        G1, G2, G3 = self.G
        n1, n2, n3 = rhog0.shape
        freqlist1 = fftfreq(n1, d=1 / n1)
        freqlist2 = fftfreq(n2, d=1 / n2)
        freqlist3 = fftfreq(n3, d=1 / n3)

        r = atom.abs_coord
        G1, G2, G3 = G1 @ r, G2 @ r, G3 @ r
        look_up_table_h = np.exp(-1j * (freqlist1 * G1))
        look_up_table_k = np.exp(-1j * (freqlist2 * G2))
        look_up_table_l = np.exp(-1j * (freqlist3 * G3))
        eigr = np.kron(look_up_table_h, look_up_table_k)
        eigr = np.kron(eigr, look_up_table_l)
        eigr = np.reshape(eigr, newshape=(n1, n2, n3))

        return rhog0 * eigr

    @property
    def ase_cell(self):
        """ Get an ASE Atoms object of current cell."""
        ase_cell = Atoms(cell=self.R * bohr_to_angstrom)
        for atom in self.atoms:
            ase_cell.append(atom.ase_atom)
        return ase_cell

    def show(self):
        """ Visualize the structure by VESTA."""
        fname = "/tmp/cell{}.cif".format(randint(1000, 9999))
        self.save(fname)
        Popen(["vesta", fname])
        Popen(["sleep 120 && rm {}".format(fname)], shell=True)

    def save(self, fname):
        """ Save the structure to file."""
        self.ase_cell.write(fname)

    def export(self, fmt="qb", pseudos=None):
        """ Export the structure to various formats. """
        output = ""

        if fmt.lower() == "qb":
            output += "set cell {}\n".format(
                " ".join("{:06f}".format(self.R[i, j]) for i, j in np.ndindex(3, 3))
            )
            output += "\n"

            for species in self.species:
                output += "species {} {}\n".format(
                    species, pseudos[species] if pseudos else SG15PP[species]["xmlfile"]
                )
            output += "\n"

            for iatom, atom in enumerate(self.atoms):
                x, y, z = atom.abs_coord
                output += "atom {}{} {}  {:.8f}  {:.8f}  {:.8f}\n".format(
                    atom.symbol, iatom + 1, atom.symbol, x, y, z
                )

        else:
            raise NotImplementedError

        return output

    def nel(self, pseudos="SG15"):
        """ Compute # of electrons according to certain pseudopotential family."""
        if pseudos == "SG15":
            nel = 0
            for symbol in self.ase_cell.get_chemical_symbols():
                nel += SG15PP[symbol]["nel"]
        else:
            raise NotImplementedError
        return nel

    def get_maxforce(self):
        maxforce = 0
        for atom in self.atoms:
            f = np.linalg.norm(atom.Ftotal)
            if f > maxforce:
                maxforce = f
                maxforceatom = atom

        return maxforce, maxforceatom

    def __repr__(self):
        return "Cell \"{}\" natoms={}\nLattice constants:\n{}\nReciprocal lattice constants:\n{}\n".format(
            self.ase_cell.get_chemical_formula(), self.natoms, self.R, self.G
        )

    def __str__(self):
        return self.__repr__()
