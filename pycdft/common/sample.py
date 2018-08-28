import numpy as np
from .cell import Cell
from .ft import FFTGrid


class Sample(object):
    """ The physical system to be simulated.

    Attributes:
        cell (Cell): the cell of the system.
        nspin (int): number of spin channels (1 or 2).
    """
    def __init__(self, cell, fftgrid, nspin):
        assert isinstance(cell, Cell)
        assert isinstance(fftgrid, FFTGrid)
        assert isinstance(nspin, int)

        self.cell = cell
        self.fftgrid = fftgrid
        self.nspin = nspin

        self.rhor = np.zeros([nspin, fftgrid.n1, fftgrid.n2, fftgrid.n3])
        self.Etotal = None
