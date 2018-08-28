from __future__ import absolute_import, division, print_function

import os
from abc import ABCMeta, abstractmethod
from mpi4py import MPI

_mpiroot = MPI.COMM_WORLD.Get_rank() == 0


class WavefunctionParser(object):
    __metaclass__ = ABCMeta

    def __init__(self, path):
        self.path = path
        self.wfc = None
        self.parse()
        self.info()

    @abstractmethod
    def parse(self):
        """Scan current directory, construct wavefunction object"""
        if _mpiroot:
            print("\n{}: parsing directory \"{}\"...\n".format(
                self.__class__.__name__, self.path
            ))

    def info(self):
        if _mpiroot:
            wfc = self.wfc
            print("Scan finished. nspin = {}, nkpt = {}, nbnd = {}".format(
                wfc.nspin, wfc.nkpt, wfc.nbnd
            ))
            print("\nSystem Overview:")
            print("  Cell: ")
            print(wfc.cell.__repr__())
            if wfc.dgrid is not None:
                print("  Density FFT Grid: ")
                print(wfc.dgrid.__repr__())
