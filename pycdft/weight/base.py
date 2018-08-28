from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
import numpy as np


class Weight(object):
    """ Weight.

    Attributes:
        sample (Sample): whole system.
        fragment (Fragment): the fragment of the system where the weight is computed.
        w (np.array, shape == [nspin, n1, n2, n3]): weight.
    """
    __metaclass__ = ABCMeta

    def __init__(self, fragment):
        self.sample = fragment.sample
        self.fragment = fragment
        fftgrid = self.sample.fftgrid
        n1, n2, n3 = fftgrid.n1, fftgrid.n2, fftgrid.n3
        self.w = np.zeros([self.sample.nspin, n1, n2, n3])

    @abstractmethod
    def update(self, i):
        """ Update weight w(r) with new structure. """
        pass

    @abstractmethod
    def compute_w_grad(self, atom):
        """ Compute derivative of weight w.r.t. coordinate of atom. """
        pass

    def read_w(self, f):
        self.w = np.load(f)

    def write_w(self, f):
        np.save(f, self.w)
