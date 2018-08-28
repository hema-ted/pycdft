from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.linalg import det, inv
from ..common.wfc import Wavefunction

class ElectronicCouplingBase(object):
    """

    """
    __metaclass__ = ABCMeta

    def __init__(self, file_path_to_a, file_path_to_b, weight):
        """
        :param file_path_to_a: (string) absolute/ relative path to the folder of state a
        :param file_path_to_b: (string) absolute/ relative path to the folder of state b
        :param weight:         (np.array) the weight in the form (np.array, shape == [nspin, n1, n2, n3])
        """
        self.path_to_a = file_path_to_a
        self.path_to_b = file_path_to_b
        self.weight = weight
        self.wfc_a = None
        self.wfc_b = None

    @abstractmethod
    def loadwfc(self):
        pass

    def __get_all_wavefunction(self, wfc_a, wfc_b):
        """
        :param wfc_a: (Wavefunction) wavefunction of a
        :param wfc_b: (Wavefunction) wavefunction of b
        :return: two 1-d np.array of Wavefunction object
        """
        assert len(wfc_a._occ) == len(wfc_b._occ)
        assert len(wfc_a._occ[0]) == len(wfc_b._occ[0])
        assert len(wfc_a._occ[0][0]) == len(wfc_b._occ[0][0])
        assert wfc_a._psir[0, 0, 0].shape == wfc_b._psir[0, 0, 0]
        nspin, nkpoint, nband = len(wfc_a._occ), len(wfc_a._occ[0]), len(wfc_a._occ[0][0])
        x, y, z = wfc_a._psir[0, 0, 0].shape
        all_wfc_a, all_wfc_b = np.zeros((nspin * nkpoint * nband, x, y, z), dtype=float), np.zeros(
            (nspin * nkpoint * nband, x, y, z), dtype=float)
        for i, wfc_index in enumerate(np.ndindex(nspin, nkpoint, nband)):
            ispin, ikpoint, iband = wfc_index
            all_wfc_a[i] = wfc_a._psir[ispin, ikpoint, iband]
            all_wfc_b[i] = wfc_b._psir[ispin, ikpoint, iband]
        return all_wfc_a, all_wfc_b

    def _compute_overlap_matrix(self, wfc_a, wfc_b):
        """
        :param wfc_a: (Wavefunction) wavefunction of a
        :param wfc_b: (Wavefunction) wavefunction of b
        :return: the overlap matrix define as:
                            Oab[i][j] = phi(b)[i] * phi(a)[j]
        """
        all_wfc_a, all_wfc_b = self.__get_all_wavefunction(wfc_a, wfc_b)
        overlap_matrix = np.zeros((len(all_wfc_a), len(all_wfc_a)), dtype=float)
        for i in range(len(all_wfc_a)):
            for j in range(len(all_wfc_a)):
                overlap_matrix[i][j] = np.sum(all_wfc_a[j]*all_wfc_b[i])
        return overlap_matrix

    @abstractmethod
    def _compute_F(self, file_path):
        pass

    def _compute_S(self, wfc_a, wfc_b, Oab):
        """
        :param wfc_a: (Wavefunction) wavefunction of a
        :param wfc_b: (Wavefunction) wavefunction of b
        :param Oab:   (np.array) the overlap matrix define as:
                                    Oab[i][j] = phi(b)[i] * phi(a)[j]
        :return: number Sab
        """
        # TODO: implement

    @abstractmethod
    def _compute_V(self, file_path):
        pass

    def _compute_W(self, wfc_a, wfc_b, weight, Oab):
        """
        :param wfc_a:  (Wavefunction) wavefunction of a
        :param wfc_b:  (Wavefunction) wavefunction of b
        :param weight: (np.array) the weight in the form (np.array, shape == [nspin, n1, n2, n3])
        :param Oab:    (np.array) the overlap matrix define as:
                                    Oab[i][j] = phi(b)[i] * phi(a)[j]
        :return:
        """
        C = np.transpose(inv(Oab) @ (det(Oab) * np.eye(len(Oab))))
        all_wfc_a, all_wfc_b = self.__get_all_wavefunction(wfc_a, wfc_b)
        wba = np.zeros((len(all_wfc_a), len(all_wfc_a)), dtype=float)
        for i in range(len(all_wfc_a)):
            for j in range(len(all_wfc_a)):
                # TODO: implement: complete <phi(B)|w|phi(A)>
        return np.sum(wba * C)


    def calculate(self):
        """
        :return: the electron coupling Hab
        """
        Fb = self._compute_F(file_path=self.path_to_b)
        Fa = self._compute_F(file_path=self.path_to_a)
        Oab = self._compute_overlap_matrix(self.wfc_a, self.wfc_b)
        Sab = self._compute_S(self.wfc_a, self.wfc_b, Oab)
        Sba = self._compute_S(self.wfc_b, self.wfc_a, Oab)

        Vb = self._compute_V(self.path_to_b)
        Va = self._compute_V(self.path_to_a)
        Wab = self._compute_W(self.wfc_a, self.wfc_b, self.weight, Oab)
        Wba = self._compute_W(self.wfc_b, self.wfc_a, self.weight, Oab)
        return 0.5*(Fb*Sab + Fa*Sba - Vb*Wab - Va*Wba)
