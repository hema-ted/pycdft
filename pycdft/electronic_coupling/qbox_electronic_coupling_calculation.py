import numpy as np

import os
import glob
from .base import ElectronicCouplingBase
from ..wavefunction_loader.qboxwfcloader import QboxWavefunctionParser
import xml.etree.ElementTree as ET


class QboxElectronicCouplingCalculator(ElectronicCouplingBase):

    def __init__(self, file_path_to_a, file_path_to_b, weight, path_to_a_wfc="", path_to_b_wfc=""):
        super(file_path_to_a, file_path_to_b, weight)
        if(path_to_a_wfc == ""):
            path_to_a_wfc = file_path_to_a
        if(path_to_b_wfc == ""):
            path_to_b_wfc = file_path_to_b
        self.wfc_a = self.loadwfc(path_to_a_wfc)
        self.wfc_b = self.loadwfc(path_to_b_wfc)

    def loadwfc(self, path_to_wfc):
        wfc_parser = QboxWavefunctionParser(path_to_wfc)
        wfc_parser.parse()
        return wfc_parser.wfc

    def _compute_F(self, file_path):
        if( file_path[-1] != "/"):
            file_path += "/"
        file_path += "c1"
        return np.loadtxt(file_path, dtype=float)[-1][-1]

    def _compute_V(self, file_path):
        if (file_path[-1] != "/"):
            file_path += "/"
        file_path += "qbox_outputs/*"
        list_of_file = glob.glob(file_path)
        latest_file = max(list_of_file, key=os.path.getctime())
        energy_file_xml = ET.ElementTree(file=latest_file).getroot()
        for i in energy_file_xml.findall('iteration'):
            etotal = float((i.findall('etotal'))[0].text)
        return etotal













