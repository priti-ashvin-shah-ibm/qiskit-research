# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Coupling Groups."""
import unittest
from qiskit import QuantumCircuit, transpile
import copy
from qiskit.tools import job_monitor
from qiskit.visualization import (
    plot_histogram,
    plot_gate_map,
    plot_circuit_layout,
    plot_error_map,
)

from qiskit_research.utils import backend
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import FakeWashington, FakeWashingtonV2
from qiskit_research.utils import (
    get_backend,
    convert_list_map_to_dict,
    convert_dict_to_list,
    matrix_to_dict,
    get_outward_coupling_map,
    get_layered_ansatz_coupling_map,
    PopulateCouplingMapDictAndMatrixDict,
    GetEntanglingMapFromInitLayout,
)

from qiskit_ibm_provider import IBMProvider


class TestEntanglingMap(unittest.TestCase):
    """Test passes."""

    def setUp(self):
        # setUp is run before every individual test method
        ###########  Use simulated backend.  ###############
        self.backend = FakeWashington()  # 127 qubit device.
        self.config = self.backend.configuration()
        # the_backend_plot = plot_error_map(backend)
        self.the_coupling_map_list = self.config.coupling_map

        self.the_initial_layout_set_wrong = {
            0,
            1,
            2,
            3,
            400,
        }  # This was test for raise error.
        self.the_initial_layout_set = {
            20,
            21,
            22,
            23,
            24,
            34,
            43,
            42,
            41,
            40,
            39,
            33,
            19,
            15,
            25,
            44,
            53,
            38,
        }  # Meaningful list for FakeWashington

    def tearDown(self):
        None

    def get_entangling_map_2(self, qubit_distance: int = 2):

        new_layers = GetEntanglingMapFromInitLayout(
            self.the_coupling_map_list,
            self.the_initial_layout_set,
            qubit_distance=qubit_distance,
        )
        (
            dict_of_layers_of_pairs,
            reduced_layers_of_pairs,
        ) = new_layers.pairs_from_n_and_reduced_coupling_map()

        # Then compare to expected results using https://docs.python.org/3/library/unittest.html.

    def get_entangling_map_3(self, qubit_distance: int = 3):

        new_layers = GetEntanglingMapFromInitLayout(
            self.the_coupling_map_list,
            self.the_initial_layout_set,
            qubit_distance=qubit_distance,
        )
        (
            dict_of_layers_of_pairs,
            reduced_layers_of_pairs,
        ) = new_layers.pairs_from_n_and_reduced_coupling_map()

        # Then compare to expected results using https://docs.python.org/3/library/unittest.html.
