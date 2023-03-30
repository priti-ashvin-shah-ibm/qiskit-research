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
    """Test passes using two different qubit_lengths.
    Use  # Then compare to expected results using https://docs.python.org/3/library/unittest.html.
    """

    def setUp(self):
        """Setup unit test runs before every individual test method."""

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

        self.n2_combined_layers_min = [
            [(20, 33), (21, 22), (23, 24), (34, 43), (39, 40), (41, 53)],
            [(15, 22), (20, 21), (24, 34), (33, 39), (40, 41), (42, 43)],
            [(22, 23), (24, 25), (38, 39), (19, 20), (41, 42), (43, 44)],
        ]

        self.n3_combined_layers_min = [
            [(21, 22), (24, 25), (33, 39), (41, 53), (43, 44)],
            [(15, 22), (19, 20), (24, 34), (38, 39), (41, 42)],
            [(20, 21), (23, 24), (39, 40), (42, 43)],
            [(20, 33), (22, 23), (34, 43), (40, 41)],
        ]

    def tearDown(self):
        """Tie any loose ends after each test."""
        pass

    def test_get_entangling_map_2(self, qubit_distance: int = 2):
        """Ensure code changes don't change basic output. This is just a sanity check.

        Args:
            qubit_distance (int, optional): Relates to desired distance between pairs.
                        The value of 2 means pairs can be adjacent to each other or further apart.
                        Defaults to 2.
        """

        new_layers = GetEntanglingMapFromInitLayout(
            self.the_coupling_map_list,
            self.the_initial_layout_set,
            qubit_distance=qubit_distance,
        )
        (
            dict_of_layers_of_pairs,
            unique_layers_of_pairs,
            min_layer_unique_layer_of_pairs,
            combined_layers_min,
        ) = new_layers.pairs_from_n_and_reduced_coupling_map()
        self.assertEqual(len(combined_layers_min), 11)
        self.assertEqual(len(min_layer_unique_layer_of_pairs), 11)
        self.assertEqual(len(unique_layers_of_pairs), 17)
        self.assertEqual(len(dict_of_layers_of_pairs), 18)
        self.assertListEqual(self.n2_combined_layers_min, combined_layers_min[0])

    def test_get_entangling_map_3(self, qubit_distance: int = 3):
        """Ensure code changes don't change basic output. This is just a sanity check.

        Args:
            qubit_distance (int, optional): Relates to desired distance between pairs.
                        The value of 3 means pairs can be 1 qubit, or further apart.
        """

        new_layers = GetEntanglingMapFromInitLayout(
            self.the_coupling_map_list,
            self.the_initial_layout_set,
            qubit_distance=qubit_distance,
        )
        (
            dict_of_layers_of_pairs,
            unique_layers_of_pairs,
            min_layer_unique_layer_of_pairs,
            combined_layers_min,
        ) = new_layers.pairs_from_n_and_reduced_coupling_map()

        self.assertEqual(len(combined_layers_min), 6)
        self.assertEqual(len(min_layer_unique_layer_of_pairs), 6)
        self.assertEqual(len(unique_layers_of_pairs), 16)
        self.assertEqual(len(dict_of_layers_of_pairs), 18)
        self.assertListEqual(self.n3_combined_layers_min, combined_layers_min[0])

    @unittest.expectedFailure
    def test_bad_coupling_map(self, qubit_distance: int = 3):
        """Have code to catch bad input data.
        This case will ask for qubit that is is not in the backend.

        Args:
            qubit_distance (int, optional): _description_. Defaults to 3.
        """
        new_layers = GetEntanglingMapFromInitLayout(
            self.the_coupling_map_list,
            self.the_initial_layout_set_wrong,
            qubit_distance=qubit_distance,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
