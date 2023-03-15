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

"""Utilities for dealing with backends and their coupling maps."""
import warnings
from typing import Optional, Union, Tuple
from copy import deepcopy
from collections import defaultdict, OrderedDict
from qiskit import BasicAer
from qiskit.providers import Backend, Provider
from qiskit_aer import AerSimulator
import numpy as np
from numpy.linalg import matrix_power
from collections import deque


def get_backend(
    name: str, provider: Optional[Provider], seed_simulator: Optional[int] = None
) -> Backend:
    """Retrieve a backend."""
    if provider is not None:
        return provider.get_backend(name)
    if name == "aer_simulator":
        return AerSimulator(seed_simulator=seed_simulator)
    if name == "statevector_simulator":
        return BasicAer.get_backend("statevector_simulator")
    raise ValueError("The given name does not match any supported backends.")


def convert_list_map_to_dict(list_map: list) -> defaultdict:
    """Reorganize the coupling map since qubits may not be symmetric.

    Args:
        list_map (list): The map obtained from the backend.

    Raises:
        ValueError: Each sublist-pair within coupling_map should be a start and end qubit integers.

    Returns:
        defaultdict: Each key is a start qubit, the value hold a list of qubits that can be
                    be second qubit.  This accounts for if the qubits are non-symmetric.
    """

    if list_map:  # If there is something inside the list_map.
        map_dict = defaultdict(list)
    else:
        warnings.warn("The list_map is empty. No dict will be returned.")
        return None

    for pair in list_map:
        len_sublist = len(pair)
        if len_sublist == 2:
            first_qubit, second_qubit = pair
            map_dict[first_qubit].append(second_qubit)
        else:
            error_string = (
                f"The length of each sublist within list map should contain "
                f"only 2 integers. You have {len_sublist},  for pair: {pair}"
            )
            raise ValueError(error_string)
    return map_dict


def convert_dict_to_list(coupling_map_dict: defaultdict) -> list:
    """Change the format from dict to list of potential qubits.

    Args:
        coupling_map_dict (defaultdict): A dict where the key is fist_qubit, and the value is the
                                        list of potential second qubits.

    Returns:
        list: Reformatted list of qubit pairs.
    """
    coupling_map_list = []
    for first_qubit, second_qubit_list in coupling_map_dict.items():
        for second_qubit in second_qubit_list:
            coupling_map_list.append((first_qubit, second_qubit))

    return coupling_map_list


def matrix_to_dict(
    entangled_result: "numpy.ndarray",
    sorted_init_layout: list,
) -> dict:
    """Extract qubit ids for when the matrix is equal to 1 and formats the data
    with key=start_qubit with a value(list) of the second_qubit.

    The number in the matrix corresponds to how many paths in the graphs connect
    the two qubits. For a value of 3 then, the qubits could be next to each other
    but you can find paths that double-back to the qubit in question. For this reason,
    it is important to only consider the pairs whose value is 1, along with not being
    on the diagonal.

    Args:
        entangled_result (numpy.ndarray): 2-D square array with each index corresponding to
                                        qubits in the sorted_init_layout.
        sorted_init_layout (list): Sorted list of the user-provided init_layout.  The init_layout,
                                ideally denotes the "good" qubits.

    Returns:
        dict: Using the entangled_result, provide key=start_qubit with a value(list)
            of the second_qubit
    """

    entangling_dict = defaultdict(list)
    for first_index, row in enumerate(entangled_result):
        for second_index, value in enumerate(row):
            if value == 1 and first_index != second_index:
                entangling_dict[sorted_init_layout[first_index]].append(
                    sorted_init_layout[second_index]
                )
    return entangling_dict


class PopulateCouplingMapDictAndMatrixDict:
    """Return entangling_map_dict and reduced_coupling_list."""

    def __init__(self, coupling_map: list, init_layout: set, qubit_distance: int = 2):
        """Prepare the data so that logic to pair the qubits can be implemented.

        Args:
            coupling_map (list): From provider's backend.
            init_layout (set): Qubit_ids which are desired and a subset of available
                                qubits from coupling map.
            qubit_distance (int, optional): Determines exponent for matrix multiplication. Defaults to 2.
        """
        self.coupling_map = coupling_map
        # Used to determine self.the_diff
        self.coupling_set = None
        # coupling_map_dict (defaultdict): Reformatted coupling_map key=first_qubit,
        #                                     value=list of all qubits connected to it.
        self.coupling_map_dict = None

        self.init_layout = init_layout
        # sorted_init_layout (list): Can use the index of the qubits to determine layout of matrix axis.
        self.sorted_init_layout = None

        # the_diff (set): Qubits that are within the coupling_map VS init_layout.
        self.the_diff = None

        self.qubit_distance = qubit_distance
        self.populate_coupling_map_dict()

        self.entangled_result = None  # The result of matrix multiplication.
        self.entangle_matrix = None
        # Populate a dict to contain the information about the reduced coupling map.
        self.reduced_coupling_map = defaultdict(list)
        self.create_entangle_matrix()
        self.entangling_dict = self._matrix_to_get_entangle_dict()  # defaultdict(list)

    def populate_coupling_map_dict(self):
        """Do some error checking and convert the coupling map to a dict.

        Raises:
            ValueError: User requested a qubit with does not exist in the coupling map.
        """
        self.sorted_init_layout = sorted(self.init_layout)
        # Working on this.
        answer, self.coupling_set = self._confirm_init_layout_qubits_in_coupling_map()
        if not answer:
            error_string = "A request to use qubit which does not exist in backend."
            raise ValueError(error_string)

        self.the_diff = self.coupling_set.symmetric_difference(self.sorted_init_layout)
        self.coupling_map_dict = convert_list_map_to_dict(self.coupling_map)

    def create_entangle_matrix(self):
        """Give an equal or subset of desired qubits denoted in self.init_layout, which should be limited by qubits
        within the coupling map, generate a new dict of entangling qubits.  The entangling qubits
        is equal or a subset of available qubits from the self.coupling map that are apart by self.qubit_distance.
        """

        if self.the_diff:
            for qubit_id in self.the_diff:
                self.coupling_map_dict.pop(qubit_id)
        # Note: coupling_map_dict has been reduced by init_layout just for ONLY the key.
        # The value will be reduced later and put into reduced_coupling_map.

        # Interim solution, still need to address self.qubit_distance.
        size_of_matrix = len(self.sorted_init_layout)

        # Populate the matrix with zero.
        self.entangle_matrix = np.zeros([size_of_matrix, size_of_matrix], dtype=int)

    def _confirm_init_layout_qubits_in_coupling_map(self) -> tuple[bool, set]:
        """Confirm that init_layout has qubits that are equal or less than the coupling map.

        Returns:
            bool: If self.init_layout has qubits within coupling_map.
            set: The qubits from coupling map.
        """
        cm_set = set()
        il_set = set(self.init_layout)
        for item in self.coupling_map:
            subset = set(item)
            cm_set.update(subset)

        a_subset = il_set.issubset(cm_set)
        if not a_subset:
            message = (
                f"The qubits in init_layout: {il_set} are not in coupling_map: {cm_set}"
            )
            warnings.warn(message)
        return a_subset, cm_set

    def _matrix_to_get_entangle_dict(
        self,
    ) -> dict:
        """Give an equal or subset of desired qubits denoted in self.init_layout, which should be limited by qubits
        within the coupling map, generate a new dict of entangling qubits.  The entangling qubits
        is equal or a subset of available qubits from the self.coupling map that are apart by self.qubit_distance.

        Returns:
            defaultdict(list): Contains only qubits which are desired from init_layout. The list has been sorted
                by both the first and second qubits pairs. Then put desired qubits formatted within a matrix
                and multiplied by self.qubit_distance times. The number in the matrix corresponds to how many
                paths in the graphs connect the two qubits.  Within the result of matrix multiplication, use
                the qubits with "1"  entry within the matrix, which is not on the diagonal.

        """

        initial_layout_lookup = defaultdict(int)

        # For each qubit, denote the index on the matrix axis.
        for index, qubit in enumerate(self.sorted_init_layout):
            initial_layout_lookup[qubit] = index

        # For every connection between first and second qubits, fill the connections with value of 1.

        # Sort just the keys of dict which represents the first_qubit of pair.
        for first_qubit, connection in sorted(self.coupling_map_dict.items()):
            # The value is a list of connections for second_qubit, so sort that separately.
            for second_qubit in sorted(connection):
                # Rebuild the reduced list map for qubits that user denoted in self.sorted_init_layout.
                if second_qubit in self.sorted_init_layout:
                    self.entangle_matrix[
                        initial_layout_lookup[first_qubit],
                        initial_layout_lookup[second_qubit],
                    ] = 1

                    # This dict has both the key and value limited by init_layout.
                    self.reduced_coupling_map[first_qubit].append(second_qubit)

        self.entangled_result = matrix_power(self.entangle_matrix, self.qubit_distance)
        entangling_dict = matrix_to_dict(self.entangled_result, self.sorted_init_layout)

        return entangling_dict


class GetEntanglingMapFromInitLayout(PopulateCouplingMapDictAndMatrixDict):
    """Use the parent class to gather the data for sorting. Then this class will sort
    according to desired Ansatz.
    """

    def __init__(self, coupling_map: list, init_layout: set, qubit_distance: int = 2):
        """Pass arguments to parent init and denote variable to hold result of sorting.

        Args:
            coupling_map (list): From provider's backend.
            init_layout (set): Qubit_ids which are desired and a subset of available
                                qubits from coupling map.
            qubit_distance (int, optional): Determines exponent for matrix multiplication. Defaults to 2.
        """
        PopulateCouplingMapDictAndMatrixDict.__init__(
            self,
            coupling_map=coupling_map,
            init_layout=init_layout,
            qubit_distance=qubit_distance,
        )
        self.dict_of_layers_of_pairs = defaultdict(lambda: defaultdict(list))
        self.reduced_layers_of_pairs = []  # Will be a lists of lists which is sorted.
        self.reduced_coupling_list_to_del = None

    def pairs_from_n_and_reduced_coupling_map(self):
        """By using the reduced_coupling_map to look at qubit_distance within self.entangling_dict,
        create list of layers.  Each layer is a grouping of tuples of qubit pairs. A grouping of layers is
        made by starting with each qubit identified from the reduced_coupling_map.

        Returns:

            defualtdict[defaultdict]]:  self.dict_of_layers_of_pairs gets populated now. The key is the qubit which is used
                                        to start the search. There is a second key which denotes the unused pairs within the
                                        search.
            list[list(tuple)]: Each sublist are lists of pairs which are not repeated.  Each sublist should be unique.
        """

        reduced_coupling_deque = deque(self.reduced_coupling_map.items())

        ### Rotate reduced_coupling_deque until going through each qubit.
        for idx in range(len(reduced_coupling_deque)):
            reduced_coupling_deque.rotate(1)  # Rotate just once for each idx.
            key = reduced_coupling_deque[0][0]
            list_of_layers_of_pairs = []
            tally_used_pairs = []
            self.reduced_coupling_list_to_del = convert_dict_to_list(
                self.reduced_coupling_map
            )
            for first_qubit, second_qubit_list in reduced_coupling_deque:
                for second_qubit in second_qubit_list:
                    grouping_pair = []
                    grouping_pair = self.handle_pair(
                        first_qubit, second_qubit, grouping_pair
                    )
                    if grouping_pair:
                        grouping_pair.sort()
                        # if grouping_pair not in list_of_layers_of_pairs:
                        list_of_layers_of_pairs.append(grouping_pair)
                        tally_used_pairs.extend(grouping_pair)

                        # Need to rebuild a further reduced_coupling_map by tally_used_pairs.
                        self.reduced_coupling_list_to_del = convert_dict_to_list(
                            self.reduced_coupling_map
                        )
                        self.reduced_coupling_list_to_del = [
                            (q1, q2)
                            for (q1, q2) in self.reduced_coupling_list_to_del
                            if not (
                                (q1, q2) in tally_used_pairs
                                or (q2, q1) in tally_used_pairs
                            )
                        ]
                        a = 5  # for breakpoint.

            self.dict_of_layers_of_pairs[key]["useful"] = list_of_layers_of_pairs

            # Gather the information if there were any unused pairs.
            self.dict_of_layers_of_pairs[key][
                "unused"
            ] = self.reduced_coupling_list_to_del

            # Check to see if the list_of_layer_pairs are unique and
            # gather them in self.reduced_layers_of_pairs.
            if list_of_layers_of_pairs:
                if list_of_layers_of_pairs not in self.reduced_layers_of_pairs:
                    self.reduced_layers_of_pairs.append(list_of_layers_of_pairs)
                else:
                    a = 5  # For breakpoint

        return self.dict_of_layers_of_pairs, self.reduced_layers_of_pairs

    def handle_pair(
        self, first_qubit: int, second_qubit: int, grouping_pair: list
    ) -> list:
        """This is the beginning of a recursive algorithm. Return a list of pair that are n
        away from the first and second qubit.

        Args:
            first_qubit (int): A qubit in a pair from the reduced_coupling_map.
            second_qubit (int): Another qubit in a pair from the reduced_coupling_map.
            grouping_pair (list): Holds a grouping of pairs for which are n away from each other.

        Returns:
            list: List of Tuples which are a pair that are n away from each other.
        """
        a_pair = (first_qubit, second_qubit)
        a_pair_sorted = tuple(sorted(a_pair))
        n_away_first_qubit_list = self.entangling_dict[first_qubit]
        n_away_second_qubit_list = self.entangling_dict[second_qubit]

        # find the pairs from reduced coupling map that are n away from each other.
        if (
            a_pair_sorted in self.reduced_coupling_list_to_del
            or a_pair in self.reduced_coupling_list_to_del
        ):
            grouping_pair.append(a_pair_sorted)

            # Reduce to a new list without qubits used for a_pair_sorted
            self.reduced_coupling_list_to_del = [
                (q1, q2)
                for (q1, q2) in self.reduced_coupling_list_to_del
                if not (
                    first_qubit == q1
                    or first_qubit == q2
                    or second_qubit == q1
                    or second_qubit == q2
                )
            ]

        grouping_pair = self.handle_a_sub_pair(
            grouping_pair, n_away_first_qubit_list, n_away_second_qubit_list
        )

        return grouping_pair

    def handle_a_sub_pair(
        self,
        grouping_pair: list,
        n_away_first_qubit_list: list,
        n_away_second_qubit_list: list,
    ):
        """Part of recursive algorithm to determine the pairs which are n away from each other.

        Args:
            grouping_pair (list): Holds a grouping of pairs for which are n away from each other.
            n_away_first_qubit_list (list): From the entangling_dict to determine which qubits are
                                            n away first qubit.
            n_away_second_qubit_list (list):  From the entangling_dict to determine which qubits are
                                            n away second qubit.

        Returns:
            _type_: Holds the list of Tuples which are a pair that are n away from each other.
        """
        for qubit_start in n_away_first_qubit_list:
            for qubit_test in n_away_second_qubit_list:
                a_pair_test = (qubit_start, qubit_test)
                a_pair_test_sorted = tuple(sorted(a_pair_test))

                if (
                    a_pair_test_sorted in self.reduced_coupling_list_to_del
                    or a_pair_test in self.reduced_coupling_list_to_del
                ):
                    grouping_pair.append(a_pair_test_sorted)

                    # Reduce to a new list without qubits used for a_pair
                    self.reduced_coupling_list_to_del = [
                        (q1, q2)
                        for (q1, q2) in self.reduced_coupling_list_to_del
                        if not (
                            qubit_start == q1
                            or qubit_start == q2
                            or qubit_test == q1
                            or qubit_test == q2
                        )
                    ]

                    grouping_pair = self.handle_pair(
                        qubit_start, qubit_test, grouping_pair
                    )
                    break
        return grouping_pair


def get_outward_coupling_map(coupling_map, ent_map, start_qubits):
    """
    Recursive method for rearranging a coupling map to generate an entanglement
    map outward from a set of starting qubit(s). The new entanglement map is then
    suitable for use by canned algorithms such as TwoLocal in the entanglement= kwarg.

    Args:
        coupling_map: coupling map of backend (TODO: need to currently deepcopy this before calling method)
        ent_map: rearranged coupling map used for generating entanglement in correct order (TODO: currently take empty list)
        start_qubits: qubits to find the pairs containing (TODO: currently takes [middle_qubit])
    """
    next_qubits = []
    for pair in [pair for pair in coupling_map if pair[0] in start_qubits]:
        next_qubits.append(pair[1])
        ent_map.append(pair)
        coupling_map.remove(pair)
        coupling_map.remove(list(reversed(pair)))

    if next_qubits:
        get_outward_coupling_map(coupling_map, ent_map, next_qubits)


def get_layered_ansatz_coupling_map(coupling_map):
    """
    Method to layer entangling gates of an ansatz type into those that can be executed
    simultaneously. Currently assumes spacing of 1 qubit within a layer.
    TODO: extend to more than single-qubit spacing
    """
    # TODO: may want to make the below directional to support certain backends, i.e. ibm_sherbrooke
    ordered_cm = [pair for pair in coupling_map if pair[0] < pair[1]]

    # split entanglement map into sets of non-overlapping qubits
    ent_map = []
    for pair in ordered_cm:
        if ent_map == []:
            ent_map.append([pair])
        elif all(
            [
                any([pair[0] in epair or pair[1] in epair for epair in emap])
                for emap in ent_map
            ]
        ):
            ent_map.append([pair])
        else:
            for emap in ent_map:
                if all(
                    [pair[0] not in epair and pair[1] not in epair for epair in emap]
                ):
                    emap.append(pair)
                    break

    return ent_map
