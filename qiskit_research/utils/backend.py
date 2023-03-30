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


def get_entangling_map_from_init_layout(
    coupling_map: list, init_layout: set, qubit_distance: int = 2
) -> Tuple[dict, list]:
    """Give an equal or subset of desired qubits denoted in init_layout, which should be limited by qubits
    within the coupling map, generate a new dict of entangling qubits.  The entangling qubits
    is equal or a subset of available qubits from the coupling map that are apart by qubit_distance.

    Args:
        coupling_map (list): From provider's backend.
        init_layout (set): Qubit_ids which are desired and a subset of available
                            qubits from coupling map.
        qubit_distance (int, optional): Determines exponent for matrix multiplication. Defaults to 2.

    Raises:
        ValueError: User requested a qubit with does not exist in the coupling map.

    Returns:
        dict: Contains only qubits which are desired from init_layout. The list has been sorted
            by both the first and second qubits pairs. Then put desired qubits formatted within a matrix
            and multiplied by qubit_distance times. The number in the matrix corresponds to how many
            paths in the graphs connect the two qubits.  Within the result of matrix multiplication, use
            the qubits with "1"  entry within the matrix.
        list[list]: same data as list dict but each sublist is a pair of key,value from dict.  The value is
            a list, so the pair is made with each entry of the list.
    """
    sorted_init_layout = sorted(init_layout)
    # Working on this.
    answer, coupling_set = confirm_init_layout_qubits_in_coupling_map(
        coupling_map, sorted_init_layout
    )
    if not answer:
        error_string = "A request to use qubit which does not exist in backend."
        raise ValueError(error_string)

    the_diff = coupling_set.symmetric_difference(sorted_init_layout)
    coupling_map_dict = convert_list_map_to_dict(coupling_map)
    entangling_map_dict, reduced_coupling_list = matrix_to_get_entangle_dict(
        the_diff,
        coupling_map_dict,
        sorted_init_layout,
        qubit_distance,
    )
    return entangling_map_dict, reduced_coupling_list


def matrix_to_get_entangle_dict(
    the_diff: set,
    coupling_map_dict: defaultdict,
    sorted_init_layout: list,
    qubit_distance: int = 2,
) -> Tuple[defaultdict, list]:
    """_summary_

    Args:
        the_diff (set): Qubits that are within the coupling_map VS init_layout.
        coupling_map_dict (defaultdict): Reformatted coupling_map key=first_qubit,
                                            value=list of all qubits connected to it.
        sorted_init_layout (list): Can use the index of the qubits to determine layout of matrix axis.
        qubit_distance (int, optional): The exponent of square matrix.  Defaults to 2.

    Returns:
        defaultdict: Contains only qubits which are desired from init_layout. The list has been sorted
            by both the first and second qubits pairs. Then put desired qubits formatted within a matrix
            and multiplied by qubit_distance times. The number in the matrix corresponds to how many
            paths in the graphs connect the two qubits.  Within the result of matrix multiplication, use
            the qubits with "1"  entry within the matrix, which is not on the diagonal.
        list[list(tuple)]: Each sublist is a list of pair where the first qubit is
                            qubit_distance away from second qubit. The pairs are not repeated.
    """

    if the_diff:
        for qubit_id in the_diff:
            coupling_map_dict.pop(qubit_id)
    # Note: coupling_map_dict has been reduced by init_layout just for ONLY the key.
    # The value will be reduced later and put into reduced_coupling_map.

    # Interim solution, still need to address qubit_distance.
    size_of_matrix = len(sorted_init_layout)

    # Populate the matrix with zero.
    entangle_matrix = np.zeros([size_of_matrix, size_of_matrix], dtype=int)
    initial_layout_lookup = defaultdict(int)

    # Populate a dict to contain the information about the reduced coupling map.
    reduced_coupling_map = defaultdict(list)

    # For each qubit, denote the index on the matrix axis.
    for index, qubit in enumerate(sorted_init_layout):
        initial_layout_lookup[qubit] = index

    # For every connection between first and second qubits, fill the connections with value of 1.

    # Sort just the keys of dict which represents the first_qubit of pair.
    for first_qubit, connection in sorted(coupling_map_dict.items()):
        # The value is a list of connections for second_qubit, so sort that separately.
        for second_qubit in sorted(connection):
            # Rebuild the reduced list map for qubits that user denoted in sorted_init_layout.
            if second_qubit in sorted_init_layout:
                entangle_matrix[
                    initial_layout_lookup[first_qubit],
                    initial_layout_lookup[second_qubit],
                ] = 1

                # This dict has both the key and value limited by init_layout.
                reduced_coupling_map[first_qubit].append(second_qubit)

    entangled_result = matrix_power(entangle_matrix, qubit_distance)
    entangling_dict = matrix_to_dict(entangled_result, sorted_init_layout)
    list_of_layers_of_pairs = pairs_from_n_and_reduced_coupling_map(
        entangling_dict, reduced_coupling_map
    )

    return entangling_dict, list_of_layers_of_pairs


def pairs_from_n_and_reduced_coupling_map(
    entangling_dict: defaultdict, reduced_coupling_map: defaultdict
) -> list:
    """By using the reduced coupling map to look at qubit_distance within entangling_dict,
    create list of layers.  Each layer is a grouping of tuples of qubit pairs.

    Args:
        entangling_dict (defaultdict): Contains only qubits which are desired from init_layout. The list has been sorted
            by both the first and second qubits pairs. Then put desired qubits formatted within a matrix
            and multiplied by qubit_distance times. The number in the matrix corresponds to how many
            paths in the graphs connect the two qubits.  Within the result of matrix multiplication, use
            the qubits with "1"  entry within the matrix, which is not on the diagonal.
        reduced_coupling_map (defaultdict): This dict has both the key and value limited by init_layout.

    Returns:
        list[list(tuple)]: Each sublist is a list of pair where the first qubit is
                            qubit_distance away from second qubit. The pairs are not repeated
    """

    # This sort may not be needed.
    # The grouping by layer is different if the below sort is executed.
    sorted_reduced_coupling_map = OrderedDict(
        sorted(reduced_coupling_map.items(), key=lambda x: len(x[1]), reverse=True)
    )
    # Get a list of tuples for what is desired.
    reduced_coupling_list_to_del = dict_to_list(sorted_reduced_coupling_map)
    pairs_with_n = []
    for first_qubit, second_qubit_list in sorted_reduced_coupling_map.items():
        for second_qubit in second_qubit_list:
            n_away_first_qubit_list = entangling_dict[first_qubit]
            # n_away_first_qubit_list.append(first_qubit)
            n_away_second_qubit_list = entangling_dict[second_qubit]
            # n_away_second_qubit_list.append(second_qubit)

            # find the pairs from reduced coupling map that are n away from each other.
            grouping_pair = []
            a_pair = (first_qubit, second_qubit)
            if a_pair in reduced_coupling_list_to_del:
                grouping_pair.append(a_pair)
                reduced_coupling_list_to_del.remove(a_pair)

            for qubit_start in n_away_first_qubit_list:
                for qubit_test in n_away_second_qubit_list:
                    a_pair = (qubit_start, qubit_test)
                    if a_pair in reduced_coupling_list_to_del:
                        grouping_pair.append(a_pair)
                        reduced_coupling_list_to_del.remove(a_pair)
                        # After finding a pair with first qubit, don't look for any more pairs.
                        break
            if grouping_pair:
                pairs_with_n.append(grouping_pair)

    return pairs_with_n


def dict_to_list(coupling_map_dict: defaultdict) -> list:
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


def confirm_init_layout_qubits_in_coupling_map(
    coupling_map: list[list], init_layout: list
) -> tuple[bool, set]:
    """Confirm that init_layout has qubits that are equal or less than the coupling map.

    Args:
        coupling_map (list[list]): From backend determined by provider.
        init_layout (list): Determined by user, typically the best qubits on a given backend.

    Returns:
        bool: If init_layout has qubits within coupling_map.
        set: The qubits from coupling map.
    """
    cm_set = set()
    il_set = set(init_layout)
    for item in coupling_map:
        subset = set(item)
        cm_set.update(subset)

    a_subset = il_set.issubset(cm_set)
    if not a_subset:
        message = (
            f"The qubits in init_layout: {il_set} are not in coupling_map: {cm_set}"
        )
        warnings.warn(message)
    return a_subset, cm_set


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
