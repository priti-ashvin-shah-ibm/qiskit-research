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


from typing import Optional
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
) -> dict:
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
    entangling_map_dict = matrix_to_get_entangle_dict(
        the_diff,
        coupling_map_dict,
        sorted_init_layout,
        qubit_distance,
    )
    return entangling_map_dict


def matrix_to_get_entangle_dict(
    the_diff: set,
    coupling_map_dict: defaultdict,
    sorted_init_layout: list,
    qubit_distance: int = 2,
) -> defaultdict:
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
            the qubits with "1"  entry within the matrix.
    """

    if the_diff:
        for qubit_id in the_diff:
            coupling_map_dict.pop(qubit_id)

    # Interim solution, still need to address qubit_distance.
    size_of_matrix = len(sorted_init_layout)

    # Populate the matrix with zero.
    entangle_matrix = np.zeros([size_of_matrix, size_of_matrix], dtype=int)
    initial_layout_lookup = defaultdict(int)

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

    entangled_result = matrix_power(entangle_matrix, qubit_distance)
    entangling_dict = matrix_to_dict(entangled_result, sorted_init_layout)

    return entangling_dict


def matrix_to_dict(
    entangled_result: "numpy.ndarray",
    sorted_init_layout: list,
) -> dict:
    """Extract qubit ids for when the matrix is equal to 1 and formats the data
    with key=start_qubit with a value(list) of the second_qubit.

    The number in the matrix corresponds to how many paths in the graphs connect
    the two qubits. For a value of 3 then, the qubits could be next to each other
    but you can find paths that double-back to the qubit in question. For this reason,
    it is important to only consider the pairs whose value is 1.

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
            if value == 1:
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
