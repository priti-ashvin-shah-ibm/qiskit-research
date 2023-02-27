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
from typing import Optional
from collections import defaultdict, OrderedDict
from qiskit import BasicAer
from qiskit.providers import Backend, Provider
from qiskit_aer import AerSimulator


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
    coupling_map: list, init_layout: set
) -> list[list]:
    """Give a sub-set of desired qubits denoted in init_layout, which is limited by qubits
    within the coupling map, generate a new list of entangling qubits.  The entangling qubits
    is a subset of availabe qubits from the coupling map.

    Args:
        coupling_map (list): From provider's backend.
        init_layout (set): Qubit_ids which are desired and a subset of available
                            qubits from coupling map.

    Raises:
        ValueError: User requested a qubit with does not exist in the coupling map.

    Returns:
        list[list]: Same format at coupling map, but contains only qubits which
                    are desired from init_layout. The list has been sorted
                    by both the first and second qubits pairs.
    """
    # Working on this.
    answer, coupling_set = confirm_init_layout_qubits_in_coupling_map(
        coupling_map, init_layout
    )
    if not answer:
        error_string = "A request to use qubit which does not exist in backend."
        raise ValueError(error_string)

    the_diff = coupling_set.symmetric_difference(init_layout)
    if the_diff:
        # The coupling_map_dict is an OrderedDict SO the keys are already sorted.
        coupling_map_dict = convert_list_map_to_dict(coupling_map)
        for qubit_id in the_diff:
            coupling_map_dict.pop(qubit_id)

        # Rebuild the reduced list map for qubits that user denoted in init_layout.
        entangling_map = []

        # Choose to sort just twice when exporting the information in dict,
        # versus using OrderedDict which sorts each time data is added to dict.

        # Sort just the keys of dict which represents the first_qubit of pair.
        qubits_sorted = sorted(coupling_map_dict)
        for first_qubit in qubits_sorted:
            # Sort the value of the dict which represents the second_qubit of pair.
            connection = sorted(coupling_map_dict[first_qubit])
            for second_qubit in connection:
                if second_qubit in init_layout:
                    entangling_map.append([first_qubit, second_qubit])

        # The number of qubits is LESS, than what was provided by provider.
        return entangling_map
    else:
        # The number of qubits within init_layout is the same as provided by provider.
        return coupling_map


def confirm_init_layout_qubits_in_coupling_map(
    coupling_map: list[list], init_layout: set
) -> tuple[bool, set]:
    """Confirm that init_layout has qubits that are equal or less than the coupling map.

    Args:
        coupling_map (list[list]): From backend determined by provider.
        init_layout (set): Determined by user, typically the best qubits on a given backend.

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
        OrderedDict: Each key is a start qubit, the value hold a list of qubits that can be
                    be second qubit.  This accounts for if the qubits are non-symmetric.
    """

    if list_map:  # If there is something inside the list_map.
        map_dict = defaultdict(list)
        # map_dict = OrderedDict()
    else:
        warnings.warn("The list_map is empty. No dict will be returned.")
        return None

    for pair in list_map:
        len_sublist = len(pair)
        if len_sublist == 2:
            first_qubit, second_qubit = pair

            # Use this syntax to use ordered dict, and then use list as default.
            # map_dict.setdefault(first_qubit, []).append(second_qubit)
            map_dict[first_qubit].append(second_qubit)  # When using defaultdict.
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
