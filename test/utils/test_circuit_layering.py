# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test circuit layering."""

import unittest

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import PauliSumOp
from qiskit.providers.fake_provider import FakeKolkata
from qiskit.transpiler import PassManager

from qiskit_research.utils.circuit_layering import ExpandBlockOperators, FindBlockTrotterEvolution, LayerBlockOperators

# TODO: ceate data for different params & distances
class TestLayeredPauliGates(unittest.TestCase):
    
    num_qubits = 9
    op = PauliSumOp.from_list([("I"*idx + pair + "I"*(9-idx-2), 1) for idx in range(num_qubits-2) for pair in ["XX", "YY", "ZZ"]])
    qc = QuantumCircuit(num_qubits)
    qc.append(PauliEvolutionGate(op, 1.3), range(num_qubits))

    block_ops = ['XX','YY','ZZ']
    qc_fbte = PassManager(FindBlockTrotterEvolution(block_ops=block_ops)).run(qc)

    backend = FakeKolkata()
    coupling_map = backend.configuration().coupling_map
    qc_l = transpile(qc_fbte, coupling_map=coupling_map, seed_transpiler=12345)

    qc_layered = PassManager([
        LayerBlockOperators(block_ops=block_ops, coupling_map=coupling_map, qubit_distance=4),
        ExpandBlockOperators(block_ops=block_ops)
        ]).run(qc_l)
    print(qc_l.draw(idle_wires=False))
    print(qc_layered.draw(idle_wires=False))
    import pdb; pdb.set_trace()