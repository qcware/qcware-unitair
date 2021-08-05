import hypothesis
import torch

from unitair import Field
from unitair.simulation.operations import apply_phase
from unitair.simulation.operations import act_first_qubits

from unitair.states import count_qubits, count_qubits_gate_matrix
from math import pi

from hypothesis import given, assume
from tests.hypothesis_strategies.specialized import state_and_phase_angles
from tests.hypothesis_strategies.tensors import operators, state_vectors
from unitair.simulation.utils import gate_batch_size


@given(state_and_phase_angles())
def test_apply_phase(data):
    angles = data['angles']
    original_state = data['state_vector']

    state = apply_phase(angles, original_state)
    state = apply_phase(-angles, state)
    assert state.isclose(original_state, atol=1e-4).all()

    complement_angles = 2 * pi - angles
    state = apply_phase(angles, original_state)
    state = apply_phase(complement_angles, state)
    # The weird atol here stops an issue with large angles ruining accuracy.
    # I think this is occurring because pi does not have sufficient precision.
    assert state.isclose(
        original_state,
        atol=1e-4 + angles.abs().sum() * .001
    ).all()


@given(
    op=operators(max_num_qubits=5, field=Field.COMPLEX),
    state=state_vectors(field=Field.COMPLEX)
)
def test_act_first_qubits(op, state):
    """
    Just a smoke-test for the time being
    """
    state_num_qubits = count_qubits(state)
    operator_num_qubits = count_qubits_gate_matrix(op)

    state_batch_shape = state.size()[:-2]
    op_batch_shape = gate_batch_size(gate=op, field=Field.COMPLEX)
    valid_no_op_batch = (op_batch_shape == torch.Size([]))
    valid_batch_match = (op_batch_shape == state_batch_shape)

    assume(state_num_qubits >= operator_num_qubits)
    assume(valid_batch_match or valid_no_op_batch)

    # Just a smoke test for now.
    act_first_qubits(
        operator=op,
        state=state,
        field=Field.COMPLEX,
    )


