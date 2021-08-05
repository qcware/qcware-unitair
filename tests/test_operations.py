import random

import hypothesis
import pytest
import torch

from unitair import Field
from unitair.simulation.operations import apply_phase
from unitair.simulation.operations import act_first_qubits
from unitair.simulation.operations import apply_all_qubits
from unitair.simulation.operations import swap

from unitair.states import count_qubits, count_qubits_gate_matrix
from math import pi

from hypothesis import given, assume
import hypothesis.strategies as st

from tests.hypothesis_strategies.specialized import state_and_phase_angles
from tests.hypothesis_strategies.specialized import operator_and_state

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
    op_and_state=operator_and_state(),
    indices=st.lists(st.integers(min_value=0))
)
def test_act_first_qubits_batching(op_and_state, indices):
    indices = torch.Size(indices)
    op_data = op_and_state['operator_data']
    state_data = op_and_state['state_data']

    operator = op_data['operator']
    state_vector = state_data['state_vector']
    op_batch_dims = op_data['batch_dims']
    state_batch_dims = state_data['batch_dims']

    out = act_first_qubits(
        operator=operator,
        state=state_vector,
        field=op_data['field'],
    )

    # If there are no state batch dims, there's nothing to check
    # although the above is still a smoke-test.
    if len(state_batch_dims) == 0:
        return

    assume(len(indices) == len(state_batch_dims))
    assume(all(i < j for i, j in zip(indices, state_batch_dims)))
    out_entry = out[indices]

    state_vector_entry = state_vector[indices]

    if len(op_batch_dims) > 0:
        operator_entry = operator[indices]
    else:
        operator_entry = operator

    unbatched_out = act_first_qubits(
        operator=operator_entry,
        state=state_vector_entry,
        field=op_data['field'],
    )
    assert (out_entry == unbatched_out).all()


# @pytest.mark.skip()
@given(
    op_and_state=operator_and_state(op_max_num_qubits=1),
    indices=st.lists(st.integers(min_value=0))
)
def test_apply_all_qubits_batching(op_and_state, indices):
    """
    Just a smoke-test for the time being
    """
    indices = torch.Size(indices)
    op_data = op_and_state['operator_data']
    state_data = op_and_state['state_data']

    operator = op_data['operator']
    state_vector = state_data['state_vector']
    op_batch_dims = op_data['batch_dims']
    state_batch_dims = state_data['batch_dims']

    out = apply_all_qubits(
        operator=operator,
        state=state_vector,
        field=op_data['field'],
    )

    # If there are no state batch dims, there's nothing to check
    # although the above is still a smoke-test.
    if len(state_batch_dims) == 0:
        return


    assume(len(indices) == len(state_batch_dims))
    assume(all(i < j for i, j in zip(indices, state_batch_dims)))
    out_entry = out[indices]

    state_vector_entry = state_vector[indices]

    if len(op_batch_dims) > 0:
        operator_entry = operator[indices]
    else:
        operator_entry = operator

    unbatched_out = apply_all_qubits(
        operator=operator_entry,
        state=state_vector_entry,
        field=op_data['field'],
    )
    assert (out_entry == unbatched_out).all()


@given(
    state=state_vectors(max_num_qubits=8),
    i=st.integers(min_value=0, max_value=8),
    j=st.integers(min_value=0, max_value=8)
)
def test_swap(state, i, j):
    n = count_qubits(state)

    # These ensure n >= 2 without further assumption.
    assume(i != j)
    assume(i < n)
    assume(j < n)

    state_swapped = swap(state, qubit_pair=(i, j))
    state_swapped_other_way = swap(state, qubit_pair=(j, i))

    state_double_swapped = swap(state_swapped, qubit_pair=(i, j))

    # test that swap is involutory
    assert (state == state_double_swapped).all()

    # test that swap is index-symmetrical
    assert (state_swapped == state_swapped_other_way).all()
