from typing import Optional

import hypothesis.strategies as st
from hypothesis import assume
import torch

from unitair import Field, count_qubits
from .tensors import state_vectors, tensors_size_fixed, \
    state_vectors_with_metadata, operators_batch_fixed


@st.composite
def state_and_phase_angles(
        draw,
        batch_rank_limit: int = 3,
        batch_size_limit: int = 3,
        min_num_qubits: int = 1,
        max_num_qubits: int = 8,
):
    """Draws batches of state vectors along with compatible phase angles.

    This is a specialized strategy for unitair.simulation.apply_phase.
    """
    state = draw(state_vectors(
        batch_rank_limit=batch_rank_limit,
        batch_size_limit=batch_size_limit,
        min_num_qubits=min_num_qubits,
        max_num_qubits=max_num_qubits,
        field=Field.COMPLEX
    ))
    batch_dims = state.size()[:-2]
    num_qubits = count_qubits(state)
    shape = batch_dims + (2 ** num_qubits,)
    angles = draw(tensors_size_fixed(shape=shape))
    return {
        'angles': angles,
        'state_vector': state
    }


@st.composite
def operator_and_state(
        draw,
        min_num_qubits: int = 1,
        max_num_qubits: int = 8,
        op_min_num_qubits: int = 1,
        op_max_num_qubits: int = 5,
        batch_max_num_indices: int = 3,
        batch_max_index_range: int = 5,
        field: Optional[Field] = None,
        nonzero: bool = False,
        max_abs: Optional[float] = None
):
    """Draws batches of operators and states that are compatible.

    By "compatible", we mean that either the operator has no batch
    dimensions or that the operator and state have identical batch
    dimensions.
    """
    state_data = draw(state_vectors_with_metadata(
        batch_rank_limit=batch_max_num_indices,
        batch_size_limit=batch_max_index_range,
        min_num_qubits=min_num_qubits,
        max_num_qubits=max_num_qubits,
        field=field
    ))

    state_num_qubits = state_data['num_qubits']
    state_batch_dims = state_data['batch_dims']
    field = state_data['field']

    assume(
        state_num_qubits >= op_min_num_qubits
    )

    # Compatability conditions.
    op_batch_dims = draw(st.sampled_from(
        [
            state_batch_dims,
            torch.Size([])
        ]
    ))

    op_num_qubits = draw(st.integers(
        min_value=op_min_num_qubits,
        max_value=min(state_num_qubits, op_max_num_qubits)
    ))

    operator = draw(operators_batch_fixed(
        batch_dims=op_batch_dims,
        min_num_qubits=op_num_qubits,
        max_num_qubits=op_num_qubits,
        field=field,
        nonzero=nonzero,
        max_abs=max_abs
    ))
    op_data = dict(
        operator=operator,
        num_qubits=op_num_qubits,
        batch_dims=op_batch_dims,
        field=field
    )

    return dict(operator_data=op_data, state_data=state_data)


