from typing import Optional

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume

from unitair import count_qubits
from unitair.simulation.utils import gate_batch_size
from .tensors import state_vectors, tensors_size_fixed, \
    state_vectors_with_metadata, operators


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
        strictly_complex=True
    ))
    batch_dims = state.size()[:-1]
    num_qubits = count_qubits(state)
    shape = batch_dims + (2 ** num_qubits,)
    angles = draw(tensors_size_fixed(shape=shape, dtype=np.dtype('float32')))
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
        strictly_complex: bool = True,
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
        strictly_complex=strictly_complex
    ))

    state_num_qubits = state_data['num_qubits']
    state_batch_dims = state_data['batch_dims']
    state_dtype = state_data['dtype']

    assume(
        state_num_qubits >= op_min_num_qubits
    )

    op_num_qubits = draw(st.integers(
        min_value=op_min_num_qubits,
        max_value=min(state_num_qubits, op_max_num_qubits)
    ))

    operator = draw(operators(
        min_num_qubits=op_num_qubits,
        max_num_qubits=op_num_qubits,
        batch_max_num_indices=batch_max_num_indices,
        batch_max_index_range=batch_max_index_range,
        forced_dtype=state_dtype,
        nonzero=nonzero,
        max_abs=max_abs
    ))
    op_batch_dims = gate_batch_size(gate=operator)

    assume(
        len(op_batch_dims) == 0
        or len(state_batch_dims) == 0
        or op_batch_dims == state_batch_dims
    )

    op_data = dict(
        operator=operator,
        num_qubits=op_num_qubits,
        batch_dims=op_batch_dims,
        dtype=state_dtype
    )

    return dict(operator_data=op_data, state_data=state_data)


