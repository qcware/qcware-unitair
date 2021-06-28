
import hypothesis.strategies as st

from unitair import Field, count_qubits
from .tensors import state_vectors, tensors_size_fixed


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



