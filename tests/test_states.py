from unitair.states.shapes import count_qubits
from tests.hypothesis_strategies import state_vectors_with_metadata
from hypothesis import given


@given(state_vectors_with_metadata())
def test_count_qubits(state_data):
    state_vector = state_data['state_vector']
    num_qubits = state_data['num_qubits']

    assert count_qubits(state_vector) == num_qubits
