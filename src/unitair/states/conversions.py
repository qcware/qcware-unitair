import torch
from . import shapes


def to_tensor_layout(state: torch.Tensor) -> torch.Tensor:
    """Convert a state in vector layout to one in tensor layout.

    TODO: insert explanation of tensor layout.

    todo: batching implemented

    This function automatically detects if the state's field
    is real or complex.

    Args:
        state: Tensor with size (2, 2^num_qubits) or (2^num_qubits,)
            representing a quantum state in vector layout.

        field: When COMPLEX (default), the state should have size
            (2, 2^num_qubits,).  When REAL, state has size (2^num_qubits,)

    Returns:
        `state` viewed as a tensor with size (2, 2, ..., 2).
    """
    num_qubits = shapes.count_qubits(state)

    if not state.is_contiguous():
        state = state.contiguous()

    qubit_size = torch.Size(num_qubits * [2])
    remaining_size = (state.size())[:-1]
    return state.view(remaining_size + qubit_size)


def to_vector_layout(
        state_tensor: torch.Tensor, num_qubits: int
) -> torch.Tensor:
    """Convert a state in tensor layout to one in vector layout.

    TODO: insert explanation of tensor and vector layout.

    If state_tensor is not contiguous, this function will first
    get a contiguous copy.

    If there is more than one qubit, the field cannot be inferred in
    tensor layout so field is specified as an argument here.

    Args:
        state_tensor: Tensor with size (2, 2, ..., 2) where the number
            of 2's is either num_qubits or num_qubits+1 in the real and
            complex cases respectively.

        num_qubits: The number of qubits for the state.

        field: Specifies if the state is in a real or complex Hilbert space.

    Returns:
        Tensor with size (2^num_qubits,) or (2, 2^num_qubits)
            representing a quantum state in vector layout.
    """
    if not state_tensor.is_contiguous():
        state_tensor = state_tensor.contiguous()

    qubit_size = torch.Size((2 ** num_qubits,))
    remaining_size = (state_tensor.size())[:-num_qubits]
    return state_tensor.view(remaining_size + qubit_size)
