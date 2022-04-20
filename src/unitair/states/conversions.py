import torch
from . import shapes


def to_tensor_layout(state: torch.Tensor) -> torch.Tensor:
    """Convert a state in vector layout to one in tensor layout.

    Tensor layout is a convenient view of a state where different indices
    are used for different qubits. An n qubit state with arbitrary
    batch dimensions is represented, in tensor layout, as a torch.Tensor
    with size

        (*optional_batch_dims, 2, 2, ..., 2)

    where the number of 2's is equal to the number of qubits. (This can
    be confusing for batch dims with length 2!). Tensor layout should be
    contrasted with "vector layout" which is the standard form of states in
    Unitair. In vector layout, states have size

        (*optional_batch_dims, 2^n).

    This function converts from vector to tensor layout.

    Note:
        Even though tensor layout has the word "tensor", both tensor and
        vector layout states are torch.Tensor objects.

    Args:
        state: Tensor with size (*batch_dims, 2^num_qubits) representing a
            quantum state in vector layout.

    Returns:
        `state` viewed as a tensor with size (*batch_dims, 2, 2, ..., 2).
    """
    num_qubits = shapes.count_qubits(state)

    # We might consider adding a `reject_if_not_contiguous` optional parameter.
    # If so, this conversion to a contiguous Tensor would not happen without
    # the user being aware of it.
    if not state.is_contiguous():
        state = state.contiguous()

    qubit_size = torch.Size(num_qubits * [2])
    batch_size = (state.size())[:-1]
    return state.view(batch_size + qubit_size)


def to_vector_layout(
        state_tensor: torch.Tensor,
        num_qubits: int
) -> torch.Tensor:
    """Convert a state in tensor layout to one in vector layout.

    Vector layout is the standard form of quantum states in Unitair.
    States in vector layout are torch.Tensor objects with size

        (*optional_batch_dims, 2^n)

    where n is the number of qubits. Meanwhile, Tensor layout is a convenient
    view of a state where different indices are used for different qubits. An
    n qubit state is represented, in tensor layout, as a torch.Tensor
    with size

        (*optional_batch_dims, 2, 2, ..., 2)

    where the number of 2's is equal to the number of qubits. (This can
    be confusing for batch dims with length 2!).

    If state_tensor is not contiguous, this function will first
    get a contiguous copy. We may consider making this behavior optional
    with a `reject_if_not_contiguous` optional parameter.

    Note:
        Even though tensor layout has the word "tensor", both tensor and
        vector layout states are torch.Tensor objects.

    Args:
        state_tensor: Tensor with size (*batch_dims, 2, 2, ..., 2) where the
            number of 2's is the number of qubits.

        num_qubits: The number of qubits for the state.

    Returns:
        Tensor with size (*batch_dims, 2^num_qubits,) representing a quantum
        state in vector layout.
    """
    if not state_tensor.is_contiguous():
        state_tensor = state_tensor.contiguous()

    qubit_size = torch.Size((2 ** num_qubits,))
    batch_size = (state_tensor.size())[:-num_qubits]
    return state_tensor.view(batch_size + qubit_size)
