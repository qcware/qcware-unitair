import enum
import math
from typing import Union, List, Tuple

import torch


class StateLayout(str, enum.Enum):
    VECTOR = 'vector'
    TENSOR = 'tensor'


class Field(str, enum.Enum):
    REAL = 'real'
    COMPLEX = 'complex'


def count_qubits(state: torch.Tensor):
    """Get the number of qubits of a state in vector layout.

    Args:
        state: State (or batch of states) in vector layout.

    """
    length = state.size(-1)
    num_bits = round(math.log2(length))
    if 2 ** num_bits != length:
        raise StateShapeError(data=state, expected_layout=StateLayout.VECTOR)
    return num_bits


def count_qubits_gate_matrix(gate: torch.Tensor):
    """Get the number of qubits that a gate matrix acts on.

    By convention, a gate matrix has the shape
        Complex case: (*optional_batch_dims, 2, 2^k, 2^k)
        Real case: (*optional_batch_dims, 2^k, 2^k)

    where k is the number of qubits that the gate acts on. Note that
    k might be smaller than the number of qubits in a state that we
    are going to apply the gate on.
    """
    length = gate.size(-1)
    num_bits = round(math.log2(length))
    if 2 ** num_bits != length:
        raise RuntimeError(f'Given gate matrix has size {gate.size()} which '
                           f'is not consistent with any number of qubits.')
    return num_bits


def hilbert_space_dim(state: torch.Tensor):
    """Get the dimension of the Hilbert space for a state in vector layout."""
    return 2 ** count_qubits(state)


def count_qubits_tensor(
        state_tensor: torch.Tensor, field: Field = Field.COMPLEX
):
    """Get the number of qubits of a state in tensor layout."""
    field = Field(field.lower())
    if field is Field.REAL:
        return state_tensor.dim()
    elif field is Field.COMPLEX:
        return state_tensor.dim() - 1
    else:
        assert False


def real_imag(state: torch.Tensor):
    """Extract the real and imaginary parts of a state in vector layout.
    """
    real = state.select(dim=-2, index=0)
    imag = state.select(dim=-2, index=1)
    return real, imag


def real_imag_tensor(state_tensor: torch.Tensor, num_qubits: int):
    """Extract the real and imaginary parts of a state in tensor layout.
    """
    real = state_tensor.select(dim=-(num_qubits+1), index=0)
    imag = state_tensor.select(dim=-(num_qubits+1), index=1)
    return real, imag


def get_qubit_indices(
        index: Union[int, List[int], Tuple[int, ...], torch.Tensor],
        state_tensor: torch.Tensor,
        num_qubits: int
):
    """Convert qubit indices 0, ..., n-1 to correct PyTorch tensor indices.

    Consider a state with 2 qubits in tensor format (with complex field). If
    there are no batch dimensions, then the torch.Size will be (2, 2, 2). The
    first dimension (with torch index 0) refers to the real and imaginary parts
    while the last two refer to qubits. If we assign the first and second
    qubits "qubit indices" 0 and 1 respectively, then the torch indices of
    (0 and 1) are (1 and 2) respectively.

    If there are batch dimensions, a similar issue arises: a batch of states
    in tensor layout may have dimension (500, 17, 2, 2, 2) but only have two
    qubits. In this case, the torch index of qubit 0 is 3 and the torch index
    of qubit 1 is 4.

    Examples:
        >>> state = torch.rand(2, 2, 2)
        >>> get_qubit_indices(0, state, num_qubits=2)
        1
        >>> state = torch.rand(500, 17, 2, 2, 2)
        >>> get_qubit_indices(0, state, num_qubits=2)
        3
        >>> state = torch.rand(500, 17, 2, 2, 2)
        >>> get_qubit_indices([1, 0], state, num_qubits=2)
        [4, 3]
        >>> # Negative indices behave as expected:
        >>> state = torch.rand(500, 17, 2, 2, 2)
        >>> get_qubit_indices(-1, state, num_qubits=2)
        -1
    """
    batch_dims = state_tensor.dim() - num_qubits
    range_message = 'Expected index in {-num_qubits, ..., num_qubits - 1}.\n'
    range_message += f'Num_qubits: {num_qubits}, index: {index}.'

    convert = False
    if not isinstance(index, torch.Tensor):
        convert = True
        index = torch.tensor(index)

    if (index >= num_qubits).any() or (index < -num_qubits).any():
        raise ValueError(range_message)
    index = index + (index >= 0) * batch_dims
    if convert:
        return index.tolist()
    else:
        return index


class StateShapeError(ValueError):
    def __init__(
            self,
            data: torch.Tensor = None,
            expected_layout: StateLayout = None,
            expected_field: Field = None
    ):
        if data is not None:
            self.data_size = tuple(data.size())
        else:
            self.data_size = None
        if expected_layout is not None:
            expected_layout = StateLayout(expected_layout.lower())

        self.layout = expected_layout
        self.field = expected_field

    def __str__(self):
        message = (
            "There is a problem with the shape of a state.\n"
        )
        if self.layout is StateLayout.VECTOR:
            message += (
                "Expected VECTOR layout. This means that states are specified"
                " by a tensor with one of these sizes:\n"
                "   (2, 2^num_bits) for complex vectors\n"
                "   (2^num_bits,) for real vectors.\n"
        )
        if self.layout is StateLayout.TENSOR:
            message += (
                "Expected TENSOR layout. This means that states are specified"
                " by a tensor with size (2, 2, ..., 2) where the number "
                "of 2's is:\n"
                "   num_qubits in the real case\n"
                "   num_qubits + 1 in the complex case\n"
            )
        if self.field is not None:
            message += f"Expected field: {self.field.value.upper()}\n"
        if self.data_size is not None:
            message += f"Given state has size {self.data_size}"
        return message
