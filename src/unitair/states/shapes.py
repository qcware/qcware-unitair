import enum
import math
from typing import Union, List, Tuple

import torch


class StateLayout(str, enum.Enum):
    VECTOR = 'vector'
    TENSOR = 'tensor'


def count_qubits(state: torch.Tensor):
    """Get the number of qubits of a state in vector layout.

    This function works for complex and real cases and it is compatible
    with arbitrary batch dimensions.

    Args:
        state (Tensor): State (or batch of states) in vector layout.

    """
    length = state.size()[-1]
    num_bits = round(math.log2(length))
    if 2 ** num_bits != length:
        raise StateShapeError(data=state, expected_layout=StateLayout.VECTOR)
    return num_bits


# TODO: sharpen nomenclature around "gate", "operator", "matrix", etc.
def count_qubits_gate_matrix(gate: torch.Tensor):
    """Get the number of qubits that a gate matrix acts on.

    By convention, a gate matrix has the shape

        (*optional_batch_dims, 2^k, 2^k)

    where k is the number of qubits that the gate acts on. Note that
    k might be smaller than the number of qubits in a state that we
    are going to apply the gate to.
    """
    length = gate.size()[-1]
    num_bits = round(math.log2(length))
    if 2 ** num_bits != length:
        raise RuntimeError(f'Given gate matrix has size {gate.size()} which '
                           f'is not consistent with any number of qubits.')
    return num_bits


def hilbert_space_dim(state: torch.Tensor):
    """Get the dimension of the Hilbert space for a state in vector layout."""
    return 2 ** count_qubits(state)


def count_qubits_tensor(
        state_tensor: torch.Tensor,
        num_batch_dims: int,
):
    """Get the number of qubits of a state in tensor layout."""
    return state_tensor.dim() - num_batch_dims


def count_batch_dims_tensor(
        state_tensor: torch.Tensor,
        num_qubits: int,
):
    """Count the number of batch dimensions for a state in tensor layout.
    """
    return state_tensor.dim() - num_qubits


def get_qubit_indices(
        index: Union[int, List[int], Tuple[int, ...], torch.Tensor],
        state_tensor: torch.Tensor,
        num_qubits: int
):
    """Convert qubit indices 0, ..., n-1 to correct PyTorch tensor indices.

    Consider a state with two qubits in tensor layout and with one batch
    dimension. The size is (7, 2, 2) if the batch length is 7.
    The first dimension (with torch index 0) refers to the batch
    while the last two refer to qubits. If we assign the first and second
    qubits "qubit indices" 0 and 1 respectively, then the torch indices of
    (0 and 1) are (1 and 2) respectively.

    Examples:
        >>> state = torch.rand(2, 2)
        >>> get_qubit_indices(0, state, num_qubits=2)
        0
        >>> state = torch.rand(500, 17, 2, 2, 2)
        >>> get_qubit_indices(0, state, num_qubits=3)
        2
        >>> state = torch.rand(500, 17, 2, 2, 2)
        >>> get_qubit_indices([1, 0], state, num_qubits=3)
        [3, 2]
        >>> # Negative indices are not converted to positive:
        >>> state = torch.rand(500, 17, 2, 2, 2)
        >>> get_qubit_indices([-1, 0], state, num_qubits=3)
        [-1, 2]
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


# TODO: This function doesn't necessarily apply to states only so it would
#   be sensible to move it to utils or something.
def subset_roll_to_back(tensor: torch.Tensor, subset_num_dims):
    """Transpose front indices to the end of `tensor`.

    Examples:
        >>> x = torch.rand(4, 5, 6, 7, 8, 9)
        >>> subset_roll_to_back(x, 2).size()
        torch.Size([6, 7, 8, 9, 4, 5])
    """
    complement_range = range(subset_num_dims, tensor.dim())
    subset_range = range(subset_num_dims)
    perm = list(complement_range) + list(subset_range)
    return torch.permute(tensor, perm)


def subset_roll_to_front(tensor: torch.Tensor, subset_num_dims):
    """Transpose back indices to the front of `tensor`.

    Examples:
        >>> x = torch.rand(4, 5, 6, 7, 8, 9)
        >>> subset_roll_to_front(x, 2).size()
        torch.Size([8, 9, 4, 5, 6, 7])
    """
    subset_range = range(tensor.dim() - subset_num_dims, tensor.dim())
    complement_range = range(tensor.dim() - subset_num_dims)
    perm = list(subset_range) + list(complement_range)
    return torch.permute(tensor, perm)


class StateShapeError(ValueError):
    def __init__(
            self,
            data: torch.Tensor = None,
            expected_layout: StateLayout = None
    ):
        if data is not None:
            self.data_size = tuple(data.size())
        else:
            self.data_size = None
        if expected_layout is not None:
            expected_layout = StateLayout(expected_layout.lower())

        self.layout = expected_layout

    def __str__(self):
        message = (
            "There is a problem with the shape of a state.\n"
        )
        if self.layout is StateLayout.VECTOR:
            message += (
                "Expected VECTOR layout. This means that states are specified"
                " by a tensor with size (*optional_batch_dims, 2^num_qubits)\n"
        )
        if self.layout is StateLayout.TENSOR:
            message += (
                "Expected TENSOR layout. This means that states are specified"
                " by a tensor with size\n"
                "    (*optional_batch_dims, 2, 2, ..., 2)\n"
                "where the number of 2's is the number of qubits.\n"
            )
        if self.data_size is not None:
            message += f"Given state has size {self.data_size}"
        return message
