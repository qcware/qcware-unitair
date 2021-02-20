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

# def get_field(state: torch.Tensor):
#     """Given a state (in vector layout), return the associated field."""
#     if state.dim() == 1:
#         return Field.REAL
#     elif state.dim() == 2:
#         return Field.COMPLEX
#     else:
#         raise StateShapeError(data=state, expected_layout=StateLayout.VECTOR)
#
#
# def get_field_tensor(state_tensor: torch.Tensor, num_qubits: int):
#     """Given a state in tensor layout, return the associated field."""
#     if state_tensor.dim() == num_qubits:
#         return Field.REAL
#     elif state_tensor.dim() == num_qubits + 1:
#         return Field.COMPLEX
#     else:
#         raise StateShapeError(
#             data=state_tensor, expected_layout=StateLayout.TENSOR
#         )


def count_qubits(state: torch.Tensor):
    """Get the number of qubits of a state in vector layout.
    """
    length = state.size(-1)
    num_bits = round(math.log2(length))
    if 2 ** num_bits != length:
        raise StateShapeError(data=state, expected_layout=StateLayout.VECTOR)
    return num_bits


def hilbert_space_dim(state: torch.Tensor):
    """Get the dimension of the Hilbert space for a state in vector layout."""
    return 2 ** count_qubits(state)

# def count_qubits_tensor(
#         state_tensor: torch.Tensor, field: Field = Field.COMPLEX
# ):
#     """Get the number of qubits of a state in tensor layout."""
#     field = Field(field.lower())
#     if field is Field.REAL:
#         return state_tensor.dim()
#     elif field is Field.COMPLEX:
#         return state_tensor.dim() - 1
#     else:
#         assert False


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
    """Convert qubit indices 0, ..., n-1 to correct PyTorch tensor indices."""
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
        return convert

    # # Assume now that index is an int
    # if 0 <= index < num_qubits:
    #     # possible complex dimension is included as a batch dim.
    #
    #     return index + batch_dims
    # elif -num_qubits <= index < 0:
    #     return index
    # else:
    #     raise ValueError(range_message)


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
