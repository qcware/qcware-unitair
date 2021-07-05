from typing import Union, Sequence, Optional

import torch

import unitair
from . import states
from .states import Field


def unit_vector(
        index: int, dim: int,
        device: torch.device = None,
        field: Field = Field.COMPLEX,
        dtype: torch.dtype = torch.float
):
    """Create a real or complex unit vector in a Hilbert space."""
    if device is None:
        device = unitair.get_default_device()
    field = Field(field.lower())
    if field is Field.REAL:
        vector = torch.zeros(dim, device=device, dtype=dtype)
        vector[index] = 1
    elif field is Field.COMPLEX:
        vector = torch.zeros(2, dim, device=device, dtype=dtype)
        vector[0, index] = 1
    else:
        assert False, f"Impossible enumeration {field}"
    return vector


def rand_state(
        num_qubits,
        batch_dims: Optional[Sequence] = None,
        device: torch.device = None,
        field: Union[Field, str] = Field.COMPLEX,
        requires_grad: bool = False
) -> torch.Tensor:
    if batch_dims is None:
        batch_dims = torch.Size()
    else:
        batch_dims = torch.Size(batch_dims)
    field = Field(field.lower())
    if device is None:
        device = unitair.get_default_device()

    if field is Field.COMPLEX:
        size = (2, 2 ** num_qubits)
    elif field is Field.REAL:
        size = (2 ** num_qubits,)
    else:
        assert False, f"Impossible enumeration {field}."
    state_num_dims = len(size)

    size = batch_dims + size
    state = torch.rand(
        size,
        device=device,
        requires_grad=requires_grad
    )
    norm = states.norm_squared(state, field=field).sqrt()
    norm_size_for_division = norm.size() + (1,) * state_num_dims
    return state / norm.view(norm_size_for_division)


def uniform_superposition(
        num_qubits: int,
        batch_dims: Optional[Sequence] = None,
        device: torch.device = None,
        requires_grad: bool = False
):
    """Create the uniform superposition state |+...+>.

    This function is defined for the complex case only. Optional batch
    dimensions can be provided in which case the same uniform superposition
    is copied to form a batch with specified shape.
    """
    if device is None:
        device = unitair.get_default_device()
    real = 2**(-num_qubits/2.) * torch.ones(2**num_qubits, device=device)
    imag = torch.zeros(2**num_qubits, device=device)
    state = torch.stack((real, imag), dim=0)
    if batch_dims is not None:
        batch_dims = torch.Size(batch_dims)
        state = state.repeat(batch_dims + (1, 1))

    state.requires_grad_(requires_grad)
    return state
