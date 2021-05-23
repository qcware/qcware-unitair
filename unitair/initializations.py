from typing import Union

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


# TODO: add batch dimensions!
def rand_state(
        num_qubits,
        device: torch.device = None,
        field: Union[Field, str] = Field.COMPLEX,
        requires_grad: bool = False
) -> torch.Tensor:
    field = Field(field.lower())
    if device is None:
        device = unitair.get_default_device()

    if field is Field.COMPLEX:
        size = (2, 2 ** num_qubits)
    elif field is Field.REAL:
        size = (2 ** num_qubits,)
    else:
        assert False, f"Impossible enumeration {field}."

    state = torch.rand(
        size,
        device=device, requires_grad=requires_grad
    )

    norm = states.norm_squared(state, field=field).sqrt()
    return state / norm


def uniform_superposition(num_qubits: int, device: torch.device = None):
    """Create the uniform superposition state |+...+>."""
    if device is None:
        device = unitair.get_default_device()
    real = 2**(-num_qubits/2.) * torch.ones(2**num_qubits, device=device)
    imag = torch.zeros(2**num_qubits, device=device)
    return torch.stack((real, imag), dim=0)
