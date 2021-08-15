from typing import Union, Sequence, Optional

import torch

from . import states
from .states import Field


def unit_vector_from_bitstring(
        bitstring: Union[str, Sequence[int]],
        device: torch.device = torch.device("cpu"),
        field: Field = Field.COMPLEX,
        dtype: torch.dtype = torch.float
):
    """Create a unit vector from a classical bit specification.

    This is meant for convenience in interactive experimentation.
    The function `unit_vector` is preferable in many contexts.

    Examples:
        >>> unit_vector_from_bitstring('000')  # doctest: +SKIP
        tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.]])

        Bitstring specification is somewhat flexible.
        >>> unit_vector_from_bitstring([1, 0])  # doctest: +SKIP
        tensor([[0., 0., 1., 0.],
                [0., 0., 0., 0.]])

    """
    try:
        bitstring = torch.tensor(bitstring)
    except TypeError:
        bitstring = torch.tensor([int(i) for i in bitstring])

    implied_num_bits = bitstring.size()[0]
    powers = torch.tensor([2 ** k for k in reversed(range(implied_num_bits))])
    index_implied = (powers * bitstring).sum()

    return unit_vector(
        index=index_implied,
        num_qubits=implied_num_bits,
        device=device,
        field=field,
        dtype=dtype
    )


def unit_vector(
        index: int,
        dim: Optional[int] = None,
        num_qubits: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        field: Field = Field.COMPLEX,
        dtype: torch.dtype = torch.float
):
    """Create a real or complex unit vector in a Hilbert space."""
    if dim is None:
        if num_qubits is None:
            raise TypeError(
                'To specify a unit vector, provide either `dim` or '
                '`num_qubits`.')
        dim = 2 ** num_qubits
    else:
        if num_qubits is not None:
            raise TypeError(
                'Unit vector can be specified by `dim` or `num_qubits` but '
                'not both.'
            )
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
        device: torch.device = torch.device("cpu"),
        field: Union[Field, str] = Field.COMPLEX,
        requires_grad: bool = False
) -> torch.Tensor:
    """Create a normalized random state in vector layout.

    States are uniformly distributed on the unit sphere in C^N where N = 2^n.

    When batch_dims is provided, a batch of random states is generated with
    specified shape.
    """
    if batch_dims is None:
        batch_dims = torch.Size()
    else:
        batch_dims = torch.Size(batch_dims)
    field = Field(field.lower())

    if field is Field.COMPLEX:
        size = (2, 2 ** num_qubits)
    elif field is Field.REAL:
        size = (2 ** num_qubits,)
    else:
        assert False, f"Impossible enumeration {field}."
    state_num_dims = len(size)

    size = batch_dims + size
    state = torch.randn(
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
        device: torch.device = torch.device("cpu"),
        requires_grad: bool = False
):
    """Create the uniform superposition state |+...+>.

    This function is defined for the complex case only. Optional batch
    dimensions can be provided in which case the same uniform superposition
    is copied to form a batch with specified shape.
    """
    real = 2**(-num_qubits/2.) * torch.ones(2**num_qubits, device=device)
    imag = torch.zeros(2**num_qubits, device=device)
    state = torch.stack((real, imag), dim=0)
    if batch_dims is not None:
        batch_dims = torch.Size(batch_dims)
        state = state.repeat(batch_dims + (1, 1))

    state.requires_grad_(requires_grad)
    return state
