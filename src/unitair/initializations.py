from typing import Union, Sequence, Optional

import torch

from . import states


def unit_vector_from_bitstring(
        bitstring: Union[str, Sequence[int]],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.complex64
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
        dtype=dtype
    )


def unit_vector(
        index: int,
        num_qubits: Optional[int] = None,
        dim: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.complex64
):
    """Create a real or complex unit vector in a Hilbert space."""
    if dim is None:
        if num_qubits is None:
            raise TypeError(
                'To specify a unit vector, provide either `num_qubits` or '
                ' `dim`.')
        dim = 2 ** num_qubits
    else:
        if num_qubits is not None:
            raise TypeError(
                'Unit vector can be specified by `dim` or `num_qubits` but '
                'not both.'
            )
    vector = torch.zeros(dim, device=device, dtype=dtype)
    vector[index] = 1
    return vector


def rand_state(
        num_qubits: int,
        batch_dims: Optional[Sequence] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.complex64,
        requires_grad: bool = False
) -> torch.Tensor:
    """Create a normalized random state in vector layout.

    States are drawn from a uniform distribution on the unit sphere embedded
    in F^N where N = 2^(num_qubits) and F the complex numbers when the
    specified dtype is complex and the reals when the dtype is real.

    When batch_dims is provided, a batch of random states is generated with
    specified shape.
    """
    if batch_dims is None:
        size = torch.Size((2 ** num_qubits,))
    else:
        size = torch.Size(batch_dims) + torch.Size((2 ** num_qubits,))

    state = torch.randn(
        size,
        device=device,
        dtype=dtype
    )
    norm = states.norm_squared(state).sqrt()
    norm_size_for_division = norm.size() + (1,)
    state /= norm.view(norm_size_for_division)
    return state.requires_grad_(requires_grad)


def uniform_superposition(
        num_qubits: int,
        batch_dims: Optional[Sequence] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.complex64,
        requires_grad: bool = False
):
    """Create the uniform superposition state |+...+>.

    This function is defined for the complex case only. Optional batch
    dimensions can be provided in which case the same uniform superposition
    is copied to form a batch with specified shape.
    """
    scaling = 2 ** (-num_qubits/2.)
    state = scaling * torch.ones(
        2 ** num_qubits,
        device=device,
        dtype=dtype
    )

    if batch_dims is not None:
        batch_dims = torch.Size(batch_dims)
        state = state.repeat(batch_dims + (1,))

    state.requires_grad_(requires_grad)
    return state
