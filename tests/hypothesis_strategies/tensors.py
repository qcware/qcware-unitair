from typing import Iterable, Optional, Tuple, Union

import torch
from hypothesis.extra.numpy import arrays, array_shapes, from_dtype
from hypothesis import assume
import numpy as np

from unitair.states.shapes import Field


import hypothesis.strategies as st


def _norm(x: torch.Tensor, field: Field):
    """Get the norm of a state in vector layout.

    This function is redundant with unitair functionality to avoid
    circular testing.
    """
    if field is Field.REAL:
        return torch.norm(x, dim=-1)

    elif field is Field.COMPLEX:
        return torch.norm(x, dim=[-1, -2])
    else:
        raise TypeError(f'{field} is not a Field.')


def hilbert_space_dimensions(max_num_qubits, allow_0_qubits: bool = False):
    """Build a strategy that draws 2^k with k up to max_num_qubits."""
    def exponentiate(k):
        return 2 ** k

    if allow_0_qubits:
        min_num_qubits = 0
    else:
        min_num_qubits = 1
    return st.builds(exponentiate, st.integers(min_num_qubits, max_num_qubits))


@st.composite
def tensors(
        draw,
        min_dims: int = 0,
        max_dims: int = 3,
        min_side: int = 1,
        max_side: int = 1024,
        dtype: np.dtype = np.dtype('float32'),
        allow_infinity: bool = False
):
    """Strategy for torch.Tensor objects.

    Essentially wraps the standard array strategy with some assumptions."""
    shape_strategy = array_shapes(
        min_dims=min_dims,
        max_dims=max_dims,
        min_side=min_side,
        max_side=max_side
    )
    element_strategy = from_dtype(
        dtype=dtype,
        allow_nan=False,
        allow_infinity=allow_infinity
    )
    array = draw(arrays(
        dtype=dtype,
        shape=shape_strategy,
        elements=element_strategy
    ))

    return torch.tensor(array)


@st.composite
def tensors_size_fixed(
        draw,
        shape: Union[Tuple[int], torch.Size],
        dtype: np.dtype = np.dtype('float32'),
        allow_infinity: bool = False
):
    """Strategy for Tensor objects with definite shape."""
    shape = tuple(shape)
    element_strategy = from_dtype(
        dtype=dtype, allow_nan=False, allow_infinity=allow_infinity)

    array = draw(arrays(
        dtype=dtype,
        shape=shape,
        elements=element_strategy
    ))
    return torch.tensor(array)


def state_vector_size(
        batch_rank_limit: int = 3,
        batch_size_limit: int = 3,
        min_num_qubits: int = 1,
        max_num_qubits: int = 8,
        field: Field = Field.COMPLEX
):
    def construct_state_size(
            batch_dims: Tuple[int, ...],
            num_qubits: int
    ):
        if field is Field.COMPLEX:
            field_dim = (2,)
        elif field is Field.REAL:
            field_dim = ()
        else:
            raise TypeError('Field must be REAL or COMPLEX.')

        return batch_dims + field_dim + (2 ** num_qubits,)

    batch_dim_strategy = array_shapes(
        min_dims=0,
        max_dims=batch_rank_limit,
        min_side=1,
        max_side=batch_size_limit
    )
    return st.builds(
        construct_state_size,
        batch_dims=batch_dim_strategy,
        num_qubits=st.integers(min_num_qubits, max_num_qubits)
    )


@st.composite
def state_vectors(
        draw,
        batch_rank_limit: int = 3,
        batch_size_limit: int = 3,
        min_num_qubits: int = 1,
        max_num_qubits: int = 8,
        field: Optional[Field] = Field.COMPLEX,
):
    """Strategy for drawing normalized states in vector layout.

    Args:
        draw: Hypothesis draw parameter--ignore this for testing.
        batch_rank_limit: The largest allowed number of batch indices.
        batch_size_limit: The largest allowed dimension for batch indices.
        min_num_qubits: The smallest number of qubits allowed.
        max_num_qubits: The largest number of qubits allowed.
        field: Whether the state is real or complex. If None, the strategy
            can draw from either.
    """
    if field is None:
        field = draw(st.sampled_from(Field))
    size = draw(
        state_vector_size(
            batch_rank_limit=batch_rank_limit,
            batch_size_limit=batch_size_limit,
            min_num_qubits=min_num_qubits,
            max_num_qubits=max_num_qubits,
            field=field)
    )
    element_strategy = from_dtype(
        dtype=np.dtype('float32'),
        min_value=-1, max_value=1,
        allow_nan=False, allow_infinity=False
    )
    array = draw(arrays(
        dtype=np.dtype('float32'),
        shape=size,
        elements=element_strategy
    ))
    state = torch.tensor(array)

    state_norm = _norm(state, field)
    # Manually try to avoid zero states with a random shift
    if (state_norm.abs() < 1e-6).any():
        state = state + .1 * torch.rand_like(state)
        state_norm = _norm(state, field)

    # At this point, we "assume" that the states are all nonzero
    assume(
        (state_norm.abs() > 1e-6).all()
    )
    if field is Field.REAL:
        state_norm.unsqueeze_(-1)
    else:
        state_norm.unsqueeze_(-1).unsqueeze_(-1)

    return state / state_norm


@st.composite
def state_vectors_with_metadata(
        draw,
        batch_rank_limit: int = 3,
        batch_size_limit: int = 3,
        min_num_qubits: int = 1,
        max_num_qubits: int = 8,
        field: Optional[Field] = Field.COMPLEX
):
    num_qubits = draw(st.integers(min_num_qubits, max_num_qubits))
    if field is None:
        field = draw(st.sampled_from(Field))

    state_vector = draw(state_vectors(
        batch_rank_limit,
        batch_size_limit,
        min_num_qubits=num_qubits,
        max_num_qubits=num_qubits,
        field=field,
    ))

    if field is Field.REAL:
        batch_dims = state_vector.size()[:-1]
    elif field is Field.COMPLEX:
        batch_dims = state_vector.size()[:-2]
    else:
        raise RuntimeError('Unexpected Field: ', field)
    return {
        'state_vector': state_vector,
        'num_qubits': num_qubits,
        'batch_dims': batch_dims,
        'field': field,
    }


def state_vectors_no_batch(
        min_num_qubits: int = 1,
        max_num_qubits: int = 8,
        field: Optional[Field] = Field.COMPLEX
):
    """Convenience wrapper of state_vectors strategy with batching off.

    See documentation for state_vectors.
    """
    return state_vectors(
        batch_rank_limit=0,
        batch_size_limit=1,
        min_num_qubits=min_num_qubits,
        max_num_qubits=max_num_qubits,
        field=field,
    )
