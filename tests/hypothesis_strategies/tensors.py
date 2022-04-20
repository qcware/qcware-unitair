from typing import Optional, Tuple, Union

import torch
from hypothesis.extra.numpy import arrays, array_shapes, from_dtype
from hypothesis import assume
import numpy as np


import hypothesis.strategies as st

NONZERO_CUT = 1e-6


def hilbert_space_dimensions(max_num_qubits, allow_0_qubits: bool = False):
    """Build a strategy that draws 2^k with k up to max_num_qubits."""
    def exponentiate(k):
        return 2 ** k

    if allow_0_qubits:
        min_num_qubits = 0
    else:
        min_num_qubits = 1
    return st.builds(exponentiate, st.integers(min_num_qubits, max_num_qubits))


torch_dtype_to_numpy = {
    torch.complex64: np.dtype('complex64'),
    torch.complex128: np.dtype('complex128'),
    torch.float32: np.dtype('float32'),
    torch.float64: np.dtype('float64'),
}

@st.composite
def tensors(
        draw,
        min_dims: int = 0,
        max_dims: int = 3,
        min_side: int = 1,
        max_side: int = 1024,
        dtype: np.dtype = np.dtype('complex64'),
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


def real_and_complex_torch_dtypes(strictly_complex: bool):
    if strictly_complex:
        return st.just(torch.complex64)
    else:
        return st.sampled_from([torch.complex64, torch.float32])


def real_and_complex_np_dtypes(strictly_complex: bool):
    if strictly_complex:
        return st.just(np.dtype('complex64'))
    else:
        return st.sampled_from([np.dtype('complex64'), np.dtype('float32')])


@st.composite
def tensors_size_fixed(
        draw,
        shape: Union[Tuple[int], torch.Size],
        dtype: np.dtype = np.dtype('complex64'),
        allow_infinity: bool = False
):
    """Strategy for Tensor objects with definite shape and dtype."""
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
):
    def construct_state_size(
            batch_dims: Tuple[int, ...],
            num_qubits: int
    ):
        return batch_dims + (2 ** num_qubits,)

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
        strictly_complex: bool = True
):
    """Strategy for drawing normalized states in vector layout.

    Args:
        draw: Hypothesis draw parameter--ignore this for testing.
        batch_rank_limit: The largest allowed number of batch indices.
        batch_size_limit: The largest allowed dimension for each batch index.
        min_num_qubits: The smallest number of qubits allowed.
        max_num_qubits: The largest number of qubits allowed.
        strictly_complex: When True, only draw complex dtypes.
    """
    dtype = draw(real_and_complex_np_dtypes(strictly_complex))
    size = draw(
        state_vector_size(
            batch_rank_limit=batch_rank_limit,
            batch_size_limit=batch_size_limit,
            min_num_qubits=min_num_qubits,
            max_num_qubits=max_num_qubits,
        ))
    element_strategy = from_dtype(
        dtype=dtype,
        min_value=-1, max_value=1,
        allow_nan=False, allow_infinity=False
    )
    array = draw(arrays(
        dtype=dtype,
        shape=size,
        elements=element_strategy
    ))

    # min_value and max_value don't avoid large arrays for complex cases.
    # TODO: This might have something to do with tests running very slowly.
    #   Need to further investigate!
    if np.iscomplexobj(array):
        assume((np.abs(array) < 2.).all())
    state = torch.tensor(array)

    state_norm = torch.norm(state, dim=-1)
    # Manually try to avoid zero states with a random shift
    if (state_norm.abs() < NONZERO_CUT).any():
        state = state + .1 * torch.rand_like(state)
        state_norm = torch.norm(state, dim=-1)

    # At this point, we "assume" that the states are all nonzero
    assume(
        (state_norm.abs() > NONZERO_CUT).all()
    )
    state_norm.unsqueeze_(-1)

    return state / state_norm


@st.composite
def state_vectors_with_metadata(
        draw,
        batch_rank_limit: int = 3,
        batch_size_limit: int = 3,
        min_num_qubits: int = 1,
        max_num_qubits: int = 8,
        strictly_complex: bool = True,
):
    num_qubits = draw(st.integers(min_num_qubits, max_num_qubits))
    state_vector = draw(state_vectors(
        batch_rank_limit,
        batch_size_limit,
        min_num_qubits=num_qubits,
        max_num_qubits=num_qubits,
        strictly_complex=strictly_complex
    ))
    batch_dims = state_vector.size()[:-1]

    return {
        'state_vector': state_vector,
        'num_qubits': num_qubits,
        'batch_dims': batch_dims,
        'dtype': state_vector.dtype,
    }


def state_vectors_no_batch(
        min_num_qubits: int = 1,
        max_num_qubits: int = 8,
        strictly_complex: bool = True
):
    """Convenience wrapper of state_vectors strategy with batching off.

    See documentation for state_vectors.
    """
    return state_vectors(
        batch_rank_limit=0,
        batch_size_limit=1,
        min_num_qubits=min_num_qubits,
        max_num_qubits=max_num_qubits,
        strictly_complex=strictly_complex
    )


@st.composite
def sizes(
        draw,
        min_num_dims: int = 0,
        max_num_dims: int = 3,
        min_index_range: int = 1,
        max_index_range: int = 6,
):
    """Strategy for torch.Size objects."""
    size = draw(st.lists(
        elements=st.integers(min_index_range, max_index_range),
        min_size=min_num_dims,
        max_size=max_num_dims
    ))
    return torch.Size(size)


@st.composite
def operators(
        draw,
        min_num_qubits: int = 1,
        max_num_qubits: int = 1,
        batch_max_num_indices: int = 3,
        batch_max_index_range: int = 5,
        strictly_complex: Optional[bool] = True,
        forced_dtype: Optional[torch.dtype] = None,
        nonzero: bool = False,
        max_abs: Optional[float] = None
):
    if forced_dtype is not None:
        dtype = torch_dtype_to_numpy[forced_dtype]
    else:
        dtype = draw(real_and_complex_np_dtypes(strictly_complex))
    batch_dims = draw(
        sizes(
            min_num_dims=0,
            max_num_dims=batch_max_num_indices,
            min_index_range=1,
            max_index_range=batch_max_index_range
        )
    )

    num_qubits = draw(st.integers(min_num_qubits, max_num_qubits))
    matrix_dims = (2 ** num_qubits, 2 ** num_qubits)

    all_dims = batch_dims + matrix_dims

    result = draw(tensors_size_fixed(shape=all_dims, dtype=dtype))
    if nonzero:
        assume(
            (result.abs() > NONZERO_CUT).all()
        )
    if max_abs is not None:
        assume(
            (result.abs() < max_abs).all()
        )

    return result


@st.composite
def operators_batch_fixed(
        draw,
        batch_dims: torch.Size = torch.Size([]),
        min_num_qubits: int = 1,
        max_num_qubits: int = 1,
        strictly_complex: bool = True,
        nonzero: bool = False,
        max_abs: Optional[float] = None
):
    dtype = draw(real_and_complex_np_dtypes(strictly_complex))

    num_qubits = draw(st.integers(min_num_qubits, max_num_qubits))
    matrix_dims = (2 ** num_qubits, 2 ** num_qubits)

    all_dims = batch_dims + matrix_dims

    result = draw(tensors_size_fixed(shape=all_dims, dtype=dtype))
    if nonzero:
        assume(
            (result.abs() > NONZERO_CUT).all()
        )

    if max_abs is not None:
        assume(
            (result.abs() < max_abs).all()
        )
    return result
