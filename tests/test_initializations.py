import torch

from unitair.initializations import rand_state, uniform_superposition
from hypothesis import given
import hypothesis.strategies as st
from tests.hypothesis_strategies import sizes, real_and_complex_torch_dtypes
from unitair.states.innerprod import norm_squared


@given(
    num_qubits=st.integers(1, 5),
    batch_dims=sizes(),
    field=st.sampled_from([torch.complex64, torch.float32])
)
def test_rand_state(num_qubits, batch_dims, field):
    state = rand_state(num_qubits=num_qubits, batch_dims=batch_dims)

    # Confirm expected batch sizes.
    expected_size = batch_dims + (2 ** num_qubits,)

    assert state.size() == expected_size

    # Check that norm_squared == 1 for all entries.
    actual = norm_squared(state)
    expected = torch.tensor(1.)
    assert (actual.isclose(expected)).all()


@given(num_qubits=st.integers(1, 5), batch_dims=sizes())
def test_uniform_superposition(num_qubits, batch_dims):
    state = uniform_superposition(
        num_qubits=num_qubits, batch_dims=batch_dims,
    )

    assert state.size() == batch_dims + (2**num_qubits,)

    expected_value_real = torch.tensor(2 ** (-num_qubits/2.))
    expected_value_imag = torch.tensor(0.)
    real, imag = state.real, state.imag
    assert real.isclose(expected_value_real).all()
    assert imag.isclose(expected_value_imag).all()
