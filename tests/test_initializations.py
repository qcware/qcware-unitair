import torch

from unitair import Field
from unitair.initializations import rand_state, unit_vector, uniform_superposition
from hypothesis import given
import hypothesis.strategies as st
from tests.hypothesis_strategies import sizes
from unitair.states.innerprod import norm_squared
from unitair.states.shapes import real_imag


@given(
    num_qubits=st.integers(1, 5),
    batch_dims=sizes(),
    field=st.sampled_from(Field)
)
def test_rand_state(num_qubits, batch_dims, field):
    state = rand_state(
        num_qubits=num_qubits, batch_dims=batch_dims, field=field
    )

    # Confirm expected batch sizes.
    if field is Field.COMPLEX:
        expected_size = batch_dims + (2, 2**num_qubits)
    else:
        expected_size = batch_dims + (2**num_qubits,)
    assert state.size() == expected_size

    # Check that norm_squared == 1 for all entries.
    actual = norm_squared(state, field=field)
    expected = torch.tensor(1.)
    assert (actual.isclose(expected)).all()


@given(num_qubits=st.integers(1, 5), batch_dims=sizes())
def test_uniform_superposition(num_qubits, batch_dims):
    state = uniform_superposition(
        num_qubits=num_qubits, batch_dims=batch_dims,
    )

    assert state.size() == batch_dims + (2, 2**num_qubits)

    expected_value_real = torch.tensor(2 ** (-num_qubits/2.))
    expected_value_imag = torch.tensor(0.)
    real, imag = real_imag(state)
    assert real.isclose(expected_value_real).all()
    assert imag.isclose(expected_value_imag).all()
