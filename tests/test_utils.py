import torch
from hypothesis import given, assume
import hypothesis.strategies as st

from unitair import utils
MAX_BITS = 10


@given(x=st.integers(0, 2 ** (MAX_BITS - 1)), num_bits=st.integers(1, MAX_BITS))
def test_int_to_bin(x: int, num_bits: int):
    assume(x < 2 ** num_bits)
    binary_digits = utils.int_to_bin(x, num_bits=num_bits)

    powers = torch.tensor([2**n for n in reversed((range(num_bits)))])
    assert x == (binary_digits * powers).sum()



