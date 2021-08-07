from random import shuffle
from typing import List

import torch
from hypothesis import given, assume
import hypothesis.strategies as st
from hypothesis.strategies import composite

from unitair import utils
MAX_BITS = 10


@composite
def list_permutations(draw, min_length=1, max_length=100):
    """Draws list-format permutations"""
    length = draw(st.integers(min_value=min_length, max_value=max_length))

    perm = draw(st.lists(
        min_size=length,
        max_size=length,
        elements=st.integers(min_value=0, max_value=length - 1),
        unique=True
    ))
    return perm


@given(permutation=list_permutations())
def test_inverse_permutation_inverts(
        permutation: List[int]
):
    def permute(p: List[int], inp: list):
        """Apply list permutation p to inp."""
        return [inp[p[i]] for i in range(len(p))]

    n = len(permutation)
    inv_permutation = utils.inverse_list_permutation(permutation)
    assert len(inv_permutation) == n

    test_objects = list(range(n))
    shuffle(test_objects)

    objects_permuted = permute(permutation, test_objects)
    permute_then_inverse = permute(inv_permutation, objects_permuted)

    assert permute_then_inverse == test_objects


@given(x=st.integers(0, 2 ** (MAX_BITS - 1)), num_bits=st.integers(1, MAX_BITS))
def test_int_to_bin(x: int, num_bits: int):
    assume(x < 2 ** num_bits)
    binary_digits = utils.int_to_bin(x, num_bits=num_bits)

    powers = torch.tensor([2**n for n in reversed((range(num_bits)))])
    assert x == (binary_digits * powers).sum()



