import torch

from .shapes import StateShapeError, StateLayout, Field
from . import shapes


def abs_squared(state: torch.Tensor, field: Field = Field.COMPLEX):
    """Compute the vector of measurement probabilities for state.

    todo: batching implemented

    Args:
        state:
        field:

    Returns:

    """
    field = Field(field.lower())
    if field is Field.COMPLEX:
        return (state ** 2).sum(dim=-2)
    elif field is Field.REAL:
        return state ** 2
    else:
        assert False


def norm_squared(state: torch.Tensor, field: Field = Field.COMPLEX):
    """Compute < state | state >.

    todo: confirm batching functionality
    """
    field = Field(field.lower())
    return torch.sum(abs_squared(state, field=field), dim=-1)


def expectation_value(function_values, state):
    """Get the expectation value of a real-valued function of binary values."""
    return torch.sum(abs_squared(state) * function_values, dim=-1)


def inner_product(state_1, state_2, field: Field = Field.COMPLEX):
    """Compute < state_1 | state_2 > (the left entry is conjugate-linear).

    todo: implement batching
    """
    field = Field(field.lower())

    if field is Field.REAL:
        return torch.sum(state_1 * state_2)
    elif field is Field.COMPLEX:
        real_1, imag_1 = shapes.real_imag(state_1)
        real_2, imag_2 = shapes.real_imag(state_2)
        real_part = torch.sum(real_1 * real_2 + imag_1 * imag_2)
        imag_part = torch.sum(real_1 * imag_2 - imag_1 * real_2)
        return torch.stack((real_part, imag_part))
    else:
        assert False, f"Impossible enumeration {field}"
