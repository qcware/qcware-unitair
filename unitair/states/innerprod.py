import torch

from .shapes import StateShapeError, StateLayout, Field
from . import shapes


def abs_squared(state: torch.Tensor, field: Field = Field.COMPLEX):
    """Compute the vector of measurement probabilities for state.

    Args:
        state (Tensor): State or batch of states in vector layout. Tensor size
            can be (*batch_dims, 2, 2^n) or (*batch_dims, 2^n) for the real
            and complex cases.

        field (Field): Field of `state`.

    Returns:
        Tensor with size (*batch_dims, 2^n) giving all measurement
        probabilities for all states in the batch.
    """
    field = Field(field.lower())
    if field is Field.COMPLEX:
        return (state ** 2).sum(dim=-2)
    elif field is Field.REAL:
        return state ** 2
    else:
        assert False


def norm_squared(state: torch.Tensor, field: Field = Field.COMPLEX):
    """Compute the L^2 norm-squared < state | state >.

    `state` can be a batch of states. The L^2 norm is used in the real and
    complex case.

    Args:
        state (Tensor): State or batch of states in vector layout. Tensor size
            can be (*batch_dims, 2, 2^n) or (*batch_dims, 2^n) for the real
            and complex cases.

        field (Field): Field of `state`.

    Returns:
        Tensor with size (*batch_dims,) giving the squared norm of every state
        in the batch.
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
