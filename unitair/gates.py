import torch
from typing import Union
import decorator


@decorator.decorator
def single_qubit_gate(gate_function, *args):
    """Build a single qubit gate given a gate function.

    This is intended to be used as a decorator. The gate function takes
    in a tensor of parameters and returns a nested list of tensors
    or a tensor.
    # TODO: add documentation to explain this better.
    """
    params = args[0]
    squeeze = False
    if not isinstance(params, torch.Tensor):
        params = torch.tensor([float(params)])
        squeeze = True
    elif params.dim() == 0:
        params = params.unsqueeze(0)
        squeeze = True
    elif params.dim() != 1:
        raise ValueError(
            "angle should be a tensor with no more than 1 index.")

    gate = gate_function(params)
    if not isinstance(gate, torch.Tensor):
        gate = nested_stack(gate, roll=True)
    if squeeze:
        gate = gate.squeeze(0)
    return gate


@single_qubit_gate
def exp_x(angle: Union[torch.Tensor, float]):
    """Get the operator e^(-i angle X).

    Args:
        angle: Tensor with size (batch_length,) or just ().

    Returns: Tensor with size (batch_length, 2, 2, 2) or (2, 2, 2) if there
        is no batch dimension. The (2, 2, 2) is such that the first dimension
        means the real and imaginary parts and the last two dimension are
        the matrices of the real an imaginary parts of the gates.
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    zero = torch.zeros(angle.size(), device=angle.device)
    return [
        [[cos, zero],
         [zero, cos]],

        [[zero, -sin],
         [-sin, zero]]
    ]


@single_qubit_gate
def exp_y(angle: Union[torch.Tensor, float]):
    """Get the operator e^(-i angle Y)."""
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    zero = torch.zeros(angle.size(), device=angle.device)
    return [
        [[cos, -sin],
         [sin, cos]],

        [[zero, zero],
         [zero, zero]]
    ]


@single_qubit_gate
def exp_z(angle: Union[torch.Tensor, float]):
    """Get the operator e^(-i angle Z)."""
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    zero = torch.zeros(angle.size(), device=angle.device)
    return [
        [[cos, zero],
         [zero, cos]],

        [[-sin, zero],
         [zero, sin]]
    ]


def nested_stack(params, roll: bool = False):
    """Form a tensor from a nexted list of tensors.

    This function is a generalization of torch.stack. For proper usage,
    it's important that params is a nested list with shape consistent with
    and array. The innermost elements of that nested list should be PyTorch
    tensors, all of which have identical size.

    For an example, suppose that a, b, c, and d are all tensors of size (5,).
    Then, nested_stack([[a, b], [c, d]]) returns a tensor of size (2, 2, 5).

    If roll is set to True, then the dimensions of the tensors (like a, b, c
    and d in the example above) will be permuted to the start of the output.
    This is useful if those dimensions were supposed to be batch dimensions.
    In the example, the output with roll=True would have size (5, 2, 2).
    If instead a, b, c, and d all had size (6, 9, 8), then the output size
    would be (6, 9, 8, 2, 2) if roll=True and (2, 2, 6, 9, 8) if roll=False.
    """
    def recursive_stack(params_):
        if isinstance(params_[0], torch.Tensor):
            return torch.stack(params_)
        num_rows = len(params_)
        return torch.stack(
            [nested_stack(params_[i]) for i in range(num_rows)]
        )
    stacked = recursive_stack(params).squeeze(0)

    if roll:
        inner = params[0]
        while not isinstance(inner, torch.Tensor):
            inner = inner[0]
        inner_dim = inner.dim()
        perm = list(range(stacked.dim()-inner_dim, stacked.dim())) + list(range(stacked.dim()-inner_dim))
        return stacked.permute(perm)
    else:
        return stacked
