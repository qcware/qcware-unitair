import torch
from typing import Union
import wrapt


@wrapt.decorator
def single_qubit_gate(gate_function):
    """Build a single qubit gate given a gate function.

    This is intended to be used as a decorator. The gate function takes
    in a tensor of parameters and returns a nested list of tensors
    or a tensor.
    # TODO: add documentation to explain this better.
    """
    def run_gate(params):
        """Get the gate with specified params."""
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
    return run_gate


@single_qubit_gate
def exp_x(angle: Union[torch.Tensor, float]):
    """Get the operator e^(-i angle X)."""
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

    For example, suppose that a, b, c, and d are all tensors of size (5,).
    Then, matrix_stack([[a, b], [c, d]]) returns a tensor of size (2, 2, 5).

    If roll is set to True, then the last output dimension is rolled to
    the first dimension. This is useful if that dimension was supposed to
    be a batch dimension. For example, setting roll=True in the example
    above will result in a tensor of size (5, 2, 2).
    """
    def recursive_stack(params_):
        if isinstance(params_[0], torch.Tensor):
            return torch.stack(params_)
        num_rows = len(params_)
        return torch.stack(
            [nested_stack(params_[i]) for i in range(num_rows)]
        )

    out = recursive_stack(params).squeeze(0)
    if roll:
        perm = [out.dim() - 1] + list(range(out.dim() - 1))
        return out.permute(perm)
    else:
        return out
