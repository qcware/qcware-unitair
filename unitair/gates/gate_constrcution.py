import torch
import decorator
from typing import Optional


@decorator.decorator
def parameterized_gate(gate_function, *args):
    """Build a single qubit gate given a gate function.

    This is intended to be used as a decorator. The gate function takes
    in a tensor of parameters and returns a nested list of tensors
    or a tensor.
    # TODO: add documentation to explain this better.
    """
    params = args[0]
    squeeze = False
    if not isinstance(params, torch.Tensor):
        params = torch.tensor([float(params)], device=torch.device("cpu"))
        squeeze = True
    elif params.dim() == 0:
        params = params.unsqueeze(0)
        squeeze = True
    # elif params.dim() != 1:
    #     raise ValueError(
    #         "angle should be a tensor with no more than 1 index.")

    gate = gate_function(params)
    if not isinstance(gate, torch.Tensor):
        gate = nested_stack(gate, roll=True)
    if squeeze:
        gate = gate.squeeze(0)
    return gate


@decorator.decorator
def constant_gate(gate_function, real_or_imag: Optional[str] = None, *args):
    device = args[0]
    if device is None:
        device = torch.device("cpu")
    gate = torch.tensor(gate_function(), device=device)
    if real_or_imag is None:
        return gate
    else:
        zeros = torch.zeros(size=gate.size(), device=device)
        if real_or_imag.lower() == 'real':
            complex_gate = [gate, zeros]
        elif real_or_imag.lower() == 'imag':
            complex_gate = [zeros, gate]
        else:
            raise ValueError(
                'argument real_or_imag should be \'real\' or \'imag\'.\n'
                f'Found {real_or_imag}.')
        return torch.stack(complex_gate)


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
