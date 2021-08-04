import torch
from typing import Union, Optional
from .gate_constrcution import parameterized_gate, constant_gate


@parameterized_gate
def exp_x(angle: Union[torch.Tensor, float]):
    """Get the operator e^(-i angle X).

    PyTorch device is inherited from the device of `angle`. If `angle` is
    a float, CPU is used.

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


@parameterized_gate
def exp_y(angle: Union[torch.Tensor, float]):
    """Get the operator e^(-i angle Y).

    PyTorch device is inherited from the device of `angle`. If `angle` is
    a float, CPU is used.

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
        [[cos, -sin],
         [sin, cos]],

        [[zero, zero],
         [zero, zero]]
    ]


@parameterized_gate
def exp_z(angle: Union[torch.Tensor, float]):
    """Get the operator e^(-i angle Z).

    PyTorch device is inherited from the device of `angle`. If `angle` is
    a float, CPU is used.

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

        [[-sin, zero],
         [zero, sin]]
    ]


@constant_gate(real_or_imag='real')
def pauli_x(device: Optional[torch.device] = None):
    """Get the Pauli X operator.

    Args:
        device: If the torch device is not specified, CPU is used.
    """
    return [
        [0., 1.],
        [1., 0.]
    ]


@constant_gate(real_or_imag='imag')
def pauli_y(device: Optional[torch.device] = None):
    """Get the Pauli Y operator.

    Args:
        device: If the torch device is not specified, CPU is used.
    """
    return [
        [0., -1.],
        [1., 0.]
    ]


@constant_gate(real_or_imag='real')
def pauli_z(device: Optional[torch.device] = None):
    """Get the Pauli Z operator.

    Args:
        device: If the torch device is not specified, CPU is used.
    """
    return [
        [1., 0.],
        [0., -1.]
    ]


@constant_gate(real_or_imag='real')
def cnot(device: Optional[torch.device] = None):
    """Get a CNOT gate.

    Args:
        device: If the torch device is not specified, CPU is used.
    """
    return [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [0., 0., 1., 0.]
    ]

