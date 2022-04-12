import torch
from typing import Union, Optional
from .gate_construction import parameterized_gate, constant_gate


@parameterized_gate(strictly_complex=True)
def exp_x(
        angle: Union[torch.Tensor, float],
        dtype: torch.dtype = torch.complex64
):
    """Get the operator e^(-i angle X).

    PyTorch device is inherited from the device of `angle`. If `angle` is
    a float, CPU is used.

    Args:
        angle: Tensor with size (batch_length,) or just ().

        dtype: Data type for the gate. Required to be complex.

    Returns: Tensor with size (batch_length, 2, 2, 2) or (2, 2, 2) if there
        is no batch dimension. The (2, 2, 2) is such that the first dimension
        means the real and imaginary parts and the last two dimension are
        the matrices of the real an imaginary parts of the gates.
    """
    angle = angle.to(dtype=dtype)
    cos = torch.cos(angle)
    i_sin = torch.sin(angle) * 1.j
    return [
        [[cos, -i_sin],
         [-i_sin, cos]],
    ]


@parameterized_gate(strictly_complex=True)
def exp_y(
        angle: Union[torch.Tensor, float],
        dtype: torch.dtype = torch.complex64
):
    """Get the operator e^(-i angle Y).

    PyTorch device is inherited from the device of `angle`. If `angle` is
    a float, CPU is used.

    Args:
        angle: Tensor with size (batch_length,) or just ().

        dtype: Data type for the gate. Required to be complex.

    Returns: Tensor with size (batch_length, 2, 2, 2) or (2, 2, 2) if there
        is no batch dimension. The (2, 2, 2) is such that the first dimension
        means the real and imaginary parts and the last two dimension are
        the matrices of the real an imaginary parts of the gates.
    """
    angle = angle.to(dtype=dtype)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    return [
        [[cos, -sin],
         [sin, cos]],
    ]


@parameterized_gate(strictly_complex=True)
def exp_z(
        angle: Union[torch.Tensor, float],
        dtype: torch.dtype = torch.complex64
):
    """Get the operator e^(-i angle Z).

    PyTorch device is inherited from the device of `angle`. If `angle` is
    a float, CPU is used.

    Args:
        angle: Tensor with size (batch_length,) or just ().

        dtype: Data type for the gate. Required to be complex.

    Returns: Tensor with size (batch_length, 2, 2, 2) or (2, 2, 2) if there
        is no batch dimension. The (2, 2, 2) is such that the first dimension
        means the real and imaginary parts and the last two dimension are
        the matrices of the real an imaginary parts of the gates.
    """
    angle = angle.to(dtype=dtype)
    cos = torch.cos(angle)
    i_sin = torch.sin(angle) * 1.j
    zero = torch.zeros(angle.size(), device=angle.device)
    return [
        [[cos - i_sin, zero],
         [zero, cos + i_sin]],
    ]


@constant_gate(strictly_complex=True)
def hadamard(
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
):
    """Get the Hadamard gate.

    Args:
        device: If the torch device is not specified, CPU is used.

        dtype: torch.dtype for the result.
            When not specified, torch.complex64 is used.
    """
    val = 2. ** (-.5) + 0.j
    return [
        [val, val],
        [val, -val]
    ]


@constant_gate(strictly_complex=True)
def pauli_x(
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
):
    """Get the Pauli X gate.

    Args:
        device: If the torch device is not specified, CPU is used.

        dtype: torch.dtype for the result.
            When not specified, torch.complex64 is used.
    """
    return [
        [0. + 0.j, 1. + 0.j],
        [1. + 0.j, 0. + 0.j]
    ]


@constant_gate(strictly_complex=True)
def pauli_y(
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
):
    """Get the Pauli Y gate.

    Args:
        device: If the torch device is not specified, CPU is used.
        dtype: torch.dtype for the result.
            When not specified, torch.complex64 is used.
    """
    return [
        [0., -1.j],
        [1.j, 0.]
    ]


@constant_gate(strictly_complex=True)
def pauli_z(
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
):
    """Get the Pauli Z gate.

    Args:
        device: If the torch device is not specified, CPU is used.

        dtype: torch.dtype for the result.
            When not specified, torch.complex64 is used.
    """
    return [
        [1. + 0.j, 0.j],
        [0.j, -1. + 0.j]
    ]


@constant_gate(strictly_complex=True)
def cnot(
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None
):
    """Get the CNOT gate.

    Args:
        device: If the torch device is not specified, CPU is used.

        dtype: torch.dtype for the result.
            When not specified, torch.complex64 is used.
    """
    return [
        [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
        [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
        [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j],
        [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j]
    ]


# `cx` is an alias for `cnot`.
cx = cnot


@constant_gate(strictly_complex=True)
def cz(
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
):
    """Get the controlled-Z gate.

    Args:
        device: If the torch device is not specified, CPU is used.

        dtype: torch.dtype for the result.
            When not specified, torch.complex64 is used.
    """
    return [
        [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
        [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
        [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
        [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]
    ]

