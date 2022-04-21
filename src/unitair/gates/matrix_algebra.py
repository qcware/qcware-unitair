from typing import Iterable

import torch


def fuse_single_qubit_operators(
        qubits: Iterable[int],
        operators: Iterable[torch.Tensor],
):
    """Multiply together gates acting on various single qubits.

    Suppose that we have a sequence of single-qubit gates that
    should act, one after the other, on designated qubits.
    If any qubit repeats, then we can matrix multiply the operators
    to reduce to a single matrix acting on each qubit. This function
    performs this operation and collects the "fused" operations
    in a simple dict.

    If the argument `qubits` is [2, 4, 2, 7] and the `operators` is
    a list with length four as in

        A = operators[0]
        B = operators[1]
        C = operators[2]
        D = operators[3],

    then this function will return the dict
        {
            2: CA
            4: B
            7: D
        }

    where CA is the matrix multiplication of torch.matmul(C, A).

    Args:
        qubits: Iterable of integers giving the qubits that gates act on.
        operators: Iterable of Tensor objects specifying single qubit
            operators. The size should be (*batch_dims, 2, 2).

    Returns:
        Dict mapping qubits to act on to a fused gate.
    """
    qubits_to_fused_ops = dict()
    for q, op in zip(qubits, operators):
        if q in qubits_to_fused_ops:
            qubits_to_fused_ops[q] = torch.matmul(op, qubits_to_fused_ops[q])
        else:
            qubits_to_fused_ops[q] = op

    return qubits_to_fused_ops
