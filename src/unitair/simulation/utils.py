import torch
from unitair.states import Field


def count_gate_batch_dims(gate: torch.Tensor, field: Field = Field.COMPLEX):
    """Compute the number of batch dimensions for a given gate batch.

    Args:
        gate: Tensor with size (*batch_dims, 2, 2^k, 2^k) in the complex
            case or (*batch_dims, 2^k, 2^k) in the real case. Here, k is
            the number of qubits on which the gate acts.

        field: Specification of the Field (complex or real) for the gate.
    """
    if field is Field.COMPLEX:
        out = gate.dim() - 3
    else:
        out = gate.dim() - 2

    if out < 0:
        raise RuntimeError(
            f'Gate with size {gate.size()} is incorrectly shaped for a '
            f'{field.value} operator batch.'
        )
    return out


def gate_batch_size(gate: torch.Tensor, field: Field = Field.COMPLEX):
    """Get the batch shape for given gate batch.

    Args:
        gate: Tensor with size (*batch_dims, 2, 2^k, 2^k) in the complex
            case or (*batch_dims, 2^k, 2^k) in the real case. Here, k is
            the number of qubits on which the gate acts.

        field: Specification of the Field (complex or real) for the gate.
    """
    num_batch_dims = count_gate_batch_dims(gate, field)
    return gate.size()[:num_batch_dims]
