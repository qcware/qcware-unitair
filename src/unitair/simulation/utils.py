import torch


def count_gate_batch_dims(gate: torch.Tensor):
    """Compute the number of batch dimensions for a given gate batch.

    Args:
        gate: Tensor with size (*batch_dims, 2^k, 2^k) where k is
            the number of qubits on which the gate acts.
    """
    out = gate.dim() - 2

    if out < 0:
        raise RuntimeError(
            f'Gate with size {gate.size()} is incorrectly shaped for an '
            f'operator batch. Expected size is\n'
            f'  (*optional_batch_dims, 2^k, 2^k)\n'
            f'with k the number of qubits on which the gate acts.'
        )
    return out


def gate_batch_size(gate: torch.Tensor):
    """Get the batch shape for given gate batch.

    Args:
        gate: Tensor with size (*batch_dims, 2^k, 2^k) where k is
            the number of qubits on which the gate acts.
    """
    num_batch_dims = count_gate_batch_dims(gate)
    return gate.size()[:num_batch_dims]
