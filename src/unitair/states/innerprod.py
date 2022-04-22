import torch


def abs_squared(state: torch.Tensor):
    """Compute the vector of measurement probabilities for state.

    For an $N$-dimensional Hilbert space, in a given basis a state
    can be regarded as a vector (x_1, ..., x_N). This function
    simply computes (|x_1|^2, ..., |x_N|^2).

    When state is a batch, the vector of absolute value squared coefficients
    is computed for each batch entry.

    Note:
        This function trivially wraps torch operations. We recommend its usage
        when it improves readability.

    Args:
        state (Tensor): State or batch of states in vector layout. Tensor size
            should be (*batch_dims, N). State can be real or complex.

    Returns:
        Tensor with size (*batch_dims, N) giving all measurement
        probabilities for all states in the batch.
    """
    return (state.conj() * state).real


def norm_squared(state: torch.Tensor):
    """Compute the L^2 norm-squared < state | state >.

    `state` can be a batch of states. The L^2 norm is used in the real and
    complex case.

    When `state` is normalized, norm_squared(state) should return 1.

    Args:
        state (Tensor): State or batch of states in vector layout. Tensor size
            can be (*batch_dims, N) where N is typically 2^(num_qubits). State
            dtype can be real or complex.

    Returns:
        Tensor with size (*batch_dims,) giving the squared norm of every state
        in the batch.
    """
    return torch.sum(abs_squared(state), dim=-1)


def diag_expectation_value(diag_values, state):
    """Get the expectation value of a diagonal matrix.

    Args:
        diag_values: Tensor with size (*batch_dims, 2^n) where n is the
            number of qubits. For a given batch entry, the 2^n elements
            are the diagonal matrix elements of the operator.

        state: State in vector layout.
    """
    return torch.sum(abs_squared(state) * diag_values, dim=-1)


def inner_product(state_1, state_2):
    """Compute < state_1 | state_2 > (the left entry is conjugate-linear).
    """
    return torch.sum(state_1.conj() * state_2, dim=-1)
