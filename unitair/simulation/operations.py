from typing import Iterable, Tuple, Union, Optional
import torch
from unitair.states import Field
import unitair.states as states
from .utils import count_gate_batch_dims
from unitair.utils import permutation_to_front, inverse_list_permutation
import numpy as np

# TODO: should most validation be in the "front-end" vector functions or should
#  it be in the tensor functions? Many cases here actually validate in both.


def apply_phase(angles: torch.Tensor, state: torch.Tensor):
    """Multiply the jth component of state by e^(-i angles_j).

    This function is inapplicable for real vector spaces.

    If angles and state have batch dimensions, then both
    batch dimensions should be identical. In this case, each batch
    entry of angle acts on each corresponding state batch entry.

    If angles does not have batch dimensions but state does, then
    the same angles are applied to every state in the batch.

    # TODO: clarify more possible batching cases.

    Args:
        angles: Size (*batch dims, Hilbert space dim)
        state: Size (*batch dims, 2, Hilbert space dim)
    """
    cos = angles.cos()
    sin = angles.sin()
    real, imag = states.real_imag(state)
    return torch.stack((
        cos * real + sin * imag,
        -sin * real + cos * imag
    ), dim=-2)


# TODO: documentation for apply_operator. This is tested with operator and
#   state batching in the usual two patterns.
def apply_operator(
        operator: torch.Tensor,
        qubits: Iterable[int],
        state: torch.Tensor,
        field: Union[Field, str] = Field.COMPLEX
):
    field = Field(field.lower())
    num_qubits = states.count_qubits(state)
    qubits = list(qubits)
    if not set(qubits).issubset(range(num_qubits)):
        raise ValueError(
            f'qubits={qubits} is not consistent with state vector with '
            f'{num_qubits} qubits.'
        )
    op_num_qubits = states.count_qubits_gate_matrix(operator)
    if len(qubits) != op_num_qubits:
        raise ValueError(
            f'Cannot apply operator with {op_num_qubits} to the {len(qubits)} '
            f'qubit sequence {qubits}.'
        )

    state_tensor = states.to_tensor_layout(state)
    state_tensor = apply_operator_tensor(
        operator=operator,
        qubits=qubits,
        state_tensor=state_tensor,
        num_qubits=num_qubits,
        field=field,
        operator_num_qubits=op_num_qubits
    )
    return states.to_vector_layout(state_tensor, num_qubits=num_qubits)


def apply_operator_tensor(
        operator: torch.Tensor,
        qubits: Iterable[int],
        state_tensor: torch.Tensor,
        num_qubits: int,
        field: Field = Field.COMPLEX,
        operator_num_qubits: Optional[int] = None
):
    qubits = list(qubits)
    if operator_num_qubits is None:
        operator_num_qubits = states.count_qubits_gate_matrix(operator)
    perm = permutation_to_front(num_qubits, qubits)
    inv_perm = inverse_list_permutation(perm)

    # Permute qubits to the front
    state_tensor = permute_qubits_tensor(
        permutation=perm,
        state_tensor=state_tensor,
        num_qubits=num_qubits,
        contiguous_output=True
    )

    # Apply operator on the front consecutive qubits
    state_tensor = act_first_qubits_tensor(
        operator=operator,
        state_tensor=state_tensor,
        num_qubits=num_qubits,
        field=field,
        gate_num_qubits=operator_num_qubits
    )

    # Invert the earlier permutation
    return permute_qubits_tensor(
        permutation=inv_perm,
        state_tensor=state_tensor,
        num_qubits=num_qubits,
        contiguous_output=True
    )


def act_last_qubit(
        single_qubit_operator: torch.Tensor,
        state: torch.Tensor,
        field: Union[Field, str] = Field.COMPLEX
) -> torch.Tensor:
    """Apply an operator to the last qubit.

    Args:
        single_qubit_operator: Tensor with size (2, 2, 2) or (2, 2) in
            the complex and real cases respectively. For the complex case,
            the first dimension is for real and imaginary parts.

        state: State or batch of states in vector layout.

        field: Specification of the field.
    """
    field = Field(field.lower())
    num_qubits = states.count_qubits(state)
    state_tensor = states.to_tensor_layout(state)
    state_tensor = act_last_qubit_tensor(
        single_qubit_operator, state_tensor, num_qubits=num_qubits, field=field
    )
    return states.to_vector_layout(state_tensor, num_qubits)


def act_last_qubit_tensor(
        single_qubit_operator: torch.Tensor,
        state_tensor: torch.Tensor,
        num_qubits: int,
        field: Field = Field.COMPLEX
) -> torch.Tensor:
    """Apply an operator to the last qubit of a state in tensor layout.

        Args:
            single_qubit_operator: Tensor with size (2, 2, 2) or (2, 2) in
                the complex and real cases respectively. For the complex case,
                the first dimension is for real and imaginary parts.

            state_tensor: State or batch of states in tensor layout.

            num_qubits: Number of qubits.

            field: Specification of the field.
        """
    field = Field(field.lower())

    def act(matrix, tensor):
        """Contract matrix with the last index of tensor."""
        return torch.einsum('ab, ...b -> ...a', matrix, tensor)

    if field is Field.REAL:
        if single_qubit_operator.dim() != 2:
            raise ValueError(
                f'To act on a real vector space, expected operator size '
                f'(2, 2), but got {single_qubit_operator.dim()}.'
            )
        return act(single_qubit_operator, state_tensor)

    elif field is Field.COMPLEX:
        if single_qubit_operator.dim() != 3:
            raise ValueError(
                f'Expected operator size '
                f'(2, 2, 2), but got {single_qubit_operator.dim()}.'
            )
        real_op = single_qubit_operator[0]
        imag_op = single_qubit_operator[1]

        real_imag = states.real_imag_tensor(state_tensor, num_qubits)
        real_state, imag_state = real_imag

        real_tens = (act(real_op, real_state) - act(imag_op, imag_state))
        imag_tens = (act(real_op, imag_state) + act(imag_op, real_state))
        return torch.stack((real_tens, imag_tens), dim=-(num_qubits+1))
    else:
        assert False, f"Impossible enumeration{field}"


def act_first_qubits(
        operator: torch.Tensor,
        state: torch.Tensor,
        field: Union[Field, str] = Field.COMPLEX
):
    """Apply a multi-qubit gate to the first qubits of a state."""
    field = Field(field.lower())
    num_qubits = states.count_qubits(state)
    gate_num_qubits = states.count_qubits_gate_matrix(operator)
    if num_qubits < gate_num_qubits:
        raise ValueError(
            f'Attempted to apply a {gate_num_qubits}-qubit gate to '
            f'{num_qubits} qubit(s).'
        )
    state_tensor = states.to_tensor_layout(state)
    state_tensor = act_first_qubits_tensor(
        operator=operator,
        state_tensor=state_tensor,
        num_qubits=num_qubits,
        field=field,
        gate_num_qubits=gate_num_qubits
    )
    return states.to_vector_layout(state_tensor, num_qubits=num_qubits)


def act_first_qubits_tensor(
        operator: torch.Tensor,
        state_tensor: torch.Tensor,
        num_qubits: int,
        field: Field = Field.COMPLEX,
        gate_num_qubits: Optional[int] = None,
):
    """Apply operator on first consecutive qubits of a state in tensor layout.

    `operator` represents an operator or batch of operators that act on
    k qubits (with k <= the number of qubits for the state).

    When used without batches, `operator` is a single-qubit operator specified
    by a tensor of size (2, 2^k, 2^k) in the complex
    case and (2^k, 2^k) in the real case. `state_tensor` is a state
    in tensor layout for n qubits. The operator acts on the first k consecutive
    qubits. A new state in tensor layout is then returned.

    Both operator and state_tensor can have batch dimensions, but batch
    dimensions must be compatible.

    Batching cases:
       `operator` and `state_tensor` have the same batch dimensions:
           In this case, each batch entry of `operator` acts on the
           corresponding entry of `state_tensor`.

       `operator` has no batch dimensions but `state_tensor` does:
           In this case, the same operator acts on every state_tensor
           in the batch.
    """
    field = Field(field.lower())
    if gate_num_qubits is None:
        gate_num_qubits = states.count_qubits_gate_matrix(operator)

    gate_dim = 2 ** gate_num_qubits

    state_n_batch_dims = states.count_batch_dims_tensor(
        state_tensor, num_qubits, field
    )
    op_n_batch_dims = count_gate_batch_dims(operator, field)
    state_batch_dims = state_tensor.size()[:state_n_batch_dims]

    state_tensor = states.subset_roll_to_back(state_tensor, state_n_batch_dims)
    operator = states.subset_roll_to_back(operator, op_n_batch_dims)

    def act(op, tensor):
        old_size = tensor.size()
        new_size = (gate_dim,) + (num_qubits - gate_num_qubits) * (2,) + state_batch_dims
        tensor_view = tensor.view(new_size)
        result = torch.einsum('ab..., b... -> a...', op, tensor_view)
        return result.view(old_size)

    if field is Field.REAL:
        result_batch_flipped = act(operator, state_tensor)

    elif field is Field.COMPLEX:
        real_tens = (
                act(operator[0], state_tensor[0])
                - act(operator[1], state_tensor[1])
        )
        imag_tens = (
                act(operator[0], state_tensor[1])
                + act(operator[1], state_tensor[0])
        )
        result_batch_flipped = torch.stack((real_tens, imag_tens), dim=0)
    else:
        assert False, f"Impossible enumeration{field}"

    return states.subset_roll_to_front(
        result_batch_flipped, state_n_batch_dims)


def apply_all_qubits(
        operator: torch.Tensor,
        state: torch.Tensor,
        field: Field = Field.COMPLEX
) -> torch.Tensor:
    """Apply the same single-qubit operator to each qubit of specified state.

    Batching cases:
        `operator` and `state` have the same batch dimensions:
            In this case, each batch entry of `operator` acts on the
            first qubit of the corresponding batch entry of `state`.

        `operator` has no batch dimensions but `state` does:
            In this case, the same operator acts on every state
            in the batch. In fact, this means that the same operator acts
            on every qubit of every entry in `state`

    Args:
        operator: Tensor with size (*batch_dims, 2, 2) or
            (*batch_dims, 2, 2, 2) defining a real or complex 2 by 2 matrix
            which will act on every qubit. In complex case, the first dimension
            is for the real and imaginary parts. For each batch entry `matrix`,
            this means:
                matrix = matrix[0] + i matrix[1].

        state: State in vector layout. This means that the state is a
            tensor with size (*batch_dims, 2^num_bits,) or
            (*batch_dims, 2, 2^num_bits) for the real or complex cases
            respectively. In the complex case, the first dimension of each
            batch entry is for the real and imaginary parts:
                state = state[0] + i state[1]
            (where state has no batch dimensions).

        field: Specifies whether the Hilbert space is real or complex.
    """
    if states.count_qubits_gate_matrix(operator) != 1:
        raise ValueError(
            f'Expected operator on 1 qubit, found a '
            f'{states.count_qubits_gate_matrix(operator)} qubit operator.'
        )

    num_qubits = states.count_qubits(state)
    state_tensor = states.to_tensor_layout(state)
    state_tensor = apply_all_qubits_tensor(
        operator, state_tensor, num_qubits=num_qubits, field=field
    )
    return states.to_vector_layout(state_tensor, num_qubits=num_qubits)


def apply_all_qubits_tensor(
        operator: torch.Tensor,
        state_tensor: torch.Tensor,
        num_qubits: int,
        field: Field = Field.COMPLEX
):
    """Apply one single-qubit operator to each qubit of state in tensor layout.

    Batching cases:
        `operator` and `state_tensor` have the same batch dimensions:
            In this case, each batch entry of `operator` acts on the
            first qubit of the corresponding batch entry of `state_tensor`.

        `operator` has no batch dimensions but `state_tensor` does:
            In this case, the same operator acts on every state
            in the batch. In fact, this means that the same operator acts
            on every qubit of every entry in `state_tensor`

    Args:
        operator: Tensor with size (*batch_dims, 2, 2) or
            (*batch_dims, 2, 2, 2) defining a real or complex 2 by 2 matrix
            which will act on every qubit. In complex case, the first dimension
            is for the real and imaginary parts. For each batch entry `matrix`,
            this means:
                matrix = matrix[0] + i matrix[1].

        state_tensor: State in tensor layout. Size is
            (*batch_dims, 2, 2, ...,2, 2) where the number of 2's is
            num_qubits or num_qubits + 1 for the real and complex cases.

        num_qubits: The number of qubits for the quantum state.

        field: Specifies whether the Hilbert space is real or complex.
    """
    field = Field(field.lower())

    state_tensor = act_first_qubits_tensor(
        operator=operator,
        state_tensor=state_tensor,
        num_qubits=num_qubits,
        field=field,
        gate_num_qubits=1
    )
    for i in range(1, num_qubits):
        state_tensor = swap_tensor(
            state_tensor, qubit_pair=(0, i), num_qubits=num_qubits
        )
        state_tensor = act_first_qubits_tensor(
            operator, state_tensor, num_qubits=num_qubits, field=field,
            gate_num_qubits=1
        )

    state_tensor = roll_qubits_tensor(
        state_tensor, num_qubits, num_steps=-1, field=field
    )

    return state_tensor


def apply_to_qubits(
        operators: Iterable[torch.Tensor],
        qubits: Iterable[int],
        state: torch.Tensor,
        field: Union[Field, str] = Field.COMPLEX
):
    """Apply single-qubit gates to specified qubits of state in vector layout.

    This function applies an iterable of gates to an iterable of respective
    qubits.
    """
    num_qubits = states.count_qubits(state)
    state_tensor = states.to_tensor_layout(state)
    state_tensor = apply_to_qubits_tensor(
        operators, qubits, state_tensor, num_qubits, field
    )
    return states.to_vector_layout(state_tensor, num_qubits=num_qubits)


def apply_to_qubits_tensor(
        operators: Union[Iterable[torch.Tensor]],
        qubits: Union[Iterable[int]],
        state_tensor: torch.Tensor,
        num_qubits: int,
        field: Field = Field.COMPLEX
):
    """Apply single qubit gates to specified qubits of state in tensor layout.
    """
    field = Field(field.lower())

    for gate, q in zip(operators, qubits):
        state_tensor = swap_tensor(state_tensor, (0, q), num_qubits)
        state_tensor = act_first_qubits_tensor(
            gate, state_tensor, num_qubits, field, gate_num_qubits=1
        )
        state_tensor = swap_tensor(state_tensor, (0, q), num_qubits)
    return state_tensor


def swap(state: torch.Tensor, qubit_pair: Tuple[int, int]):
    """Swap a pair of qubits for a state.

    This operation is more natural for states in tensor layout, and in fact
    this function wraps `swap_tensor`.
    """
    num_qubits = states.count_qubits(state)
    state_tensor = states.to_tensor_layout(state)
    state_tensor = swap_tensor(
        state_tensor, qubit_pair, num_qubits
    )
    return states.to_vector_layout(state_tensor, num_qubits)


def swap_tensor(
        state_tensor: torch.Tensor,
        qubit_pair: Tuple[int, int],
        num_qubits: int,
):
    """Swap a pair of qubits for a state in tensor layout.

    For the corresponding operation of states in vector format, see
    the function `swap_qubits`.
    """
    if qubit_pair[0] == qubit_pair[1]:
        return state_tensor
    q1, q2 = states.shapes.get_qubit_indices(
        index=qubit_pair,
        state_tensor=state_tensor,
        num_qubits=num_qubits
    )
    return state_tensor.transpose(q1, q2)


def roll_qubits(
        state: torch.Tensor, num_steps=1, field: Field = Field.COMPLEX
):
    """Perform a cyclic permutation of qubits for a state.

    This operation is more natural for states in tensor layout, and
    this function wraps `roll_qubits_tensor`.

    This function rolls qubits forward by the specified number of
    steps. For example, is psi is a quantum state with

    < 0100 | psi > = c

    where c is some complex number, then the calling
    roll_qubits(psi, num_steps=1) will return a new state psi' that will have

    < 0010 | psi' > = c.
    """
    field = Field(field.lower())
    num_qubits = states.count_qubits(state)

    state_tensor = states.to_tensor_layout(state)
    state_tensor = roll_qubits_tensor(
        state_tensor, num_qubits, num_steps, field
    )
    return states.to_vector_layout(state_tensor, num_qubits)


def roll_qubits_tensor(
        state_tensor, num_qubits, num_steps=1, field: Field = Field.COMPLEX
):
    """Perform a cyclic permutation of qubits for a state in tensor layout.

    If the initial tensor is psi, the output of this function
    is a new tensor psi_rolled which is related to psi by

    psi_rolled[a_0, a_1, ..., a_{n-1}]
        = psi[a_k, a_{k+1}, ..., a_{n-1}, a_0, ..., a_{k-1}].

    Note that in this formula is for the real case. In the complex case
    there is one more initial index which is not altered by the
    permutation.

    Args:
        state_tensor: state in tensor layout to be permuted.

        num_qubits: Number of qubits for the state.

        num_steps: Number of indices to cycle.
    """
    # TODO: document new batching
    num_batch_dims = states.count_batch_dims_tensor(
        state_tensor, num_qubits, field)
    field = Field(field.lower())
    num_steps = num_steps % num_qubits
    if num_steps == 0:
        return state_tensor
    if field is Field.REAL:
        identity = list(range(num_batch_dims, num_batch_dims + num_qubits))
        perm = identity[-num_steps:] + identity[:-num_steps]
    else:
        identity = list(range(1 + num_batch_dims, num_qubits + 1 + num_batch_dims))
        perm = [num_batch_dims] + identity[-num_steps:] + identity[:-num_steps]
    perm = list(range(num_batch_dims))+perm
    return state_tensor.permute(perm)


def permute_qubits(
        permutation: Iterable[int],
        state_vector: torch.Tensor
):
    """Permute qubits for a state in vector layout.

    If `state_vector` has batch dimensions, then the same permutation is
    applied to all batch entries. In particular, this function does not
    permute batch indices (use torch.permute for that).

    Args:
        permutation: Sequence of integers defining a permutation among qubits.
        state_vector: State in vector layout.
    """
    num_qubits = states.count_qubits(state_vector)
    state_tensor = states.to_tensor_layout(state_vector)
    permuted_state_tensor = permute_qubits_tensor(
        permutation=permutation,
        state_tensor=state_tensor,
        num_qubits=num_qubits,
    )
    return states.to_vector_layout(permuted_state_tensor, num_qubits=num_qubits)


def permute_qubits_tensor(
        permutation: Iterable[int],
        state_tensor: torch.Tensor,
        num_qubits: int,
        contiguous_output: bool = False
):
    """Permute qubits for a state in tensor layout.

    If `state_tensor` has batch dimensions, then the same permutation is
    applied to all batch entries. In particular, this function does not
    permute batch indices (use torch.permute for that).

    Args:
        permutation: Sequence of integers defining a permutation among qubits.
        state_tensor: State in vector layout.
        num_qubits: Number of qubits for the state.
        contiguous_output: When True, returns a contiguous copy of the result.
    """
    # non_qubit_dims includes the complex dimension in the complex case.
    non_qubit_dims = state_tensor.dim() - num_qubits
    identity_perm = tuple(i for i in range(non_qubit_dims))

    permutation = tuple(i + non_qubit_dims for i in permutation)

    state_tensor = torch.permute(state_tensor, identity_perm + permutation)
    if not contiguous_output:
        return state_tensor
    else:
        return state_tensor.contiguous()


def multi_cz(
        qubit_pairs: torch.Tensor,
        state_vector: torch.Tensor,
        num_bits_memory_cutoff: Optional[int] = None
):
    """Apply CZ gates to specified qubit pairs.

    This function can either apply a single CZ gate or multiple gates
    to arbitrary pairs of qubits. For example, if the parameter
    `qubit_pairs` is set to torch.tensor([0, 1]), then qubits 0 and 1
    are used as the (interchangeable) control and target and the gate
    CZ(control=0, target=1) is applied to `state_vector`. However,
    if instead we have

        qubit_pairs=torch.tensor([[0, 1], [0, 2]])

    then the gates CZ(control=0, target=1) and CZ(control=0, target=2) are
    applied to `state_vector`. (Note that the two operators commute.)

    CUDA usage note:
        This function can benefit substantially from GPU acceleration.
        This is invoked when the arguments `qubit_pairs` and `state_vector`
        are CUDA tensors. See argument `num_bits_memory_cutoff` for a
        parameter that can be tuned to optimize GPU performance.

    Args:
        qubit_pairs: Tensor with size (2,) or (num_pairs, 2) giving a collection
            of control-target pairs.

        state_vector: State in vector layout upon which to apply CZ gates.

        num_bits_memory_cutoff: This advanced parameter can be tuned to avoid
            running out of CUDA memory in the case where multiple qubit pairs
            are used. Normally multiple qubit pairs are batched and processed
            with parallelization, but if log_2(num_pairs) + num_qubits exceeds
            num_bits_memory_cutoff, we instead use a for loop. When this
            parameter is None, an estimate is used based on device properties.
    """
    num_qubits = states.count_qubits(state_vector)
    dev = qubit_pairs.device
    if qubit_pairs.dim() == 1:
        qubit_pairs = qubit_pairs.view(1, 2)

    if state_vector.device.type != qubit_pairs.device.type:
        raise ValueError(
            'state_vector and qubit_qubit_pairs must have same device.'
        )

    elif qubit_pairs.dim() != 2:
        raise ValueError('qubit_pairs must have size (2,) or (num_pairs, 2).')
    if (qubit_pairs >= num_qubits).any():
        raise ValueError(
            'Control/target indices for CZ gate must be less than num_bits.'
        )

    num_pairs = qubit_pairs.size()[0]
    if num_bits_memory_cutoff is None:
        if dev.type == 'cpu':
            num_bits_memory_cutoff = np.inf
        else:
            mem = torch.cuda.get_device_properties(dev).total_memory
            num_bits_memory_cutoff = mem - 5

    use_batched_pairs = (
            num_qubits + np.log2(num_pairs) < num_bits_memory_cutoff
    )

    # control and target are interchangeable.
    control = qubit_pairs[:, 0]
    target = qubit_pairs[:, 1]
    if (control == target).any():
        raise ValueError('Control and target qubits are not distinct.')
    pow_c = num_qubits - control - 1
    pow_t = num_qubits - target - 1
    critical_int = 2 ** pow_c + 2 ** pow_t

    all_ints = torch.arange(2 ** num_qubits, device=dev)

    if use_batched_pairs:
        match = torch.bitwise_and(
            all_ints.view(-1, 1),
            critical_int.view(1, -1)
        ) == critical_int

        phases = torch.prod(1 - 2 * match, dim=-1)
    else:
        phases = torch.ones(2 ** num_qubits, device=dev)
        for crit in critical_int:
            match = torch.bitwise_and(all_ints, crit) == crit
            phases *= 1 - 2 * match

    return phases * state_vector
