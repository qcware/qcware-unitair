from typing import Iterable, Tuple, Union, Optional
import torch
from unitair.states import Field
import unitair.states as states


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


# TODO: we probably don't need this anymore because of multi-qubit action.
def act_first_qubit(
        single_qubit_operator: torch.Tensor,
        state: torch.Tensor,
        field: Field = Field.COMPLEX
):
    """Apply a single qubit gate to the first qubit in state."""
    field = Field(field.lower())
    num_qubits = states.count_qubits(state)

    state_tensor = states.to_tensor_layout(state)
    state_tensor = act_first_qubit_tensor(
        single_qubit_operator, state_tensor, field
    )
    return states.to_vector_layout(state_tensor, num_qubits=num_qubits)


def act_first_qubit_tensor(
        single_qubit_operator: torch.Tensor,
        state_tensor: torch.Tensor,
        field: Field = Field.COMPLEX
):
    """Apply a single qubit operator on the first qubit.

    TODO: apply batching. This may be easier for act_last_qubit...
    """
    field = Field(field.lower())

    def act(operator, tensor):
        return torch.einsum('ab, b... -> a...', operator, tensor)

    if field is Field.REAL:
        return act(single_qubit_operator, state_tensor)

    elif field is Field.COMPLEX:
        real_tens = (
                act(single_qubit_operator[0], state_tensor[0])
                - act(single_qubit_operator[1], state_tensor[1])
        )
        imag_tens = (
                act(single_qubit_operator[0], state_tensor[1])
                + act(single_qubit_operator[1], state_tensor[0])
        )
        return torch.stack((real_tens, imag_tens), dim=0)
    else:
        assert False, f"Impossible enumeration{field}"


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
    state_tensor = act_first_qubits_tensor(operator, state_tensor, field,
                                    gate_num_qubits=gate_num_qubits)
    return states.to_vector_layout(state_tensor, num_qubits=num_qubits)


def act_first_qubits_tensor(
        operator: torch.Tensor,
        state_tensor: torch.Tensor,
        field: Field = Field.COMPLEX,
        gate_num_qubits: Optional[int] = None,
):
    field = Field(field.lower())
    if gate_num_qubits is None:
        gate_num_qubits = states.count_qubits_gate_matrix(operator)

    gate_dim = 2 ** gate_num_qubits
    def act(operator, tensor):
        old_size = tensor.size()
        new_size = (gate_dim,) + (tensor.dim() - gate_num_qubits) * (2,)
        tensor_view = tensor.view(new_size)
        result = torch.einsum('ab, b... -> a...', operator, tensor_view)
        return result.view(old_size)

    if field is Field.REAL:
        return act(operator, state_tensor)

    elif field is Field.COMPLEX:
        real_tens = (
                act(operator[0], state_tensor[0])
                - act(operator[1], state_tensor[1])
        )
        imag_tens = (
                act(operator[0], state_tensor[1])
                + act(operator[1], state_tensor[0])
        )
        return torch.stack((real_tens, imag_tens), dim=0)
    else:
        assert False, f"Impossible enumeration{field}"


def apply_all_qubits(
        gate: torch.Tensor,
        state: torch.Tensor,
        field: Field = Field.COMPLEX
) -> torch.Tensor:
    """Apply the same single-qubit operator to each qubit of specified state.

    Args:
        gate: Tensor with size (2, 2) or (2, 2, 2) defining a real or
            complex 2 by 2 matrix which will act on every qubit.
            In complex case, the first dimension is for the real and
            imaginary parts:
                matrix = matrix[0] + i matrix[1].

        state: State in vector layout. This means that the state is a
            tensor with size (2^num_bits) or (2, 2^num_bits) for the
            real or complex cases respectively.
            In the complex case with size (2, 2^num_bits), the first dimension
            is for the real and imaginary parts:
                state = state[0] + i state[1].

        field:
    """
    num_qubits = states.count_qubits(state)
    state_tensor = states.to_tensor_layout(state)
    state_tensor = apply_all_qubits_tensor(
        gate, state_tensor, num_qubits=num_qubits, field=field
    )
    return states.to_vector_layout(state_tensor, num_qubits=num_qubits)


def apply_all_qubits_tensor(
        gate: torch.Tensor,
        state_tensor: torch.Tensor,
        num_qubits: int,
        field: Field = Field.COMPLEX
):
    """Apply one single-qubit operator to each qubit of state in tensor layout.

    Args:
        gate: Tensor with size (2, 2) or (2, 2, 2) defining a real or
            complex 2 by 2 matrix which will act on every qubit.
            In complex case, the first dimension is for the real and
            imaginary parts:
                matrix = matrix[0] + i matrix[1].

        state_tensor: State in tensor layout. This means that the state is a
            tensor with size (2, 2, ..., 2) where the number of 2's is
            num_qubits or num_qubits + 1 for the real and complex cases.

        num_qubits: The number of qubits for the state.

        field:
    """
    unitary_error_message = (
        "apply_all_qubits is meant for applying a 2 by 2 matrix\n"
        "on each qubit for a vector on num_bits qubits. Shape should be\n"
        "(2, 2, 2) in the complex case or (2, 2) in the real case. "
        "In the complex case,\nthe first dimension is for the real and "
        "imaginary parts."
    )
    field = Field(field.lower())
    if field is Field.REAL:
        gate_size = [2, 2]
    else:
        gate_size = [2, 2, 2]

    if gate_size != list(gate.size()):
        raise ValueError(unitary_error_message)

    state_tensor = act_first_qubit_tensor(gate, state_tensor, field=field)
    for i in range(1, num_qubits):
        state_tensor = swap_tensor(
            state_tensor, qubit_pair=(0, i), num_qubits=num_qubits
        )
        state_tensor = act_first_qubit_tensor(gate, state_tensor, field=field)

    state_tensor = roll_qubits_tensor(
        state_tensor, num_qubits, num_steps=-1
    )

    return state_tensor


def apply_to_qubits(
        gates: Iterable[torch.Tensor],
        qubits: Iterable[int],
        state: torch.Tensor,
        field: Union[Field, str] = Field.COMPLEX
):
    """Apply single qubit gates to specified qubits of state."""
    num_qubits = states.count_qubits(state)
    state_tensor = states.to_tensor_layout(state)
    state_tensor = apply_to_qubits_tensor(
        gates, qubits, state_tensor, num_qubits, field
    )
    return states.to_vector_layout(state_tensor, num_qubits=num_qubits)


def apply_to_qubits_tensor(
        operators: Union[Iterable[torch.Tensor], torch.Tensor],
        qubits: Union[Iterable[int], int],
        state_tensor: torch.Tensor,
        num_qubits: int,
        field: Field = Field.COMPLEX
):
    """Apply single qubit gates to specified qubits of state in tensor layout.
    """
    if type(qubits) == int and type(operators) == torch.Tensor:
        return apply_to_qubits([operators], [qubits], state_tensor, field)
    field = Field(field.lower())

    for gate, q in zip(operators, qubits):
        state_tensor = swap_tensor(state_tensor, (0, q), num_qubits)
        state_tensor = act_first_qubit_tensor(gate, state_tensor, field)
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
    field = Field(field.lower())
    num_steps = num_steps % num_qubits
    if num_steps == 0:
        return state_tensor

    if field is Field.REAL:
        identity = list(range(num_qubits))
        perm = identity[-num_steps:] + identity[:-num_steps]
    else:
        identity = list(range(1, num_qubits + 1))
        perm = [0] + identity[-num_steps:] + identity[:-num_steps]

    return state_tensor.permute(perm)


if __name__ == '__main__':
    from unitair import get_default_device
    from unitair import gates
    import time
    selected_device = get_default_device()

    print('selected device: ', selected_device)


    def run_apply_all(n):
        rot = gates.exp_x(.3)
        state = torch.rand(2, 2**n)
        t = time.time()
        apply_all_qubits(rot, state)
        return time.time()-t

    def run_apply_to(n):
        rot = gates.exp_x(.3)
        state = torch.rand(2, 2**n)
        t = time.time()
        apply_to_qubits(
            gates=[rot for _ in range(n)],
            qubits=range(n),
            state=state
        )

        return time.time()-t

    #
    # rot = torch.tensor(
    #     [
    #         [[0.8718,  0.0000],
    #          [0.0000,  0.8718]],
    #
    #         [[0.0000, -0.4899],
    #          [-0.4899,  0.0000]]
    #     ],
    #     device=selected_device
    # )
    # state_ = torch.tensor(
    #     [[0.1316, 0.1930, 0.1382, 0.0934, 0.3797, 0.2206, 0.3941, 0.0504],
    #      [0.1461, 0.1924, 0.3403, 0.4453, 0.1680, 0.3133, 0.1078, 0.2376]],
    #     device=selected_device)
    #
    # real_rot = torch.tensor(
    #     [[[0.871766, -0.489922], [0.489922, 0.871766]], [[0, 0], [0, 0]]],
    #     device=selected_device
    # )
    #
    # real_state_ = torch.tensor(
    #     [[0.1316, 0.1930, 0.1382, 0.0934, 0.3797, 0.2206, 0.3941, 0.0504],
    #      [0, 0, 0, 0, 0, 0, 0, 0.]],
    #     device=selected_device)
    #
    # apply_all_qubits(rot, state_)