import warnings
from typing import Iterable, Tuple, Union, Optional
import torch
import unitair.states as states
from .utils import count_gate_batch_dims
from unitair.utils import permutation_to_front, inverse_list_permutation
import numpy as np
from unitair.gates.matrix_algebra import fuse_single_qubit_operators
from ..gates import hadamard

# TODO: should most validation be in the "front-end" vector functions or should
#  it be in the tensor functions? Many cases here actually validate in both.


def apply_phase(angles: torch.Tensor, state: torch.Tensor):
    """Multiply the kth component of state by e^(-i angles_k).

    Batching for this function follows PyTorch multiplication batching.
    Here are some examples:

    If `angles` is a scalar (a Tensor with only one element) and `state`
    is a state in vector layout with arbitrary batch dimensions, then the
    angle is applied to every component of every state.

    If `angles` is a vector with size (2^n) (where n is the number of qubits)
    and `state` is a Tensor with arbitrary batch dimensions, then
    the specified phase angle for each component is applied to each state in
    the batch of states.

    When `angles` has the same batch dimensions as `state`, we apply phase
    angles to states in one-to-one correspondence.

    To apply a batch of phases to a batch of states, but to use the same
    angle for each component of a given state, we can give `angles` size
    (*batch_dims, 1) while `state` will have size (*batch_dims, 2^n).

    Args:
        angles: Size (*angle_batch_dims, Hilbert space dim)
        state: Size (*state_batch_dims, Hilbert space dim)
    """
    phase_factors = torch.exp(-1.j * angles)
    return phase_factors * state


def apply_operator(
        operator: torch.Tensor,
        qubits: Iterable[int],
        state: torch.Tensor,
):
    """Apply an operator to a state in vector layout.

    This is a versatile function for applying arbitrary matrices to
    arbitrary subsets of qubits with optional batching behavior.

    Consider an example of acting with a two-qubit gate
    on the last two qubits of a state for three qubits.
    Let `psi` be state for three qubits with no batch dimensions so that
    `psi` has size (2, 8). (The first dimension with length 2 is for the real
    and complex parts of the state.)

    The operator that wish to apply is a 4 by 4 complex matrix (4 because
    4 = 2^(number of qubits that the gate acts on)). With Unitair
    conventions, this is constructed by a tensor with size (2, 4, 4).
    The first dimension is for the real and imaginary parts, while the
    last two are for the matrix. For example, the CNOT gate is

    operator = torch.tensor(
                 [[[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 0., 1.],
                   [0., 0., 1., 0.]],

                  [[0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.]]])

    To apply our operator to the desired qubits, we use

        apply_operator(operator, qubits=(1, 2), state=psi)

    which makes `operator` act on qubits 1 and 2 (in that order!) in
    the state `psi`. Note that qubit counting starts at 0, so the operator
    acts on the middle and final qubit. We could have used qubits=(0, 1)
    to act on the first pair, and we could have used qubits=(1, 0) to act
    on the first pair but in the reversed order.

    There are three batch structures allowed for the inputs with different
    resulting behaviors. These are very important to understand to get
    the benefit of vectorization and CUDA, it is is available.

    1. `operator` and `state` have identical batch dimension. In this case
        each operator will act on each corresponding state. The output
        will have the size (*batch, 2, 2^n) when the field is complex.
        Each batch entry of the output is the action of the corresponding
        batch entry of `operator` on the batch entry of `state`.
        Note that completely unbatched situation is a special case of this.

    2. `operator` has no batch dimension but `state` does. In this case
        there is one operator that acts on a batch of states. The output
        will have size (*batch, 2, 2^n) when the field is complex. Each
        batch entry of the output is the action of the one operator
        on the corresponding batch entry of `state`.

    3. `operator` has dimensions but `state` does not. In this case
        there are many operators that act on a single state. This function
        computes the actions of all of these operators on the one state
        in parallel--not in series. We don't compose operators
        and apply one after the other as in a quantum circuit.
        The output will have size (*batch, 2, 2^n) when the field is complex.
        Each batch entry of the output is the action of the corresponding
        operator on `state`, the one initial state.

    Args:
        operator: "Matrix" with size (*operator_batch, 2^k, 2^k). Here, k is
        the number of qubits that the operator acts on.

        qubits: Sequence of qubits for the operator to act on. This should
            have length k (same k as above). Note that the order of items in
            `qubits` is important.

        state: State in vector layout with size (*state_batch, 2^n, 2^n)
            in the complex case and (*state_batch, 2^n, 2^n) in the real case.
            Here, n is the number of qubits.
    """
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
        operator_num_qubits=op_num_qubits
    )
    return states.to_vector_layout(state_tensor, num_qubits=num_qubits)


def apply_operator_tensor(
        operator: torch.Tensor,
        qubits: Iterable[int],
        state_tensor: torch.Tensor,
        num_qubits: int,
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
) -> torch.Tensor:
    """Apply an operator to the last qubit.

    Args:
        single_qubit_operator: Tensor with size (2, 2, 2) or (2, 2) in
            the complex and real cases respectively. For the complex case,
            the first dimension is for real and imaginary parts.

        state: State or batch of states in vector layout.
    """
    warnings.warn(
        'act_last_qubit and act_last_qubit_tensor are outdated. These\n'
        'functions will be removed from Unitair in a later release or may\n'
        'be revised to meet our current standards.\n'
        'Please consider using apply_operator instead.'
    )
    num_qubits = states.count_qubits(state)
    state_tensor = states.to_tensor_layout(state)
    state_tensor = act_last_qubit_tensor(
        single_qubit_operator, state_tensor
    )
    return states.to_vector_layout(state_tensor, num_qubits)


def act_last_qubit_tensor(
        single_qubit_operator: torch.Tensor,
        state_tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply an operator to the last qubit of a state in tensor layout.

        Args:
            single_qubit_operator: Tensor with size (2, 2, 2) or (2, 2) in
                the complex and real cases respectively. For the complex case,
                the first dimension is for real and imaginary parts.

            state_tensor: State or batch of states in tensor layout.
        """
    def act(matrix, tensor):
        """Contract matrix with the last index of tensor."""
        return torch.einsum('ab, ...b -> ...a', matrix, tensor)

    return act(single_qubit_operator, state_tensor)


def act_first_qubits(
        operator: torch.Tensor,
        state: torch.Tensor,
):
    """Apply a multi-qubit gate to the first qubits of a state."""
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
        gate_num_qubits=gate_num_qubits
    )
    return states.to_vector_layout(state_tensor, num_qubits=num_qubits)


def act_first_qubits_tensor(
        operator: torch.Tensor,
        state_tensor: torch.Tensor,
        num_qubits: int,
        gate_num_qubits: Optional[int] = None,
):
    """Apply operator on first consecutive qubits of a state in tensor layout.

    `operator` represents an operator or batch of operators that act on
    k qubits (with k <= the number of qubits for the state).

    When used without batches, `operator` is a single-qubit operator specified
    by a tensor of size (2^k, 2^k). `state_tensor` is a state
    in tensor layout for n qubits. The operator acts on the first k consecutive
    qubits. A new state in tensor layout is then returned.

    Both operator and state_tensor can have batch dimensions, but batch
    dimensions must be compatible.

    Valid batch structures:
       `operator` and `state_tensor` have the same batch dimensions:
           In this case, each batch entry of `operator` acts on the
           corresponding entry of `state_tensor`.

       `operator` has no batch dimensions but `state_tensor` does:
           In this case, the same operator acts on every state_tensor
           in the batch.

        `operator` has batch dimensions but `state_tensor` does not:
            In this case, each operator acts on the same state.
    """
    if gate_num_qubits is None:
        gate_num_qubits = states.count_qubits_gate_matrix(operator)

    gate_dim = 2 ** gate_num_qubits

    # Determine the batch structure for the state(s) and operator(s).
    state_n_batch_dims = states.count_batch_dims_tensor(
        state_tensor, num_qubits
    )
    op_n_batch_dims = count_gate_batch_dims(operator)
    state_batch_dims = state_tensor.size()[:state_n_batch_dims]
    op_batch_dims = operator.size()[:op_n_batch_dims]

    # In the case of batched operators and a single state, we expand the state
    # into a repeated batch so that all operators act on the same state.
    if op_n_batch_dims > 0 and state_n_batch_dims == 0:
        state_tensor = state_tensor.expand(
            *op_batch_dims, *state_tensor.size()
        )
        state_n_batch_dims = op_n_batch_dims
        state_batch_dims = op_batch_dims

    state_tensor = states.subset_roll_to_back(state_tensor, state_n_batch_dims)
    operator = states.subset_roll_to_back(operator, op_n_batch_dims)

    def act(op, tensor):
        old_size = tensor.size()
        new_size = (
                (gate_dim,)
                + (num_qubits - gate_num_qubits) * (2,)
                + state_batch_dims
        )
        tensor_view = tensor.view(new_size)
        result = torch.einsum('ab..., b... -> a...', op, tensor_view)
        return result.view(old_size)

    result_batch_flipped = act(operator, state_tensor)
    return states.subset_roll_to_front(
        tensor=result_batch_flipped,
        subset_num_dims=state_n_batch_dims
    )


def apply_all_qubits(
        operator: torch.Tensor,
        state: torch.Tensor,
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
        operator: Tensor with size (*batch_dims, 2, 2) defining a 2 by 2 matrix
            which will act on every qubit.

        state: State in vector layout. This means that the state is a
            tensor with size (*batch_dims, 2^num_bits,).
    """
    if states.count_qubits_gate_matrix(operator) != 1:
        raise ValueError(
            f'Expected operator on 1 qubit, found a '
            f'{states.count_qubits_gate_matrix(operator)} qubit operator.'
        )

    num_qubits = states.count_qubits(state)
    state_tensor = states.to_tensor_layout(state)
    state_tensor = apply_all_qubits_tensor(
        operator, state_tensor, num_qubits=num_qubits
    )
    return states.to_vector_layout(state_tensor, num_qubits=num_qubits)


def apply_all_qubits_tensor(
        operator: torch.Tensor,
        state_tensor: torch.Tensor,
        num_qubits: int,
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
        operator: Tensor with size (*batch_dims, 2, 2) giving a 2 by 2 matrix.

        state_tensor: State in tensor layout. Size is
            (*batch_dims, 2, 2, ...,2, 2) where the number of 2's is equal to
            the number of qubits.

        num_qubits: The number of qubits for the quantum state.
    """
    state_tensor = act_first_qubits_tensor(
        operator=operator,
        state_tensor=state_tensor,
        num_qubits=num_qubits,
        gate_num_qubits=1
    )
    for i in range(1, num_qubits):
        state_tensor = swap_tensor(
            state_tensor, qubit_pair=(0, i), num_qubits=num_qubits
        )
        state_tensor = act_first_qubits_tensor(
            operator, state_tensor, num_qubits=num_qubits, gate_num_qubits=1
        )

    state_tensor = roll_qubits_tensor(
        state_tensor, num_qubits, num_steps=-1
    )

    return state_tensor


def apply_to_qubits(
        operators: Iterable[torch.Tensor],
        qubits: Iterable[int],
        state: torch.Tensor,
):
    """Apply single qubit gates to specified qubits of a state.

    This function applies an iterable of single-qubit operators to
    the qubits specified by an iterable of integers. Qubit indices
    must range from 0 to n-1 where n is the number of qubits.

    We collect all operators acting on a given qubit and multiply them prior
    to acting on the state; this "gate fusion" is often more efficient than
    applying gates to a state one-by-one.

    Args:
        operators: Iterable over single qubit operators. Size of each element
            should be (*batch_dims, 2, 2). All batch dims must be the same.

        qubits: Iterable over the qubit integers for each operator. `operators`
            and `qubits` must line up in the sense that zip(operators, qubits)
            produces matrix, qubit pairs appropriately.

        state: State in vector layout. The size should be (*batch_dims, 2^n). `
    """
    num_qubits = states.count_qubits(state)
    state_tensor = states.to_tensor_layout(state)
    state_tensor = apply_to_qubits_tensor(
        operators, qubits, state_tensor, num_qubits
    )
    return states.to_vector_layout(state_tensor, num_qubits=num_qubits)


def apply_to_qubits_tensor(
        operators: Iterable[torch.Tensor],
        qubits: Iterable[int],
        state_tensor: torch.Tensor,
        num_qubits: int,
):
    """Apply single qubit gates to specified qubits of state in tensor layout.

    This function collects all operators acting on a given qubit and multiplies
    them prior to acting on the state; this is often more efficient.

    Args:
        operators: Iterable over single qubit operators. Size of each element
            should be (*batch_dims, 2, 2). All batch dims must be the same.

        qubits: Iterable over the qubit integers for each operator. `operators`
            and `qubits` must line up in the sense that zip(operators, qubits)
            produces matrix, qubit pairs appropriately.

        state_tensor: State in tensor layout.

        num_qubits: The number of qubits for the state.    `
    """
    # First we fuse gates acting on the same qubit.
    qubits_to_fused_ops = fuse_single_qubit_operators(qubits, operators)

    # If qubit 0 is included, apply its gate first to avoid back and forth
    # permutation. Note that we pop the gate so qubit 0 will not be used again.
    try:
        gate = qubits_to_fused_ops.pop(0)
        state_tensor = act_first_qubits_tensor(
            gate,
            state_tensor,
            num_qubits,
            gate_num_qubits=1
        )
    except KeyError:
        pass

    perm = list(range(num_qubits))
    for q, gate in qubits_to_fused_ops.items():

        # Track the overall permutation as we swap qubits.
        # Very importantly, each q is encountered either once or never.
        perm[q], perm[0] = perm[0], q
        state_tensor = swap_tensor(state_tensor, (0, q), num_qubits)

        state_tensor = act_first_qubits_tensor(
            gate, state_tensor, num_qubits, gate_num_qubits=1
        )
    return permute_qubits_tensor(
        permutation=inverse_list_permutation(perm),
        state_tensor=state_tensor,
        num_qubits=num_qubits
    )


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
        state: torch.Tensor, num_steps=1
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
    num_qubits = states.count_qubits(state)

    state_tensor = states.to_tensor_layout(state)
    state_tensor = roll_qubits_tensor(state_tensor, num_qubits, num_steps)
    return states.to_vector_layout(state_tensor, num_qubits)


def roll_qubits_tensor(
        state_tensor: torch.Tensor,
        num_qubits: int,
        num_steps: int = 1
):
    """Perform a cyclic permutation of qubits for a state in tensor layout.

    If the initial tensor is psi, the output of this function
    is a new tensor psi_rolled which is related to psi by

    psi_rolled[a_0, a_1, ..., a_{n-1}]
        = psi[a_k, a_{k+1}, ..., a_{n-1}, a_0, ..., a_{k-1}].

    Note that this formula does not include batch dimensions which are
    allowed and arbitrary.

    Args:
        state_tensor: state in tensor layout to be permuted.

        num_qubits: Number of qubits for the state.

        num_steps: Number of indices to cycle.
    """
    num_batch_dims = states.count_batch_dims_tensor(
        state_tensor, num_qubits)
    num_steps = num_steps % num_qubits
    if num_steps == 0:
        return state_tensor

    identity = list(range(num_batch_dims, num_batch_dims + num_qubits))
    perm = identity[-num_steps:] + identity[:-num_steps]
    perm = list(range(num_batch_dims)) + perm
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
    return states.to_vector_layout(
        permuted_state_tensor, num_qubits=num_qubits
    )


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
        qubit_pairs: Tensor with size (2,) or (num_pairs, 2) giving a
            collection of control-target pairs.

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
            num_bits_memory_cutoff = 1000  # Effectively infinity.
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


def multi_controlled_z(
        qubits: Iterable[int],
        state_vector: torch.Tensor,
):
    """Apply a CC...CZ gate to the given qubits.

    Note that controlled-Z operations are independent of the control versus
    target. The parameter qubits includes all qubits that the gate acts on.
    """
    total_num_qubits = states.count_qubits(state_vector)
    qubits = list(qubits)
    qubit_set = set(qubits)

    inert_qubits = [
        q for q in range(total_num_qubits)
        if q not in qubit_set
    ]

    # Prepare diag(1, 1, ..., 1, -1) for the subsystem only.
    subsystem_factors = torch.ones(2 ** len(qubits))
    subsystem_factors[-1] = -1.
    factors = subsystem_factors.repeat(2 ** len(inert_qubits))

    # Put unused qubits on the left and C...CZ qubits on the right.
    perm = inert_qubits + qubits
    reverse_perm = [0] * total_num_qubits
    for q, i in enumerate(perm):
        reverse_perm[i] = q

    # Permute, apply the phase,
    state_vector = permute_qubits(perm, state_vector)
    state_vector = factors * state_vector
    return permute_qubits(reverse_perm, state_vector)


def multi_controlled_x(
        state_vector: torch.Tensor,
        controls: Iterable[int],
        target: int
):
    controls = list(controls)
    h = hadamard()
    state_vector = apply_operator(
        operator=h,
        qubits=[target],
        state=state_vector
    )
    state_vector = multi_controlled_z(
        qubits=controls + [target],
        state_vector=state_vector
    )
    return apply_operator(
        operator=h,
        qubits=[target],
        state=state_vector
    )
