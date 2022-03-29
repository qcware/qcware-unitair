Introductory Examples
=====================

We can introduce several core ideas of
``unitair`` with simple physical examples involving one qubit.
We use terms like "spin 1/2"
and "magnetic moment" in this section, but there is absolutely
no need to know what these mean to understand our example or
``unitair`` in general.


Initializing a State
~~~~~~~~~~~~~~~~~~~~

We start by creating an initial state for only one qubit. One
qubit has a two-dimensional complex Hilbert space of states,
and we can write its basis vectors as

.. math::

    \ket{0} =& \left(\begin{array}{c} 1\\ 0 \end{array}\right)\\
    \ket{1} =& \left(\begin{array}{c} 0\\ 1 \end{array}\right)

With PyTorch, we encode these states with tensors
of size (2, 2):

.. code-block:: python

    import torch

    ket_0 = torch.tensor([[1., 0.],
                          [0., 0.]])

    ket_1 = torch.tensor([[0., 1.],
                          [0., 0.]])

The role of the second dimension is obvious. The
first dimension is to allow for real and imaginary
parts. For example, the state

.. math::

    \psi = \frac{1}{\sqrt{2}} \left( -i\ket{0} + \ket{1} \right)

corresponds to the ``torch.Tensor``

.. code-block:: python

    torch.tensor([[ 0.0000,  0.7071],
                  [-0.7071,  0.0000]])

.. tip::

    Computational basis vectors like :math:`\ket{0}`
    can be obtained easily with ``unitair.initializations.unit_vector``.
    For example, ``unit_vector(0, num_qubits=1)`` produces :math:`\ket{0}`.


Operating on a State
~~~~~~~~~~~~~~~~~~~~

Consider an operator on two qubits like

.. math::
    Q = \left(\begin{array}{cc}
    1 & 5-i\\
    5+i & -1
    \end{array}\right)

Unitair has a convention for such a matrix which has the same
idea as in the case of states. Namely:

.. code-block:: python
    :name: q-matrix

    q = torch.tensor([[[ 1.,  5.],
                       [ 5.,  -1.]],

                      [[ 0., -1.],
                       [ 1.,  0.]]])

The ``Tensor`` ``q`` has the size (2, 2, 2). It's better to think of
this size as :math:`(2, 2^1, 2^1)` where the 1's are due to
the operator acting on one qubit. A two-qubit operator would have
size :math:`(2, 2^2, 2^2) = (2, 4, 4)`. As with states,
the first dimension is for real and imaginary parts.

We can apply the operator :math:`Q` to the state :math:`\ket{0}`:

.. math::

    Q\ket{0}
    &=\left(\begin{array}{cc}
        1 & 5-i\\
        5+i & -1
    \end{array}\right)\left(\begin{array}{c}
        1\\
        0
    \end{array}\right)\\
    &=\left(\begin{array}{c}
        1\\
        5+i
    \end{array}\right)

To perform this operation with Unitair, we use the function
``apply_operator``:

.. code-block:: python

    from unitair.simulation import apply_operator

    # q and ket_0 already defined as above
    new_state = apply_operator(
        operator=q,
        qubits=(0,),
        state=ket_0
    )

.. code-block:: python
    :caption: Interactive Interpreter

    >>> new_state
    tensor([[1., 5.],
            [0., 1.]])

This is indeed the correct state :math:`\ket{0} + (5+i)\ket{1}`
expressed as a `Tensor` with the unitair convention of the
first dimension being for real and imaginary parts.



.. tip::

    To extract real and imaginary parts of a state, you
    can use ``unitair.states.real_imag``. This function
    is especially useful when dealing with
    batches of states (discussed shortly).


Operating on Batches of States
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What if we wanted to compute the action of :math:`Q` on
both :math:`\ket{0}` and :math:`\ket{1}`? We could
use ``apply_operator`` twice, but fails to take
advantage of vectorization, the C backend of PyTorch
and, if available, CUDA.

What we want is to operate on a *batch* of two states:
``ket_0`` and ``ket_1``. This is done by creating
the tensor ``torch.stack([ket_0, ket_1])`` which is the same as

.. code-block:: python

    state_batch = torch.tensor([[[1., 0.],
                                 [0., 0.]],

                                [[0., 1.],
                                 [0., 0.]]])

Which has size (2, 2, 2). The repeated twos are
just an unfortunate coincidence, and the more general form
is

.. code-block:: python

    (batch_length, 2, hilbert_space_dimension)

where ``hilbert_space_dimension`` is :math:`2^n` for :math:`n` qubits.
All Unitair functionality is built to understand that
states are formatted with this structure, and deviating from it
is more likely to raise errors than to give incorrect results.

.. note::

    Having to remember the conventions for shapes of states in Unitair
    may seem frustrating. A ``QuantumState`` class would
    eliminate this issue, but it would come with other costs.
    Sticking with a plain ``Tensor`` means that PyTorch functionality
    can be used without the burden of converting between types and
    it makes Unitair easier to learn for PyTorch users.

Now let's apply :math:`Q` to both :math:`\ket{0}` and :math:`\ket{1}`:

.. code-block:: python

    from unitair.simulation import apply_operator

    # q and state_batch already defined as above
    new_state = apply_operator(
        operator=q,
        qubits=(0,),
        state=state_batch
    )

.. code-block:: python
    :caption: Interactive Interpreter

    >>> new_state_batch
        tensor([[[ 1.,  5.],
                 [ 0.,  1.]],

                [[ 5., -1.],
                 [-1.,  0.]]])

The result is a new batch of states with the expected structure. The first
batch entry is :math:`Q \ket{0}` and the second is :math:`Q \ket{1}`.
Although this example is trivial, it's important to not underestimate
the benefits of batching. Running ``apply_operator`` with a batch
of states can be thousands of times faster than running it repeatedly
in a loop, even on a CPU.


Batched Operations on a State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batching is a fundamental concept for NumPy and PyTorch and indeed
it is central to Unitair. Not only can one operator act on many states,
but we can have many operators act on one state. (And in fact, we can
also have a collection of operators act on a collection of states in
one-to-one correspondence.)

.. note::

    When we talk about a batch of operators acting on a state,
    we mean obtaining the results of operating
    with each individual operator on the *same* initial state
    in "parallel", not in "sequence". We are not constructing
    a circuit by composing operators.

When we
:ref:`constructed the matrix<q-matrix>` :math:`Q` as
a ``Tensor``, it had size ``(2, 2, 2)`` which has the form

.. code-block:: none
    :caption: Operator size (no batch)

    (
        2,   (Real and imaginary parts)
        2^k, (Matrix rows, k = number of qubits the matrix acts on)
        2^k, (Matrix columns)
    )

Thus, we get :math:`(2, 2, 2)` when :math:`k=1`.

To create a batch of operators, we just add an additional dimension
on the left:

.. code-block:: none
    :caption: Operator size (one batch dimension)
    :name: op-size-one-batch-dim

    (
        batch_length,
        2,   (Real and imaginary parts)
        2^k, (Matrix rows)
        2^k, (Matrix columns)
    )

Now let's create a batch of operators. Given a real number :math:`t`,
consider the operator

.. math::
    U(t) = \left(\begin{array}{cc}
    \cos t & -i \sin t \\
    -i \sin t & \cos t
    \end{array}\right)

If you have a background in quantum mechanics, you may recognize
this operator as a spin 1/2 rotation about
the :math:`x`-axis by angle :math:`2t`. Regardless, note that :math:`U(t)`
can be written as


.. math::

    U(t) &= \cos (t) - i \sin (t) \, X \\
        &= e^{-i t X}

where :math:`X` is the Pauli operator

.. math::
    X = \left(\begin{array}{cc}
    0 & 1 \\
    1  & 0
    \end{array}\right)

and we use the matrix exponential function.

Unitair allows
us to construct :math:`e^{-i t X}` very easily:

.. code-block:: python
    :caption: Interactive Interpreter

    >>> from unitair.gates import exp_x
    >>> exp_x(.5)
    tensor([[[ 0.8776,  0.0000],
             [ 0.0000,  0.8776]],

            [[ 0.0000, -0.4794],
             [-0.4794,  0.0000]]])

You can confirm that this operation is as expected.

Now what if we want to consider a batch of different parameters :math:`t`?

.. code-block::

    import torch
    from math import pi
    from unitair.gates import exp_x

    # Create t = torch.tensor([0, .01, .02, ..., approximately pi])
    t = torch.arange(0, pi, .01)
    gate_batch = exp_x(t)

.. code-block:: python
    :caption: Interactive Interpreter

    >>> gate_batch.size()
    torch.Size([315, 2, 2, 2])

    >>> gate_batch[0]
    tensor([[[1., 0.],
             [0., 1.]],

            [[0., -0.],
             [-0., 0.]]])

    >>> gate_batch[1]
    tensor([[[ 0.9999,  0.0000],
             [ 0.0000,  0.9999]],

            [[ 0.0000, -0.0100],
             [-0.0100,  0.0000]]])

This is all consistent with
the :ref:`expected batched operator size<op-size-one-batch-dim>`.

Let's now apply *all* of these operators to :math:`\ket{0}`:

.. code-block:: python

    from unitair.simulation import apply_operator

    # gate_batch and ket_0 already defined as above
    states = apply_operator(
        operator=gate_batch,
        qubits=(0,),
        state=ket_0
    )

.. code-block:: python
    :caption: Interactive Interpreter

    >>> states.size()
    torch.Size([315, 2, 2])

    # The first 3 states rotated away from |0>
    >>> states[:3]
    tensor([[[ 1.0000,  0.0000],
             [ 0.0000,  0.0000]],

            [[ 0.9999,  0.0000],
             [ 0.0000, -0.0100]],

            [[ 0.9998,  0.0000],
             [ 0.0000, -0.0200]]])

    # The last state was almost rotated by 360 degrees and returns to
    # approximately -|0> rather than |0>, a famous property of half-integer
    # spin particles. Note that the approximate result is because the last
    # parameter was pi - .01 instead of pi.
    >>> states[-1]
    tensor([[-1.0000,  0.0000],
            [ 0.0000, -0.0016]])

For convenience, we can ask Unitair about the probabilities of
measuring 0 and 1:

.. code-block:: python
    :caption: Interactive Interpreter

    >>> from unitair.states import abs_squared
    >>> probabilities = abs_squared(state)
    >>> probabilities[:5]
    tensor([[1.0000e+00, 0.0000e+00],
            [9.9990e-01, 9.9997e-05],
            [9.9960e-01, 3.9995e-04],
            [9.9910e-01, 8.9973e-04],
            [9.9840e-01, 1.5991e-03]])






