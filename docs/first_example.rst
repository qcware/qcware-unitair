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

    \ket{0} =& (1, 0)\\
    \ket{1} =& (0, 1)

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

.. code-block::
    :caption: Interactive Interpreter

    >>> new_state
    tensor([[1., 5.],
            [0., 1.]])

This is indeed the correct state :math:`\ket{0} + (5+i)\ket{1}`
expressed as a `Tensor` with the unitair convention of the
first dimension being for real and imaginary parts.



.. tip::

    To extract real and imaginary parts of a state, you
    can use ``unitair.states.real_imag``. You might think this
    is silly since, for example, ``new_state[1]`` is the imaginary part,
    but this function is more useful when dealing with
    batches of states and it improves readability.


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
is ``(batch_length, 2, hilbert_space_dimension)`` where
``hilbert_space_dimension`` is :math:`2^n` for :math:`n` qubits.
All Unitair functionality is built to understand that
states are formatted with this structure.

.. note::

    You may be thinking "Wouldn't the annoyance of knowing that
    states have to follow these specific size rules be avoided
    if Unitair just used a special ``QuantumState`` class instead
    of ``torch.Tensor``?" This is true, but there is enormous benefit
    to sticking with ``Tensor`` and the state conventions are easy
    to get used to. There is no need to convert back and
    forth between a ``Tensor`` and another class to use any PyTorch
    functionality, and Unitair has lots of internal validation to
    help avoid mistakes with state shapes.

For a spin-1/2 particle, the state :math:`\ket{0}` indicates that
spin "points" in the :math:`+z` direction. (If you are not comfortable
with this language, just think of :math:`\ket{0}` as an arrow pointing
vertically up.)
Applying a magnetic field
in the :math:`+x` direction will cause the spin to rotate its orientation
about the :math:`x` axis, resulting in a spin that points somewhere in
the :math:`y-z` plane.


