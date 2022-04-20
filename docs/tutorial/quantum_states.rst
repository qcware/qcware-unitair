**************
Quantum States
**************

.. attention::

    This section is under construction.


This section explains how quantum states are
represented in Unitair. To work with Unitair comfortably,
it's necessary to understand a subset of this section.
At minimum, a user should understand

#. `Basics <https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html>`_ of ``torch.Tensor`` objects
#. :ref:`Ordering of computational basis elements<basis-order>`
#. :ref:`Batch dimensions<batch-dims>`


.. important::

    It cannot be emphasized enough that Unitair is not particularly
    object-oriented and that there are no ``QuantumCircuit`` or
    ``QuantumState`` classes. "Everything is a tensor" is a key principle, and
    a downside to this principle having to get used to our conventions.

If you prefer to learn by playing around with functions,
we recommend looking at

* ``rand_state``
* ``unit_vector``
* ``unit_vector_from_bitstring``

all of which are in the ``unitair.initializations`` module.


.. _basis-order:

Computational Basis Order
=========================

The Hilbert space for :math:`n` qubits is :math:`{\bf C}^{2^n}`;
it has :math:`2^n` dimensions so there are :math:`2^n` basis
vectors. This is a quantum analogue of the statement that
:math:`n` bits have :math:`2^n` possible values.

For two qubits, the "computational basis vectors"
are :math:`\ket{00}, \ket{01}, \ket{10}, \ket{11}`.
When it comes to bases, order matters, and Unitair
adopts the convention that basis elements go
in the order of standard binary counting. For example:

.. math::

    \ket{00} &= \left(1,0,0,0\right)\\
    \ket{01} &= \left(0,1,0,0\right)\\
    \ket{10} &= \left(0,0,1,0\right)\\
    \ket{11} &= \left(0,0,0,1\right)\\

(We are not concerned here with writing states as column vectors.)
Not all quantum computing references and software tools use
this basis order.

Now consider an arbitrary pure quantum state :math:`\psi` for :math:`n`
qubits. We can write :math:`\psi` as a unique linear combination of
computational basis vectors with complex coefficients:

.. math::

    \psi = \sum_{x \in \{0, 1\}^n} \alpha_x \ket{x}

where each :math:`\alpha_x` is complex and somewhat confusingly,
the index :math:`x` runs over all binary strings

.. math::

    (0, 0, \ldots, & 0, 0, 0)\\
    (0, 0, \ldots, & 0, 0, 1)\\
    (0, 0, \ldots, & 0, 1, 0)\\
    (0, 0, \ldots, & 0, 1, 1)\\
    \vdots \\
    (1, 1, \ldots, & 1, 1, 1).

By keeping with this ordering, we can represent :math:`\psi` with a
vector as in

.. math::
    :label: state-linear-comb

    \psi \leftrightarrow
    \left(\alpha_0, \alpha_1, \alpha_2, \ldots, \alpha_{2^n - 1} \right)

In this equation, we are using, e.g., :math:`\alpha_3` instead of writing
:math:`\alpha_{(0,0,\ldots,0,1,1)}` to keep notation readable. (This
is a small abuse of notation.)


Examples
^^^^^^^^


Example 1
"""""""""

The quantum state :math:`\ket{00}` should be
represented in Unitair as

.. code-block:: python

    tensor([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

Example 2
"""""""""

The entangled state

.. math::

    \frac{1}{\sqrt{2}} \left( \ket{01} - \ket{10}\right)

corresponds to

.. code-block::

    tensor([ 0.+0.j,  0.7071+0.j, -0.7071+0.j,  0.+0.j])

Example 3
"""""""""

The state

.. math::

    e^{i}\ket{110}

corresponds to

.. code-block:: python

    tensor([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
            0.+0.j, 0.+0.j, 0.5403+0.8415j, 0.+0.j,])

.. tip::

    You may find ``unit_vector_from_bitstring`` convenient
    when experimenting with states. For example

    .. code-block:: python

        >>> unit_vector_from_bitstring('01')
        tensor([0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j])

    We only recommend using this function for experimentation
    as it isn't as fast as ``unit_vector``.

.. _batch-dims:

Batch Dimensions
================

States in Unitair allow for arbitrary batch dimensions.
Batched states can have size

.. math::

    (B, 2^n)

where :math:`B` is some positive integer. More generally,
there can be an arbitrary number of batch dimensions so that
states have size


.. math::

    (B_1, B_2, \ldots, B_k, 2^n)


This is the most general form of the size of a state. A PyTorch ``Tensor``
object with this size is referred to, in the Unitair context, as
a *state in vector layout*.

.. note::

    The concept of "vector layout" refers to the :math:`2^n`
    at the end of the size, which casts
    individual quantum states as a vector with one index (ignoring batches).
    Unitair also uses, especially internally,
    a *tensor layout* where the dimension with length :math:`2^n`  is
    reshaped to have :math:`n` indices, each of which runs over two values.


Examples
^^^^^^^^


Example 1
"""""""""
The two Bell pairs


.. math::

    \frac{1}{\sqrt{2}} \left( \ket{00} + \ket{11}\right) \\
    \frac{1}{\sqrt{2}} \left( \ket{00} - \ket{11}\right) \\

can be written constructed as a batch:

.. code-block:: python

    tensor([[ 0.7071+0.j,  0.+0.j,  0.+0.j,  0.7071+0.j],
            [ 0.7071+0.j,  0.+0.j,  0.+0.j, -0.7071+0.j]])


Example 2
"""""""""

When dealing with lots of batch entries and qubits, tensors
can quickly get very large.

.. code-block:: python

    from unitair.initializations import rand_state

    state_batch = rand_state(
        num_qubits=10,
        batch_dims=(30, 5),
    )

    print(state_batch.size())

.. code-block:: none
    :caption: Output

    torch.Size([30, 5, 1024])










