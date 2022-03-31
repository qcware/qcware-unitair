Quantum States
==============

.. attention::

    This section is under construction.


This section explains how quantum states are
represented in Unitair. To work with Unitair comfortably,
it's necessary to understand a subset of this section.
At minimum, a user should understand

#. `Basics <https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html>`_ of ``torch.Tensor`` objects
#. :ref:`Ordering<basis-order>` of computational basis elements
#. Batch dimensions
#. Real and imaginary parts of states

This section goes beyond these ideas, but

.. important::

    It cannot be emphasized enough that Unitair is not object-oriented
    in style and that there are no ``QuantumCircuit`` or ``QuantumState``
    classes. "Everything is a tensor" is a key principle, and the cost
    is having to get used to our conventions.

If you prefer to learn by playing around with functions,
we recommend looking at

* ``rand_state``
* ``unit_vector``
* ``unit_vector_from_bitstring``

all of which are in the ``unitair.initializations`` module.


.. _basis-order:

Computational Basis Order
-------------------------

The Hilbert space for :math:`n` qubits is :math:`{\bf C}^{2^n}`;
it has :math:`2^n` dimensions so there are :math:`2^n` basis
vectors. This is just the quantum analogue of the statement that
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

(We are not concerned here with writing states as column vectors
as this adds unnecessary confusion.) We note that not
all quantum computing references and software tools use
this basis order. There are advantages and disadvantages to both.

