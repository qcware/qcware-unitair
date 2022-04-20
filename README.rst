=======================================
Unitair: PyTorch-based quantum circuits
=======================================
.. image:: https://readthedocs.org/projects/unitair/badge/?version=latest
    :target: https://unitair.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Unitair is a lightweight Python package that 
brings quantum computing to PyTorch.

Unitair differs from other quantum computing software packages
in important ways. Quantum states are PyTorch tensors (the PyTorch
version of the NumPy ``ndarray``). There is no special class for quantum states
class nor is there an abstract ``QuantumCircuit`` class.
Unitair doesn't directly rely on a circuit model although it
fully supports circuit-model computation.

Manipulations of quantum states naturally take advantage of PyTorch's strengths. 
You can

- Apply gates and other operations to a batch of states
- Use gradients to track gate parameters or parameters used to build an initial state
- Set ``device='cuda'`` to get GPU-acceleration
- Mix ``unitair`` functions with ``torch.nn`` networks or any other PyTorch functionality

`Documentation for Unitair is now available.
<https://unitair.readthedocs.io/>`_


Rule-Breaking is Encouraged
===========================
Unitair does not hide state vectors or simulation
details within carefully crafted tools but instead exposes
states and operations on states as simple manipulations on PyTorch
``Tensor`` objects. As a result, there are few "seatbelts" preventing sser
from manipulating states in unphysical or unrealistic
ways like cloning, state-dependent time evolution, or cheating to
get a result that we somehow know has to be the case.

Because Unitair encourages users to "just get the answer", Unitair should
be regarded as an *emulator* rather than a simulator.
As a simple example, rather than applying a Hadamard gate to every
qubit starting with |0...0>, the best practice is to use
``unitair.uniform_superposition`` which reads off the state directly.

This approach has three notable downsides:

#. Unitair does not aim to simulate noise realistically.
#. Users should be aware that manipulations that look simple with Unitair
   and PyTorch may be very complex when constructed with realistic gates.
#. Deployment to hardware or translation to other quantum computing packages
   is not an intended usage of Unitair.
   
On the other hand, emulation has benefits: researchers can
test or develop quantum algorithms with lower runtimes than
is possible with full simulation, and states can
be manipulated in arbitrary ways, whether physically sensible or not.

Intended users
==============
Unitair was designed with the goal of helping to bridge the fields
of quantum computing and machine learning. Anyone with experience in 
PyTorch (or another machine learning library like TensorFlow) 
and basic knowledge of quantum computing should find
``unitair`` to be very simple. Users that are experts in
machine learning or quantum computing but not both
should find ``unitair`` helpful to start making a connection
with the other discipline.

States are Tensors
==================
The class for quantum states is `torch.Tensor` rather than a new
quantum state class. For example, the state :math:`|00>` is

.. code-block:: python

    import torch  # no need to import unitair yet!

    # Intended state: |00>
    state = torch.tensor([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

The four elements of this vector correspond to the computational basis elements
:math:`|00>, |01>, |10>,` and :math:`|11>` (in that order). Since the
quantum state can be
written as

.. math::

    1 |00> + 0 |01> +  0 |10> + 0 |11>,

the vector ``[1., 0., 0., 0.]``
is sensible.

As another example, :math:`-i |01>` is


.. code-block:: python

    import torch  # no need to import unitair yet!

    # Intended state: |00>
    state = torch.tensor([0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j])

Batches of states
=================
To exploit the strengths of PyTorch, manipulations should be batched.
Rather than constructing a tensor with size `(4,)`, we
might instead construct a ``Tensor`` with size `(3, 4)`. The `unitair` interpretation
of such a `Tensor` is that we have three quantum states for two qubits.
For example:

.. code-block:: python

    >>> import unitair
    >>> unitair.rand_state(num_qubits=2, batch_dims=(3,))
    tensor([[ 0.1958-0.3280j,  0.3178+0.4487j, -0.4322-0.0840j, -0.5906+0.0957j],
            [ 0.1541+0.4326j,  0.6663-0.0448j, -0.3485-0.0493j, -0.1967-0.4249j],
            [ 0.2089+0.4304j,  0.3997+0.6920j, -0.1714-0.2164j,  0.1803+0.1543j]])

In fact, batch dimensions can be more general thant that:

.. code-block:: python

    state_batch = unitair.rand_state(num_qubits=5, batch_dims=(10, 3,))

    >>> state_batch.size()
    torch.Size([10, 3, 32])
```

In this case, `state_batch[5, 1]`, is a quantum state for five qubits, as is any other
selection of the first two indices of ``state_batch``. This is a batch of 30
states for five qubits organized into the (10, 3) shape.


Manipulating quantum states
===========================
Because states are ``torch.Tensor`` objects, you are free to do anything to a
state that you might do to a ``torch.Tensor``.  Manipulations need not
have anything to do with quantum mechanics. On the other hand, the ``unitair``
package includes functions to perform operations that are natrual
in quantum computing.

Applying Hadamard gates
^^^^^^^^^^^^^^^^^^^^^^^

We first apply a Hadamard gate to the initial state :math:`|0>`:

.. code-block:: python

    from unitair import simulation, gates

    # Initial state: |0>
    state = unitair.unit_vector(index=0, num_qubits=1)
    h = gates.hadamard()

    state = simulation.apply_operator(
        operator=h,
        qubits=(0,)
        state=state,
    )


.. code-block:: python

    >>> state
    tensor([0.7071+0.j, 0.7071+0.j])


Unitair can apply gates to batches of quantum states, batches of gates
to a single state, and batches of gates to batches of states. For example,
we can construct a batch consisting of 5 states for one qubit and
then apply a Hadamard gate to each of those states in a single call:

.. code-block:: python

    state_batch = unitair.rand_state(num_qubits=1, batch_dims=(5,))
    h = gates.hadamard()

    state_batch = simulation.apply_operator(
        operator=h,
        qubits=(0,)
        state=state_batch,
    )


The resulting ``state_batch`` has size `(5, 2)` and, e.g.,
`state_batch[3]` is the same as if we had applied a Hadamard gate
directly to the index 3 element of the original `state_batch`.


Making a Bell state
^^^^^^^^^^^^^^^^^^^
The Bell state :math:`(|00> + |11>)/\sqrt{2}` is typically constructed by
starting with the state :math:`|00>`, applying a Hadamard gate to the first
qubit, and then applying a CNOT gate from the first to the second
qubit. We recommend just writing down this state by hand, but
the circuit construction can be done with Unitair as an example:


.. code-block:: python

    from unitair import simulation, gates

    # Initial state: |00>
    state = unitair.unit_vector(index=0, num_qubits=2)
    h = gates.hadamard()
    cnot = gates.cnot()

    state = simulation.apply_operator(
        operator=h,
        qubits=(0,),
        state=state,
    )

    state = simulation.apply_operator(
        operator=cnot,
        qubits=(0, 1),
        state=state,
    )

.. code-block:: python

    >>> state
    tensor([0.7071+0.j, 0.0000+0.j, 0.0000+0.j, 0.7071+0.j])


About Unitair
=============
Unitair was written at QC Ware Corp. by Sean Weinberg.
Fabio Sanches envisioned and suggested the project in 2020.

If you have questions or feedback, or if you would like to contribute to Unitair,
please email sean.weinberg@qcware.com.
