.. Unitair documentation master file, created by
   sphinx-quickstart on Fri Mar 25 17:13:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Unitair Documentation
=====================
Unitair is a lightweight Python package that
brings quantum computing to PyTorch.

Unitair differs from other quantum computing software packages
in important ways. Quantum states are PyTorch tensors (the PyTorch
version of the NumPy ``ndarray``). There is no special class for quantum states
class nor is there an abstract ``QuantumCircuit`` class.
Unitair doesn't directly rely on any circuit model although it
supports circuit-model computation.

Manipulations of quantum states naturally take advantage of PyTorch's
strengths. You can

* Apply gates and other operations to a batch of states
* Use gradients to track gate parameters or parameters used to build an initial state
* Set ``device='cuda'`` to get GPU-acceleration
* Mix ``unitair`` functions with ``torch.nn`` networks or any other PyTorch functionality

The design of Unitair encourages users to do things that, really, should never
be done to quantum states. Their components can be directly read and
manipulated in physically unrealistic ways. This freedom allows for
great efficiency when emulating quantum state evolution. If you know
that the effect of a series of operations is to manipulate a state
in a certain way, why not just change the components by hand rather than
applying gates in a computationally complicated way?

Intended users
~~~~~~~~~~~~~~
Unitair was designed with the goal of helping to bridge the fields
of quantum computing and machine learning. Anyone with experience in
PyTorch (or another machine learning library like TensorFlow)
and basic knowledge of quantum computing should find
Unitair to be very simple. Users that are experts in
machine learning or quantum computing but not both
should find Unitair helpful to start making a connection
with the other discipline.

What Unitair Is and Is Not
~~~~~~~~~~~~~~~~~~~~~~~~~~

Unitair...

* is a library of PyTorch linear algebra functions for quantum computing
* is a quantum computing emulation tool
* can substantially reduce circuit simulation time (when used well)
* is lightweight and easy to use if you have PyTorch experience
* batches quantum states and operations for performance on a CPU or GPU
* can simulate any quantum circuit (within computational constraints)
* tracks gradients (everything PyTorch under the hood)

but Unitair is **not**...

* a replacement for Circ, Qiskit, Quasar, Q#, etc.
* designed to simulate noise realistically
* a quantum circuit editor
* helpful for deploying quantum circuits on quantum hardware
* a quantum algorithm library

.. toctree::
   :maxdepth: 2
   :caption: Tutorial:

   tutorial/install
   tutorial/first_example
   tutorial/quantum_states

.. toctree::
   :maxdepth: 2
   :caption: API

   api/states
   api/initializations
   api/apply_operators
   api/gates



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
