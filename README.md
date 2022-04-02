#Unitair: PyTorch-based quantum circuits

[![Documentation Status](https://readthedocs.org/projects/unitair/badge/?version=latest)](https://unitair.readthedocs.io/en/latest/?badge=latest)
      

Unitair is a lightweight Python package that 
brings quantum computing to PyTorch.

Unitair differs from other quantum computing software packages
in important ways. Quantum states are PyTorch tensors (the PyTorch
version of the NumPy `ndarray`). There is no special class for quantum states
class nor is there an abstract `QuantumCircuit` class. 
Unitair doesn't directly rely on any circuit model although it 
supports circuit-model computation.

Manipulations of quantum states naturally take advantage of PyTorch's strengths. 
You can
- Apply gates and other operations to a batch of states
- Use gradients to track gate parameters or parameters used to build an initial state
- Set `device='cuda'` to get GPU-acceleration
- Mix `unitair` functions with `torch.nn` networks or any other PyTorch functionality

[Documentation for Unitair is now available.](https://unitair.readthedocs.io/)


### Cheat whenever possible
Unlike standard quantum circuit simulation, `unitair` 
is designed to encourage users to 
"just get the answer" and does not aim
to perform operations in a way that models real
quantum circuits or hardware. In this sense, `unitair` should
be regarded as an *emulator* rather than a simulator.

Functions aim to take shortcuts whenever possible. 
Rather than applying a Hadamard gate to every
qubit starting with |0...0>, 
the best practice is to use `unitair.uniform_superposition`
which reads off the state directly.

This approach has three notable downsides:
1. `unitair` does not aim to simulate noise realistically.
1. Users should be aware that manipulations that look
   simple with `unitair` may be very complex when constructed with 
   realistic gates.
1. Deployment to hardware or translation to other quantum computing
   packages is not an intended usage of `unitair`.
   
On the other hand, emulation has massive benefits: researchers can
test or develop quantum algorithms with lower runtimes than
is possible with standard simulation, and states can
be manipulated in arbitrary ways, whether physically sensible or not.

### Intended users
Unitair was designed with the goal of helping to bridge the fields
of quantum computing and machine learning. Anyone with experience in 
PyTorch (or another machine learning library like TensorFlow) 
and basic knowledge of quantum computing should find
`unitair` to be very simple. Users that are experts in 
machine learning or quantum computing but not both
should find `unitair` helpful to start making a connection 
with the other discipline.

### States are tensors
Unitair avoids unnecessary complexity by just using `torch.Tensor` as the
class for quantum states. For example, the state $|00>$ is
```python
import torch  # no need to import unitair yet!

# Intended state: |00>
state = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 0., 0.]])
```
This `Tensor` has size `(2, 4)`. 
The four columns of this matrix correspond to the computational basis elements
$|00>, |01>, |10>,$ and $|11>$ (in that order). Since the quantum state can be
written as 

1 |00> + 0 |01> +  0 |10> + 0 |11>, 

the vector `[1., 0., 0., 0.]`
is sensible. The reason that there are two rows is that coefficients of quantum
states are complex numbers. The first and second rows are for real an imaginary 
parts respectively.

```python
# Intended state: -i |01>
state = torch.tensor([[0., 0., 0., 0.],
                      [0., -1., 0., 0.]])
```

### Batches of states
To exploit the strengths of PyTorch, manipulations should be batched.
This means that rather than constructing a tensor with size `(2, 4)`, we
might instead construct a tensor with size `(3, 2, 4)`. The `unitair` interpretation
of such a `Tensor` is that we have 3 quantum states for two qubits. For example:

```python
>>> import unitair
>>> unitair.rand_state(num_qubits=2, batch_dims=(3,))
tensor([[[-0.0222,  0.1360, -0.2531, -0.6884],
         [-0.1672, -0.5563,  0.2098, -0.2482]],
         
        [[ 0.5456, -0.2316,  0.0870,  0.0334],
         [ 0.2550, -0.0349, -0.7393,  0.1649]],
         
        [[ 0.5787,  0.2652,  0.1518, -0.2148],
         [-0.5141, -0.3789,  0.3423, -0.0215]]])
```

In fact, batch dimensions can be more general thant that:
```python
state_batch = unitair.rand_state(num_qubits=2, batch_dims=(10, 3,))
```

```python
>>> state_batch.size()
torch.Size([10, 3, 2, 4])
```

In this case, `state_batch[5, 1]`, is a quantum state for two qubits, as is any other
selection of the first two indices of `state_batch`. This is a batch of 30 states for two qubits
organized into the (10, 3) shape.


### Manipulating quantum states
Since states are tensors, you are free to do anything to a state that
you might do to a `torch.Tensor`. In particular, you are free to do manipulations
that are not unitary or even manipulations that don't correspond to anything
physically possible. On the other hand, the `unitair` package has a number of functions that
will let you do operations that are natural in quantum mechanics.

#### Applying Hadamard gates
We first apply a Hadamard gate to the initial state $|0>$:

```python
from unitair import simulation
from unitair import gates

# Initial state: |0>
state = unitair.unit_vector(index=0, num_qubits=1)
h = gates.hadamard()

state = simulation.apply_operator(
    operator=h,
    state=state,
    qubits=(0,)
)
```

```python
>>> state
tensor([[0.7071, 0.7071],
        [0.0000, 0.0000]])
```

Unitair can apply gates to batches of quantum states as well. For example,
we can construct a batch consisting of 5 state for one qubit and
then apply a Hadamard gate to each of those states in a single call:
```python
state_batch = unitair.rand_state(num_qubits=1, batch_dims=(5,))
h = gates.hadamard()

state_batch = simulation.apply_operator(
    operator=h,
    state=state_batch,
    qubits=(0,)
)
```

The resulting `state_batch` has size `(5, 2, 2)` In fact, `state_batch[3]` is the same as
if we had applied a Hadamard gate directly to the index 3 element of the original `state_batch`.


#### Making a Bell state
The Bell state $(|00> + |11>) / \sqrt{2}$ is typically constructed by
starting with the state |00>, applying a Hadamard gate to the first
qubit, and then applying a CNOT gate from the first to the second
qubit. This construction can be done with `unitair`.
```python
from unitair import simulation
from unitair import gates

# Initial state: |00>
state = unitair.unit_vector(index=0, num_qubits=2)
h = gates.hadamard()
cnot = gates.cnot()

state = simulation.apply_operator(
    operator=h,
    state=state,
    qubits=(0,)
)

state = simulation.apply_operator(
    operator=cnot,
    state=state,
    qubits=(0, 1)
)
```

```python
>>> state
tensor([[0.7071, 0.0000, 0.0000, 0.7071],
        [0.0000, 0.0000, 0.0000, 0.0000]])
```

#### About Unitair
Unitair was written at QC Ware Corp. by Sean Weinberg.
Fabio Sanches envisioned and suggested the project in 2020.

If you have questions or feedback, or if you would like to contribute to Unitair,
please email sean.weinberg@qcware.com.
