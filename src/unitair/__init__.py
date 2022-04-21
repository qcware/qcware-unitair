from . import states
from . import simulation

from .states.shapes import StateLayout
from .initializations import unit_vector, uniform_superposition, rand_state

from .states import count_qubits, hilbert_space_dim
from .states import diag_expectation_value
from .states import inner_product, norm_squared, abs_squared

VECTOR_LAYOUT = states.shapes.StateLayout.VECTOR
TENSOR_LAYOUT = states.shapes.StateLayout.TENSOR
