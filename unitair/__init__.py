from . import states
from . import simulation

from .states.shapes import Field, StateLayout
from .initializations import unit_vector, uniform_superposition, rand_state

from .states import count_qubits, hilbert_space_dim
from .states import expectation_value
from .states import inner_product, norm_squared, abs_squared
from ._device_utils import cuda_info

REAL_NUMBERS = states.shapes.Field.REAL
COMPLEX_NUMBERS = states.shapes.Field.COMPLEX
VECTOR_LAYOUT = states.shapes.StateLayout.VECTOR
TENSOR_LAYOUT = states.shapes.StateLayout.TENSOR
