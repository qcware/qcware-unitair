from .shapes import StateLayout, StateShapeError
from .shapes import count_qubits
from .shapes import count_qubits_tensor
from .shapes import count_qubits_gate_matrix
from .shapes import hilbert_space_dim
from .shapes import count_batch_dims_tensor
from .shapes import subset_roll_to_back, subset_roll_to_front
from .conversions import to_tensor_layout, to_vector_layout
from .innerprod import inner_product, norm_squared
from .innerprod import abs_squared
from .innerprod import diag_expectation_value
