from typing import Union


from . import states
from . import simulation

from .states.shapes import Field, StateLayout
from .initializations import unit_vector, uniform_superposition, rand_state

from .states import count_qubits, hilbert_space_dim
from .states import expectation_value
from .states import inner_product

from ._device_utils import cuda_info, _DefaultDevice
from torch import device

REAL_NUMBERS = states.shapes.Field.REAL
COMPLEX_NUMBERS = states.shapes.Field.COMPLEX
VECTOR_LAYOUT = states.shapes.StateLayout.VECTOR
TENSOR_LAYOUT = states.shapes.StateLayout.TENSOR

__DEFAULT_DEVICE__ = _DefaultDevice()
del _DefaultDevice


def get_default_device():
    """Get the current package-wide default PyTorch backend device.

    Devices can usually be selected manually in function calls,
    but when not specified, this function is used to fix the device.
    """
    return __DEFAULT_DEVICE__.device_selected


def set_default_device(identifier: Union[device, str]):
    """Set the package-wide default PyTorch backend device.

    Devices can usually be selected manually in function calls,
    but when not specified, the default device is used.

    To change the default device, this function can be called with
    an identifier as an argument. Valid identifiers are:
        'automatic': selects CUDA if CUDA is available, CPU otherwise.
        torch.device object
        string specification of a device (e.g. 'cuda:0', 'cpu', etc.)
    """
    if identifier in ('auto', 'automatic'):
        identifier = __DEFAULT_DEVICE__.standard_default_device()
    __DEFAULT_DEVICE__.set_default_device(identifier)


def reset_default_device():
    """Reset default device selection behavior: CUDA if available, else CPU."""
    set_default_device('automatic')


del device


