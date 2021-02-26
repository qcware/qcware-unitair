import torch
from typing import Optional
import unitair


def int_to_bin(x: int, num_bits: int, str_output: bool = False, device: Optional[torch.device] = None):
    if x >= 2 ** num_bits:
        raise ValueError(f"Insufficient bits to store integer {x}.")
    if device is None:
        device = unitair.get_default_device()

    bin_format = [int(d) for d in str(bin(x))[2:]]
    padding_size = num_bits - len(bin_format)
    bin_format = padding_size * [0] + bin_format

    output = torch.tensor(bin_format, dtype=torch.bool, device=device)
    if not str_output:
        return output
    else:
        return bool_to_strings(output)


def bool_to_strings(bool_data):
    """Convert a vector of [True, False, False, True, ...] to '1001...' ."""
    def conversion(entry):
        if isinstance(bool_data, torch.Tensor):
            entry = entry.item()
        return str(int(entry))
    mapping = map(conversion, bool_data)

    return ''.join(mapping)
