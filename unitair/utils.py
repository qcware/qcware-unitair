import torch
from typing import List


def permutation_to_front(n: int, entries: List[int]):
    """Build permutation that maps given ordered entries to the front.

    This function creates a permutation on n elements that leaves each
    element alone except for those in `entries`. Those entries are moved
    to the front (i.e. the left) in the order of the list `entries`.

    Examples:
        >>> permutation_to_front(5, [2])
        [2, 0, 1, 3, 4]
        >>> permutation_to_front(5, [2, 1])
        [2, 1, 0, 3, 4]

    Permutations in PyTorch reminder:
        While permutations are naturally specified with dicts, PyTorch and
        NumPy use will things ike [3, 0, 1, 2] to indicate that object 3
        should be moved to slot 0, object 0 should be moved to slot 1, etc.
    """
    entries = entries.copy()
    arrangement = list(range(n))
    for q in entries:
        del arrangement[arrangement.index(q)]
    return entries + arrangement


def inverse_list_permutation(perm: List[int]):
    """Get the inverse permutation for a permutation formatted as a list.

    Examples:
        >>> inverse_list_permutation([1, 2, 3, 4, 0])
        [4, 0, 1, 2, 3]

    Permutations in PyTorch reminder:
        While permutations are naturally specified with dicts, PyTorch and
        NumPy use will things ike [3, 0, 1, 2] to indicate that object 3
        should be moved to slot 0, object 0 should be moved to slot 1, etc.
    """
    n = len(perm)
    inverse = [None for _ in range(n)]

    inverse_dict = {b: a for a, b in enumerate(perm)}
    for k, v in inverse_dict.items():
        inverse[k] = v

    return inverse


def int_to_bin(
        x: int,
        num_bits: int,
        str_output: bool = False,
        device: torch.device = torch.device("cpu")
):
    if x >= 2 ** num_bits:
        raise ValueError(f"Insufficient bits to store integer {x}.")
    if x < 0:
        raise ValueError(f"Expected a nonnegative integer, found {x}.")

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
