from typing import Union

import torch

from unitair import Field


def matmul(
        op_1: torch.Tensor,
        op_2: torch.Tensor,
        field: Union[Field, str] = Field.COMPLEX
):
    """Perform matrix multiplication on a pair of operators.

    Operators that act on k qubits have size (*batch, 2, k, k) in the
    complex case and (*batch, k, k) in the real case. Batching cases
    are inherited from torch.matmul.
    """
    field = Field.from_case_insensitive(field)
    if field is Field.REAL:
        return torch.matmul(op_1, op_2)
    real_1 = op_1.select(dim=-3, index=0)
    imag_1 = op_1.select(dim=-3, index=1)
    real_2 = op_2.select(dim=-3, index=0)
    imag_2 = op_2.select(dim=-3, index=1)

    real_out = torch.matmul(real_1, real_2) - torch.matmul(imag_1, imag_2)
    imag_out = torch.matmul(real_1, imag_2) + torch.matmul(imag_1, real_2)

    return torch.stack([real_out, imag_out], dim=-3)
