import torch
from unitair.states import innerprod, count_qubits
from unitair.utils import int_to_bin


def measure(
        state: torch.Tensor,
        num_samples: int,
        str_output: bool = False
):
    num_qubits = count_qubits(state)
    probs = innerprod.abs_squared(state)
    dist = torch.distributions.Categorical(probs=probs)
    samples = dist.sample([num_samples])
    if not str_output:
        output = torch.zeros(size=(num_samples, num_qubits), dtype=torch.bool)
        for i in range(num_samples):
            output[i] = int_to_bin(
                samples[i], num_qubits, str_output=False, device=state.device)
    else:
        output = [
            int_to_bin(samples[i], num_qubits, str_output=True)
            for i in range(num_samples)
        ]

    return output
