from collections import OrderedDict
from typing import Optional

import torch
from unitair.states import innerprod, count_qubits
from unitair.utils import int_to_bin


def measure(
        state: torch.Tensor,
        num_samples: int,
        raw_output: bool = False
):
    """Draw samples from the probability distribution of a given state.

    This is a very basic measurement simulation. The argument `psi` is a
    quantum state in vector layout from which the probability distribution
    for measuring in the computational basis is determined. We then draw
    the specified number of samples.

    This measurement function does not simulate the effect of measurement
    on the quantum state itself. Such simulation is not currently planned
    for unitair.

    By default, this function returns a MeasurementHistogram object
    which stores a sorted histogram of all measurements. However,
    see the raw_output argument.

    Args:
        state: The state to be measured in vector layout.
        num_samples: The number of samples to take (the state is unchanged
            after each measurement).

        raw_output: When False, the output is a more "human-readable" format.
            When True, output is a simple dict {i: count(i)} where i is an
            integer rather than a binary string. The integer i is the number
            corresponding to the binary string. For example, instead of
            '0011', we would have the int i = 3.
    """

    num_qubits = count_qubits(state)
    probs = innerprod.abs_squared(state)
    dist = torch.distributions.Categorical(probs=probs)
    samples = dist.sample([num_samples])

    unsorted_histogram = dict()
    distinct_configurations = set(samples.tolist())

    for config in distinct_configurations:
        count = (samples == config).sum().item()
        if raw_output:
            config_specification = config
        else:
            config_specification = int_to_bin(
                config, num_bits=num_qubits, str_output=True
            )
        unsorted_histogram[config_specification] = count

    if raw_output:
        return unsorted_histogram

    configs, counts = zip(*unsorted_histogram.items())
    sorted_counts, permutation = torch.sort(torch.tensor(counts),
                                            descending=True)

    histogram = MeasurementHistogram(num_qubits=num_qubits)
    for index in permutation:
        histogram[configs[index]] = counts[index]

    return histogram


class MeasurementHistogram:
    """Record of a measurement of a given number of qubits."""
    def __init__(
            self,
            num_qubits: int,
            histogram: Optional[OrderedDict] = None
    ):
        """

        """
        if histogram is None:
            histogram = OrderedDict()

        self.num_qubits = num_qubits
        self.histogram = histogram
        self._int_key_histogram = None

    def __getitem__(self, item: str):
        try:
            return self.histogram.__getitem__(item)
        except KeyError:
            if set(item).issubset({'0', '1'}) and len(item) == self.num_qubits:
                return 0
            else:
                raise KeyError(
                    f'Given key {item} is not a valid binary string for '
                    f'{self.num_qubits} bits.'
                )

    def __setitem__(self, key, value):
        self.histogram.__setitem__(key, value)

    @property
    def num_distinct_samples(self):
        return len(self.histogram)

    @property
    def observed_samples(self):
        return set(self.histogram)

    def int_key_histogram(self):
        if self._int_key_histogram is None:
            self._int_key_histogram = OrderedDict()
            factors = [
                2 ** (self.num_qubits - i - 1) for i in range(self.num_qubits)
            ]
            for config, count in self.histogram.items():
                int_key = sum([int(x) * f for x, f in zip(config, factors)])
                self._int_key_histogram[int_key] = count
        return self._int_key_histogram

    def __repr__(self):
        if self.num_distinct_samples < 50:
            short_hist = list(self.histogram.items())
            shortened = False
        else:
            short_hist = list(self.histogram.items())[:15]
            shortened = True
        short_hist = OrderedDict(short_hist)

        out = ''
        for config, count in short_hist.items():
            out += config + ': ' + str(count) + '\n'
        if shortened:
            out += f' ... ({self.num_distinct_samples - 15} lines omitted)'
        else:
            out = out[:-1]  # clip an extra \n
        return out
