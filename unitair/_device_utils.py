
import torch
from typing import Union


class _DefaultDevice:
    device_selected: torch.device

    def __init__(self):
        self.device_selected = self.standard_default_device()

    @staticmethod
    def standard_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')

    def set_default_device(self, identifier: Union[torch.device, str]):
        if isinstance(identifier, str):
            self.device_selected = torch.device(identifier)
        elif isinstance(identifier, torch.device):
            self.device_selected = identifier
        else:
            raise TypeError(
                "Device must be specified by a torch.device object or a valid "
                "string identifier like 'cpu' or 'cuda:0'."
            )


def cuda_info():
    if torch.cuda.is_available():
        print('CUDA is available.')
        print(f'Number of devices: {torch.cuda.device_count()}')
        print(f'Device name: {torch.cuda.get_device_name()}')
        print(f'Compute capability: {torch.cuda.get_device_capability()}')
    else:
        print('CUDA is not available. By default, CPU will be used.')

