import torch


def cuda_info():
    if torch.cuda.is_available():
        print('CUDA is available.')
        print(f'Number of devices: {torch.cuda.device_count()}')
        print(f'Device name: {torch.cuda.get_device_name()}')
        print(f'Compute capability: {torch.cuda.get_device_capability()}')
    else:
        print('CUDA is not available.')

