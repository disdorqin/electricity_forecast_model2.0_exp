import torch
import os
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device count:', torch.cuda.device_count())
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('Running on CPU')
print('CPU count:', os.cpu_count())
