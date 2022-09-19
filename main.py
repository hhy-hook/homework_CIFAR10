import torch
print(torch.cuda.device_count())
print(torch.version.cuda)
print(torch.backends.cudnn.enabled)