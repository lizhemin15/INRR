import torch

if torch.cuda.is_available():
    cuda_if = True
else:
    cuda_if =False

cuda_num = 0
print('Cuda is ',cuda_if)
print('Cuda num is ',cuda_num)