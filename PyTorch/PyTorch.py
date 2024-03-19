import sys
import torch
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
print(f'Welcome to Torch,\nYou are using version {sys.version}')
print(f'PyTorch version {torch.__version__}')

## Introduction to Tensors
# Scalar
# source: https://pytorch.org/docs/stable/tensors.html
scalar = torch.tensor(17)
print(f'Scalar is {scalar}')
print(f'scaral has {scalar.ndim} dimenstions')
print(f'scalar content is {scalar.item()}')
print(f'scalar shape is {scalar.shape}')

# Vector
vector = torch.tensor([3,4])
print(f'"vector" has dimension of {vector.ndim}')
print(f'"vector" content is [{vector[0].item()}, {vector[1].item()}]')
print(f'"vector" shape is {vector.shape}')

# matrix
matrix = torch.tensor([[3,4],[5,6]])
print(f'"matrix" has dimension of {matrix.ndim}')
print(f'"matrix" content is [{matrix[0][0].item()}, {matrix[0][1].item()}]')
print(f'"matrix" shape is {matrix.shape}')

# tensor
tens = torch.tensor([[[6,7,8,9],
                      [8,7,6,5],
                      [3,2,1,7]]])
print(f'"tens" has dimension of {tens.ndim}')
print(f'"tens" content is [{tens}]')
print(f'"tens" shape is {tens.shape}')

lst=[
                    [
                        [6,7,8,9],
                        [8,7,6,5],
                        [3,2,1,7]
                    ],
                    [
                        [16,17,18,19],
                        [81,17,16,15],
                        [13,12,11,71]
                    ],
                    [
                        ['66',67,88,99],
                        [81,17,66,65],
                        [3.3,1.2,1.1,7.1]
                    ]]
#print('-------------------------------------------------')
#print(lst)
#print('-------------------------------------------------')
tens2 = torch.tensor([
                    [
                        [6,7,8,9],
                        [8,7,6,5],
                        [3,2,1,7]
                    ],
                    [
                        [16,17,18,19],
                        [81,17,16,15],
                        [13,12,11,71]
                    ],
                    [
                        [66,67,88,99],
                        [81,17,66,65],
                        [3.3,1.2,1.1,7.1]
                    ]])
print(f'"tens2" has dimension of {tens2.ndim}')
print(f'"tens2" content is [{tens2}]')
print(f'"tens2" shape is {tens2.shape}')
#------------------------------------------------------------------------------
def namestr(obj):
    namespace = globals()
    name = [name for name in namespace if namespace[name] is obj]
    return name[0]
#------------------------------------------------------------------------------
def show_tensor (tensor, fShowContent=False):
    strName = namestr(tensor)
    print(f'"{strName}" has dimension of {tensor.ndim}')
    if (fShowContent):
        print(f'"{strName}" content is [{tensor}]')
    print(f'"{strName}" shape is {tensor.shape}')
#------------------------------------------------------------------------------
RT = torch.rand(5,6)
show_tensor(RT)

RT1 = torch.rand(5,5,6)
print(f'"RT1" has dimension of {RT1.ndim}')
print(f'"RT1" shape is {RT1.shape}')

RT2 = torch.rand(size=(100,100,3))
show_tensor(RT2)
#------------------------------------------------------------------------------
## Zeros and Ones
z = torch.zeros(4,5)
show_tensor(z, True)
o = torch.ones(4,5)
show_tensor(o, True)
r=torch.arange(start=7,end=60,step=1.2)
show_tensor(r, True)
zeros_like = torch.zeros_like(r)
show_tensor(zeros_like, True)
# specifying data type
ft = torch.tensor([2,4,6,8], dtype=torch.float16)
show_tensor(ft, True)
ft = torch.tensor([2,4,6,8], dtype=torch.int)
show_tensor(ft, True)
ft_int = torch.tensor([2,4,6,8], dtype=torch.int,device="cpu")
show_tensor(ft_int, True)
## Tensor agruments:
# dtype is the data type. e.g. torch.float32 (default), torch.int32, etc.
# device is the device on which the tensor lives. It can be a cpu, gpu, cuda, etc.
# require_grad tracks the tensor gradient when it goes through manipulation

ft_float16 = ft_int.type(torch.float16)
show_tensor(ft_float16, True)