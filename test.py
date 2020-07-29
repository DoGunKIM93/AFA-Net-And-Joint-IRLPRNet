import torch


tnsr = torch.tensor(range(10)).repeat(10,1)


print(tnsr[...,2:4,2:5])