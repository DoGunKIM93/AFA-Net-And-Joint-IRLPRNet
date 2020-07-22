
import torch
import numpy as np
import time



tnsr = torch.rand((200,3,1024,1024))
tnsr_cuda = tnsr.cuda()

a = time.perf_counter()

for i in range(1):
    #torch.save(tnsr_cuda, "a.pt")

    #np.save('a.pt', tnsr_cuda.cpu().numpy())
    for j in range(200):
        np.save(f'a{j}.pt', tnsr_cuda[j,:,:,:].cpu().numpy())

print(time.perf_counter() - a)


a = time.perf_counter()

for i in range(1):
    #tnsr = torch.load("a.pt", map_location = torch.device('cuda'))

    #tnsr = torch.tensor(np.load('a.pt.npy', mmap_mode='w+')).cuda()

    tnsrList = []
    for j in range(200):
        tnsrList.append(torch.tensor(np.load(f'a{j}.pt.npy', mmap_mode='w+')).cuda())
    
print(time.perf_counter() - a)