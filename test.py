import torch
import torch.nn as nn
import torch.nn.functional as F



'''
def up(x, S):
    r = []
    for c in range(x.size(1)):
        a = torch.ones(1, 1, S, S) * x[0, c, x.size(2)//2, x.size(3)//2]
        r.append(a)
    return torch.cat(r,1)






N = 3
C = 3
P = 5
S = 2





H = P
W = P

x = torch.range(1,N*C*H*W)
x = x.view(N,C,H,W)

print(x)

UNFOLD = nn.Unfold(P)
PAD = nn.ReflectionPad2d(P//2)
ß
x = PAD(x)
x = UNFOLD(x) # N C*5*5 5*5

print(x.size())

x = torch.cat(x.split(P*P, dim=1), 2).view(N*P*P,C*P*P,1,1)
x = torch.cat(x.split(P*P, dim=1), 3).view(N*P*P,P,P,C).permute(0,3,1,2)
#x = x.view(N, C, 25, 5, 5)
#x = x.view(N*25, C, 5, 5)
print(x.size())
print(x)

r = []
for i in range(x.size(0)):
    r.append(up(x[i:i+1,:,:,:], S))
    #print(x[i:i+1,:,:,:], up(x[i:i+1,:,:,:]))

# = torch.cat(r, 0)

r = torch.cat(r, 3).split(S, dim=3)
r = torch.cat(r, 1).split(C, dim=1)
r = torch.cat(r, 0).view(N,P*P,C,S,S).split(P, dim=1)#.permute(0,3,2,1)#.split(10, dim=2)
r = torch.cat(r, 3).split(1, dim=1)#.split(#.view(2,3,10,10).transpose(2,3)
r = torch.cat(r, 4).squeeze(2).squeeze(1)ßß
print(r.size())

#r = r.view(N, C, 5, 2, 10)





print(r)
'''

a = [1]
b = [2]
c = [3]

l = [a[0], b[0], c[0]]

print(l)

a[0] = 2

print(l)