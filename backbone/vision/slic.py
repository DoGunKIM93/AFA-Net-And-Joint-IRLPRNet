import torch
import numpy as np

from fast_slic import Slic
from fast_slic.avx2 import SlicAvx2



def SLIC(input, numCompo, compactness, isAvx2 = False, colorMode = 'color'):

    rst = []
    for i in range(input.size(0)):
        input_t = np.ascontiguousarray((input[i,:,:,:] * 255).permute(1,2,0).int().cpu().numpy().astype(np.uint8))
        slic = Slic(num_components=numCompo, compactness=compactness) if isAvx2 is False else SlicAvx2(num_components=numCompo, compactness=compactness)
        assignment = slic.iterate(input_t) # Cluster Map
        rst.append( torch.tensor(assignment).view(1,1,*assignment.shape).repeat(1,3 if colorMode == 'color' else 1,1,1) )
    
    return torch.cat(rst,0).cuda()#, slic.slic_model.clusters #cluster map / The cluster information of superpixels.

