'''
module.py
'''
version = '1.11.200624'

#from Python
import time
import csv
import os
import math
import numpy as np
import sys

#from Pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,grad
from torch.autograd import Function

#from this project
import param as p
import backbone.vision as vision



######################################################################################################################################################################## 

# Attention BLocks

######################################################################################################################################################################## 





## second-order Channel attention (SOCA)
class SecondOrderChannalAttentionBlock(nn.Module):
    def __init__(self, channel, dim=2, reduction=8, sub_sample=None):
        super(SecondOrderChannalAttentionBlock, self).__init__()

        assert dim in [2,3]

        self.dim = dim
        self.sub_sample = sub_sample



        if self.dim == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=2)

            # feature channel downscale and upscale --> channel weight
            self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
        elif self.dim == 3:
            self.max_pool = nn.MaxPool3d(kernel_size=2)

            # feature channel downscale and upscale --> channel weight
            self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )

    def forward(self, input):

        if self.sub_sample is None:
            x = input
        else:
            imode = 'bicubic'
            #x = F.interpolate(input.reshape(input.size(0), -1, *input.size()[-2:]), scale_factor=1/self.sub_sample, mode=imode, align_corners=False)
            #x = x.reshape(*input.size()[0:-2],*x.size()[-2:])
            x = F.max_pool3d(input, kernel_size=[1,self.sub_sample,self.sub_sample]) if self.dim == 3 else F.max_pool2d(input, kernel_size=[self.sub_sample,self.sub_sample])

        if self.dim==2:
            batch_size, C, h, w = x.shape  # x: BxCxHxW
            N = int(h * w)
        elif self.dim==3:
            batch_size, C, s, h, w = x.shape  # x: BxCxNxHxW
            N = int(s * h * w)


        if self.dim==2:

            ## MPN-COV
            cov_mat = CovpoolLayer(x) # Global Covariance pooling layer
            #cov_mat_sqrt = SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
            ##
            cov_mat_sum = torch.mean(cov_mat,1)
            cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)

        elif self.dim==3:
            

            ## MPN-COV
            cov_mat = CovpoolLayer3d(x) # Global Covariance pooling layer
            #cov_mat_sqrt = SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
            #cov_mat_sqrt = torch.sqrt(cov_mat/10000)  ######### NaN 문제로 삭제
            #print("  CA cov_sq", cov_mat_sqrt.mean())
            ##
            cov_mat_sum = torch.mean(cov_mat,1)
            cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1,1)

        

        y_cov = self.conv_du(cov_mat_sum)


        return y_cov*input




## non_local module
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, r=8, mode='second_order_embedded_gaussian',
                 sub_sample=None, bn_layer=True):
        super(NonLocalBlock, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['second_order_embedded_gaussian']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // r
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            #sub_sample = nn.Upsample
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None
        
        if mode in ['second_order_embedded_gaussian']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'second_order_embedded_gaussian':
                self.operation_function = self._second_order_embedded_gaussian

        '''
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))
        '''

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    # https://arxiv.org/pdf/1909.00295.pdf
    def _second_order_embedded_gaussian(self, input):

        if self.sub_sample is None:
            x = input
        else:
            imode = 'bicubic'
            #x = F.interpolate(input.reshape(input.size(0), -1, *input.size()[-2:]), scale_factor=1/self.sub_sample, mode=imode, align_corners=False)
            #x = x.reshape(*input.size()[0:-2],*x.size()[-2:])
            x = F.max_pool3d(input, kernel_size=[1,self.sub_sample,self.sub_sample]) if self.dimension == 3 else F.max_pool2d(input, kernel_size=[self.sub_sample,self.sub_sample])

        if self.dimension == 2:
            batch_size, C, H, W = x.shape
            M = H * W
        elif self.dimension == 3:
            batch_size, C, N, H, W = x.shape
            M = N * H * W

        #print(M, self.inter_channels)
        #print(self.g)
        ##
        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        #g_x = g_x.permute(0, 2, 1)

        #print(g_x.size())


        theta_x_T = self.theta(x).view(batch_size, self.inter_channels, -1) # b, 0.5c, thw
        theta_x = theta_x_T.permute(0, 2, 1) # b, thw, 0.5c

        I_hat = (-1. / M / M) * torch.ones(M , M , device = x.device) + (1. / M) * torch.eye(M , M , device = x.device)
        I_hat = I_hat.view(1 , M , M).repeat(batch_size , 1 , 1).type(x.dtype) # b, thw, thw

        sigma = theta_x_T.bmm(I_hat).bmm(theta_x) # b, 0.5c, 0.5c
        sigma /= math.sqrt(self.inter_channels)
        sigma_div_C = F.softmax(sigma, dim=-1)
        
        # return f_div_C
        # (b, 0.5c, 0.5c)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(sigma_div_C, g_x)


        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        


        if self.sub_sample is None:
            z = W_y + input
        else:
            z = F.interpolate(W_y.reshape(W_y.size(0), -1, *W_y.size()[-2:]), size = input.size()[-2:], mode=imode, align_corners=False).reshape(input.size()) + input 

        return z



class CrissCrossAttention(nn.Module):
    def __init__(self,CW,r,dim=2):
        super(CrissCrossAttention, self).__init__()
        

        self.dim = dim

        if self.dim == 2:
            self.query_conv = nn.Conv2d(in_channels=CW, out_channels=CW//r, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels=CW, out_channels=CW//r, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels=CW, out_channels=CW, kernel_size=1)
            self.softmax = nn.Softmax(dim=3)
            self.IN = nn.InstanceNorm2d(CW)
        elif self.dim == 3:
            self.query_conv = nn.Conv3d(in_channels=CW, out_channels=CW//r, kernel_size=1)
            self.key_conv = nn.Conv3d(in_channels=CW, out_channels=CW//r, kernel_size=1)
            self.value_conv = nn.Conv3d(in_channels=CW, out_channels=CW, kernel_size=1)
            self.softmax = nn.Softmax(dim=4)
            self.IN = nn.InstanceNorm3d(CW)

        
        self.gamma = nn.Parameter(torch.zeros(1))
        

    def INF(self,B,H,W):
        return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1) 

    def INF3dH(self,B,S,H,W):
        return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*S*W,1,1) 

    def INF3dS(self,B,S,H,W):
        rtn = -torch.diag(torch.tensor(float("inf")).cuda().repeat(S),0).unsqueeze(0).repeat(B*H*W,1,1) 
        return rtn

    def CC3d(self, x):
        m_batchsize, _, seq, height, width = x.size()
        #print("MBSZ", m_batchsize)
        proj_query = self.query_conv(x)
        #print("CC3 1", proj_query.mean(), proj_query.max(), proj_query.min())

        proj_query_S = proj_query.permute(0,3,4,1,2).contiguous().view(m_batchsize * height * width, -1, seq).permute(0, 2, 1)
        proj_query_H = proj_query.permute(0,2,4,1,3).contiguous().view(m_batchsize * width * seq, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,3,1,4).contiguous().view(m_batchsize * height * seq, -1, width).permute(0, 2, 1)

        proj_key = self.key_conv(x)
        #print("CC3 2", proj_key.mean(), proj_key.max(), proj_key.min())

        proj_key_S = proj_key.permute(0,3,4,1,2).contiguous().view(m_batchsize * height * width, -1, seq)
        proj_key_H = proj_key.permute(0,2,4,1,3).contiguous().view(m_batchsize * width * seq, -1, height)
        proj_key_W = proj_key.permute(0,2,3,1,4).contiguous().view(m_batchsize * height * seq, -1, width)

        proj_value = self.value_conv(x)
        #print("CC3 3", proj_value.mean(), proj_value.max(), proj_value.min())

        proj_value_S = proj_value.permute(0,3,4,1,2).contiguous().view(m_batchsize * height * width, -1, seq)
        proj_value_H = proj_value.permute(0,2,4,1,3).contiguous().view(m_batchsize * width * seq, -1, height)
        proj_value_W = proj_value.permute(0,2,3,1,4).contiguous().view(m_batchsize * height * seq, -1, width)

        #print("CC3 3-1", proj_value_S.mean(), proj_value_S.max(), proj_value_S.min())
        #print("CC3 3-1", proj_value_H.mean(), proj_value_H.max(), proj_value_H.min())

        #print("CC3 3-1", proj_value_W.mean(), proj_value_W.max(), proj_value_W.min())

        energy_S = (torch.matmul(proj_query_S, proj_key_S)+self.INF3dS(m_batchsize, seq, height, width)).view(m_batchsize,height,width,seq,seq).permute(0,3,1,2,4)      
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF3dH(m_batchsize, seq, height, width)).view(m_batchsize,seq,width,height,height).permute(0,1,3,2,4)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,seq,height,width,width)


        concate = self.softmax(torch.cat([energy_S, energy_H, energy_W], 4))
        #concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()

        #print("CC3 3-2", energy_S.mean(), energy_S.max(), energy_S.min())
        #print("CC3 3-2", energy_H.mean(), energy_H.max(), energy_H.min())
        #print("CC3 3-2", energy_W.mean(), energy_W.max(), energy_W.min())

        att_S = concate[:,:,:,:,0:seq].permute(0,2,3,1,4).contiguous().view(m_batchsize*width*height,seq,seq)
        att_H = concate[:,:,:,:,seq:seq+height].permute(0,1,3,2,4).contiguous().view(m_batchsize*seq*width,height,height)
        att_W = concate[:,:,:,:,seq+height:seq+height+width].contiguous().view(m_batchsize*seq*height,width,width)

        #print("CC3 4", att_S.mean(), att_S.max(), att_S.min())
        #print("CC3 4", att_H.mean(), att_H.max(), att_H.min())
        #print("CC3 4", att_W.mean(), att_W.max(), att_W.min())

        maxMBSize = 40960

        proj_value_S_Chunks = proj_value_S.split(maxMBSize, dim=0)
        att_S_Chunks = att_S.split(maxMBSize, dim=0)

        out_S_tmp = []
        for proj_value_S, att_S in zip(proj_value_S_Chunks, att_S_Chunks):
            out_S_tmp.append(torch.bmm(proj_value_S, att_S.permute(0, 2, 1)))
        out_S = torch.cat(out_S_tmp,0).view(m_batchsize,height,width,-1,seq).permute(0,3,4,1,2)

        #out_S = torch.bmm(proj_value_S, att_S.permute(0, 2, 1)).view(m_batchsize,height,width,-1,seq).permute(0,3,4,1,2)#torch.bmm(proj_value_S, att_S.permute(0, 2, 1)).view(m_batchsize,height,width,-1,seq).permute(0,3,4,1,2)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,seq,width,-1,height).permute(0,3,1,4,2)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,seq,height,-1,width).permute(0,3,1,2,4)

        #print("CC3 5", out_S.mean(), out_S.max(), out_S.min())
        #print("CC3 5", out_H.mean(), out_H.max(), out_H.min())
        #print("CC3 5", out_W.mean(), out_W.max(), out_W.min())

        #print(out_H.size(),out_W.size())
        out_sum = self.IN(out_S + out_H + out_W)
        return self.gamma*(out_sum) + x

    def CC2d(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)

        #print("CC2 1", proj_query.mean())

        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)

        #print("CC2 2", proj_key.mean())

        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)

        #print("CC2 3", proj_value.mean())

        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        #print("CC2 4", concate.mean())

        #concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)

        #print("CC2 5", att_H.mean())

        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)

        #print("CC2 6", out_W.mean())

        #print(out_H.size(),out_W.size())
        out_sum = self.IN(out_W + out_H)
        return self.gamma*(out_sum) + x

    def forward(self, x):
        if self.dim == 2:
            return self.CC2d(x)
        elif self.dim == 3:
            return self.CC3d(x)

















######################################################################################################################################################################## 

# Covariance pooling 

######################################################################################################################################################################## 


class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,h,w)
         return grad_input

class Covpool3d(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         n = x.data.shape[2]
         h = x.data.shape[3]
         w = x.data.shape[4]
         M = h*w*n
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         n = x.data.shape[2]
         h = x.data.shape[3]
         w = x.data.shape[4]
         M = h*w*n
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,n,h,w)
         return grad_input

class Sqrtm(Function):
     @staticmethod
     def forward(ctx, input, iterN):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad = False, device = x.device)
         Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,iterN,1,1)
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
                ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
                Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
                Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            ZY = 0.5*Y[:,iterN-2,:,:].bmm(I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]))
         y = ZY*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, ZY, normA, Y, Z)
         ctx.iterN = iterN
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         der_postCom = grad_output*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(2*torch.sqrt(normA))
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_sacleTrace))
         else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
                          Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
               ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) - 
                         Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) - 
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) - 
                         Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         grad_input = der_NSiter.div(normA.view(batchSize,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             grad_input[i,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *torch.ones(dim,device = x.device).diag()
         return grad_input, None

class Sqrtm3d(Function):
     @staticmethod
     def forward(ctx, input, iterN):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         Y = torch.zeros(batchSize, iterN, dim, dim, dim, requires_grad = False, device = x.device)
         Z = torch.eye(dim,dim,dim,device = x.device).view(1,dim,dim,dim).repeat(batchSize,iterN,1,1,1)
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:,:] = A.bmm(ZY)
            Z[:,0,:,:,:] = ZY
            for i in range(1, iterN-1):
               ZY = 0.5*(I3 - Z[:,i-1,:,:,:].bmm(Y[:,i-1,:,:,:]))
               Y[:,i,:,:,:] = Y[:,i-1,:,:,:].bmm(ZY)
               Z[:,i,:,:,:] = ZY.bmm(Z[:,i-1,:,:,:])
            ZY = 0.5*Y[:,iterN-2,:,:,:].bmm(I3 - Z[:,iterN-2,:,:,:].bmm(Y[:,iterN-2,:,:,:]))
         y = ZY*torch.sqrt(normA).view(batchSize, 1, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, ZY, normA, Y, Z)
         ctx.iterN = iterN
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         der_postCom = grad_output*torch.sqrt(normA).view(batchSize, 1, 1, 1).expand_as(x)
         der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(2*torch.sqrt(normA))
         I3 = 3.0*torch.eye(dim,dim,dim,device = x.device).view(1, dim, dim, dim).repeat(batchSize,1,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_sacleTrace))
         else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:,:].bmm(Z[:,iterN-2,:,:,:])) -
                          Z[:,iterN-2,:,:,:].bmm(Y[:,iterN-2,:,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:,:].bmm(Z[:,i,:,:,:])
               ZY = Z[:,i,:,:,:].bmm(Y[:,i,:,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) - 
                         Z[:,i,:,:,:].bmm(dldZ).bmm(Z[:,i,:,:,:]) - 
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) - 
                         Y[:,i,:,:,:].bmm(dldY).bmm(Y[:,i,:,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         grad_input = der_NSiter.div(normA.view(batchSize,1,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             grad_input[i,:,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *torch.ones(dim,device = x.device).diag()
         return grad_input, None

class Triuvec(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         x = x.reshape(batchSize, dim*dim)
         I = torch.ones(dim,dim).triu().t().reshape(dim*dim)
         index = I.nonzero()
         y = torch.zeros(batchSize,dim*(dim+1)/2,device = x.device)
         for i in range(batchSize):
            y[i, :] = x[i, index].t()
         ctx.save_for_backward(input,index)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,index = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         grad_input = torch.zeros(batchSize,dim,dim,device = x.device,requires_grad=False)
         grad_input = grad_input.reshape(batchSize,dim*dim)
         for i in range(batchSize):
            grad_input[i,index] = grad_output[i,:].reshape(index.size(),1)
         grad_input = grad_input.reshape(batchSize,dim,dim)
         return grad_input

def CovpoolLayer(var):
    return Covpool.apply(var)

def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)

def CovpoolLayer3d(var):
    return Covpool3d.apply(var)

def SqrtmLayer3d(var, iterN):
    return Sqrtm3d.apply(var, iterN)

def TriuvecLayer(var):
    return Triuvec.apply(var)

