import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.module.module import resBlock


class VESPCN(nn.Module):
    def __init__(self, upscale_factor, sequenceLength, colorMode = 'color'):
        super(VESPCN, self).__init__()
        
        self.upscale_factor = upscale_factor
        self.sequenceLength = sequenceLength

        inputCh = (1 if colorMode =='grayscale' else 3) * sequenceLength

        self.conv1 = nn.Conv2d(inputCh, 256, (9, 9), (1, 1), (4, 4))

        self.res1 = resBlock(256, windowSize=3)
        self.res2 = resBlock(256, windowSize=3)

        self.conv2 = nn.Conv2d(256, 128, (5, 5), (1, 1), (2, 2))

        self.res3 = resBlock(128, windowSize=3)
        self.res4 = resBlock(128, windowSize=3)
        self.res5 = resBlock(128, windowSize=3)
        self.res6 = resBlock(128, windowSize=3)

        self.res7 = resBlock(128, windowSize=3)
        self.res8 = resBlock(128, windowSize=3)
        self.res9 = resBlock(128, windowSize=3)
        self.res10 = resBlock(128, windowSize=3)

        self.conv3 = nn.Conv2d(128, (1 if colorMode =='grayscale' else 3) * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        sc = x[:,self.sequenceLength//2,:,:,:]
        x = torch.cat(x.split(1, dim=1),2).squeeze(1)

        x = F.leaky_relu(self.conv1(x), 0.2)

        res = x
        x = F.leaky_relu(self.res1(x), 0.2)
        x = F.leaky_relu(self.res2(x) + res, 0.2)

        x = F.leaky_relu(self.conv2(x), 0.2)
        
        res = x
        x = F.leaky_relu(self.res3(x), 0.2)
        x = F.leaky_relu(self.res4(x), 0.2)
        x = F.leaky_relu(self.res5(x), 0.2)
        x = F.leaky_relu(self.res6(x) + res, 0.2)

        res = x
        x = F.leaky_relu(self.res7(x), 0.2)
        x = F.leaky_relu(self.res8(x), 0.2)
        x = F.leaky_relu(self.res9(x), 0.2)
        x = F.leaky_relu(self.res10(x) + res, 0.2)
        
        x = F.tanh(self.pixel_shuffle(self.conv3(x))) + F.interpolate(sc, scale_factor=self.upscale_factor, mode='bicubic')
        return x