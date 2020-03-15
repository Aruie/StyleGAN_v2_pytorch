import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam


import logging as log


class Mappingnet(nn.Module) :
    def __init__(self) :
        super(Mappingnet, self).__init__()

        layers = []
        
        for i in range(8) : 
            layers.append(nn.Linear(512,512))
            layers.append(nn.LeakyReLU(0.2))
        
        self.mapping = nn.Sequential(*layers)
        # print(self.mapping)

    def forward(self, x) :
        y = self.mapping(x)
        return y


class Generator(nn.Module) :
    def __init__(self, seed = 42) :
        super(Generator, self).__init__()

        self.mapping = Mappingnet()

        np.random.seed(seed)

        self.const = torch.from_numpy(np.random.randn(1,512,4,4))

        self.style_blocks = nn.ModuleList()

        self.style_blocks.append(StyleBlock('UP', 'IOSKIP', 4, 512, 256))
        # self.style_blocks.append(StyleBlock('UP', 'IOSKIP', 8, 512, 512))
        # self.style_blocks.append(StyleBlock('UP', 'IOSKIP', 16, 512, 512))
        # self.style_blocks.append(StyleBlock('UP', 'IOSKIP', 32, 512, 256))
        # self.style_blocks.append(StyleBlock('UP', 'IOSKIP', 64, 256, 128))
        # self.style_blocks.append(StyleBlock('UP', 'IOSKIP', 128, 128, 64))
        # self.style_blocks.append(StyleBlock('UP', 'IOSKIP', 256, 64, 32))
        # self.style_blocks.append(StyleBlock('UP', 'IOSKIP', 512, 32, 16))
        
    def forward(self, z) :
        w = self.mapping(z)
        b, _ = w.shape
        
        x = self.const.expand(b,512,4,4)
        
        for module in self.style_blocks :
            x = module(x, w)
        
        return x



class ModulatedConv2d(nn.Module) :
    def __init__(self, input_channel, output_channel) :
        super(ModulatedConv2d, self).__init__()

        self.weight = torch.randn((output_channel, input_channel, 3,3), dtype = torch.float64, requires_grad = True)



    def forward(self, x, s) :
        

        a = torch.randn(10,20,30,40,50)
        print(a.permute(1,2,3,0,4).shape)


        s = s.unsqueeze(1).unsqueeze(3).unsqueeze(4)

        weight = self.weight.unsqueeze(0)

        print(s.shape, weight.shape)

        ## modulator
        weight = weight * s
        print(f's *weight shape : {weight.shape}')

        batch, out = weight.shape[0], weight.shape[1]

        # demodulayer
        d = torch.sqrt(torch.sum(weight.pow(2), dim = (1,2,3), keepdim = True) + 1e-8)
        print(f'ddddddddddddddddddddddddddddd : {d.shape}')
        weight = weight * d

        print(f'ddddddddddddddddddddddddddddd : {weight.shape}')
        

        weight = weight.reshape(-1, weight.shape[2], weight.shape[3], weight.shape[4])

        print(f'weight shape : {weight.shape}')


        print(f'x shape : {x.shape}')
        x = x.reshape(1,-1,x.shape[2], x.shape[3])
        print(f'x shape : {x.shape}')
        x = nn.functional.conv2d(x, weight, padding = 1)

        print(f'x shape : {x.shape}')

        x = x.reshape( -1, out, x.shape[2], x.shape[3])

        print(x.shape)

        return x



class StyleBlock(nn.Module) :
    def __init__(self, direction, type, input_size, input_channel, output_channel) :
        super(StyleBlock, self).__init__()

        if direction == 'UP' :
            pass
        elif direction == 'DOWN' :
            pass

        else :
            log.error('StyleBlock : Wrong Type')
            return

        self.mapping = nn.Linear(512, input_channel)

        self.resize = nn.Upsample()
        self.conv = ModulatedConv2d(input_channel, output_channel)


    def forward(self, x, w) :
        
        s = self.mapping(w)
        
        x = self.conv(x, s)
        #weight = self.conv.weight
        #print(weight.shape, y.shape)
        


        
        # b, c = y.shape
        # y = y.unsqueeze(2).unsqueeze(3)

        # avg = torch.mean(x, dim = (2,3)).unsqueeze(2).unsqueeze(3)
        # std = torch.std(x, dim = (2,3)).unsqueeze(2).unsqueeze(3)
        # x = x / std - avg


        
        #print(f'{x.shape},  y_s = {y_s.shape}, y_b = {y_b.shape}')
        
        
        return x

'''
class IOSkip(nn.Module) :
    def __init__(self, input_size, input_channel = 0, up_size) :
        super(IOSkip, self).__init__()

        StyleBlock(4, 8)
        ToRGB(4)
        Upsample()
'''

class Discriminator(nn.Module) :
    None


class Criterion(nn.Module) :
    None

if __name__ == '__main__' : 
    gen = Generator(30)

    latent = torch.randn((8,512), dtype = torch.float32)
    y = gen(latent)

    print(y.shape)