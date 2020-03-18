import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

import torch.nn.functional as F

import logging as log


class Mappingnet(nn.Module) :
    def __init__(self) :
        super(Mappingnet, self).__init__()

        layers = []
        
        for i in range(8) : 
            layers.append(nn.Linear(512,512))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.mapping = nn.Sequential(*layers)
        # print(self.mapping)

    def forward(self, x) :
        y = self.mapping(x)
        return y


class Generator(nn.Module) :
    def __init__(self, seed = 42) :
        super(Generator, self).__init__()
        
        # Define Mapping Network
        self.mapping = Mappingnet()

        # Define Constant Layer
        torch.manual_seed(seed)
        self.const = torch.randn([1,512,4,4], dtype = torch.float32)


        self.style_blocks = nn.ModuleList()

        self.style_blocks.append(StyleBlock(4, 512, 512))

        self.style_blocks.append(StyleBlock(4, 512, 512, direction = 'UP'))
        self.style_blocks.append(StyleBlock(8, 512, 512))

        self.style_blocks.append(StyleBlock(8, 512, 512, direction = 'UP'))
        self.style_blocks.append(StyleBlock(16, 512, 512))

        self.style_blocks.append(StyleBlock(16, 512, 512, direction = 'UP'))
        self.style_blocks.append(StyleBlock(32, 512, 512))

        self.style_blocks.append(StyleBlock(32, 512, 256, direction = 'UP'))
        self.style_blocks.append(StyleBlock(64, 256, 256))

        self.style_blocks.append(StyleBlock(64, 256, 128, direction = 'UP'))
        self.style_blocks.append(StyleBlock(128, 128, 128))

        # self.style_blocks.append(StyleBlock(128, 128, 64, direction = 'UP'))))
        # self.style_blocks.append(StyleBlock(256, 64, 64))

        # self.style_blocks.append(StyleBlock(256, 64, 32, direction = 'UP'))))
        # self.style_blocks.append(StyleBlock(512, 32, 32))

        # self.style_blocks.append(StyleBlock(512, 32, 16, direction = 'UP'))))
        # self.style_blocks.append(StyleBlock(1024, 16, 16))
        
    def forward(self, w) :
        
        b, _ = w.shape
        
        x = self.const.expand(b,512,4,4)
        sp = None

        for module in self.style_blocks :
            x, sp = module(x, w, sp)
        
        return sp


class StyleBlock(nn.Module) :
    def __init__(self, input_size, 
                input_channel, output_channel, direction = 'NO') :
        super(StyleBlock, self).__init__()

        # Resize
        assert direction in ['NO','UP']
        
        self.dir = direction

        if self.dir == 'UP' :
            self.resize = nn.Upsample(scale_factor=2)
        else :
            #self.to_rgb = nn.Conv2d(output_channel, 3, 1)
            self.to_rgb = ModulatedConv2d(output_channel, 3, 1, demodulation=False)

        # NoiseFactor Setup
        self.noise_factor = nn.Parameter(torch.zeros([1], requires_grad= True))

        # Mapping Net
        self.mapping = nn.Linear(512, input_channel)
        
        # ModulateConv and 
        self.conv = ModulatedConv2d(input_channel, output_channel, 3)
        self.bias = nn.Parameter(torch.zeros(output_channel, requires_grad = True).reshape(1,-1,1,1))

    def forward(self, x, w, skip_path) :
        
        # Mapping Network
        s = self.mapping(w)
        if self.dir != 'NO' :
            x = self.resize(x)

        x = self.conv(x, s)
        
        # Add Noise & Bias 
        noise = self.noise_factor * torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]))
        x += noise + self.bias

        x = F.leaky_relu(x, 0.2, inplace = True)
        
        # Skip Path
        if self.dir == 'NO' :
            if skip_path is None :
                skip_path = self.to_rgb(x, s)
            else :
                skip_path = skip_path + self.to_rgb(x, s)
            skip_path = F.leaky_relu(skip_path, 0.2, inplace=True)
                
        elif self.dir == 'UP' :
            skip_path = self.resize(skip_path)
                    
        return x, skip_path





class ModulatedConv2d(nn.Module) :
    def __init__(self, input_channel, output_channel, kennel_size, batch_to_channel = False, demodulation = True) :
        super(ModulatedConv2d, self).__init__()
        self.kennel_size = kennel_size
        self.demodulation = demodulation
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.weight = nn.Parameter(torch.randn((output_channel, input_channel, kennel_size, kennel_size), dtype = torch.float32, requires_grad = True))


    def forward(self, x, s) :
        
        batch = x.shape[0]
        
        # style shape from BI to BOIkk 
        s = s.unsqueeze(1).unsqueeze(3).unsqueeze(4)

        # weight add batch dims
        weight = self.weight.unsqueeze(0)

        ## modulator
        weight = weight * s

        # demodulayer
        if self.demodulation == True :
            d = torch.sqrt(torch.sum(weight.pow(2), dim = (2,3,4), keepdim = True) + 1e-8)
            weight = weight * d

        # batch to Convoultion Group
        weight = weight.reshape(-1, weight.shape[2], weight.shape[3], weight.shape[4])
        x = x.reshape(1,-1,x.shape[2], x.shape[3])

        # Group Convolution
        x = torch.conv2d(x, weight, padding = int(self.kennel_size / 2), groups = batch)
       
        # channel to Batch
        x = x.reshape( -1, self.output_channel, x.shape[2], x.shape[3])

        return x





class Discriminator(nn.Module) :
    def __init__(self) :
        super(Discriminator, self).__init__()
        
        # self.from_rgb = nn.Conv2d(3,16, 1)     
        # self.style_blocks.append(DBlock(1024, 16, 32))
        # self.style_blocks.append(DBlock(512, 32, 64))
        # self.style_blocks.append(DBlock(256, 64, 128))
        
        self.from_rgb = nn.Conv2d(3, 128, 1)     

        self.d_blocks = nn.ModuleList()
        self.d_blocks.append(DBlock(128, 128, 256))
        self.d_blocks.append(DBlock(64, 256, 512))
        self.d_blocks.append(DBlock(32, 512, 512))
        self.d_blocks.append(DBlock(16, 512, 512))
        self.d_blocks.append(DBlock(8, 512, 512))

        # need minimatch std add 
        self.stddev = MinibatchStdDev(4)
        self.conv1 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv2 = nn.Conv2d(512, 512, 4)
        self.out = nn.Linear(512, 1)
        
    def forward(self, x) :
        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        
        for module in self.d_blocks :
            x = module(x)
            x = F.leaky_relu(x, 0.2, inplace=True)    

        x = self.stddev(x)

        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = x.reshape([x.shape[0], -1])
        x = self.out(x)
        x = torch.sigmoid(x)
        
        return x




class DBlock(nn.Module) :
    def __init__(self, input_size, input_channel, output_channel) :
        super(DBlock, self).__init__()
        
        self.resize = nn.AvgPool2d(2)
        self.conv_skip = nn.Conv2d(input_channel, output_channel, 1)

        self.conv1 = nn.Conv2d(input_channel, input_channel, 3, padding = 1)
        self.conv2 = nn.Conv2d(input_channel, output_channel, 3, stride = 2, padding = 1)
        

    def forward(self, x) :
     
        path_skip = self.resize(x)
        path_skip = self.conv_skip(path_skip)

        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2, inplace=True)

        return x + path_skip


class MinibatchStdDev(nn.Module) :
    def __init__(self, group_size = 4) :
        super(MinibatchStdDev, self).__init__()
        self.group = group_size

    def forward(self, x) :
        b, c, h, w = x.shape
        # g = self.group if self.group <= b else b
        
        # Calc group std & feature mean
        y = torch.std(x, dim=(0), keepdim=True)
        y = torch.mean(y, dim=(1), keepdim=True)
        print(y)

        y = y.expand((b,1,h,w))


        return torch.cat((x, y), dim=1)

class Criterion(nn.Module) :
    None







if __name__ == '__main__' : 
    
    mapping = Mappingnet()
    gen = Generator(30)
    dis = Discriminator()

    criterion = nn.BCELoss()
    
    lr = 2e-3
    
    optim_map = Adam(mapping.parameters(), lr = lr / 100, betas=(0, 0.99))
    optim_gen = Adam(gen.parameters(), lr = lr, betas=(0, 0.99))
    optim_dis = Adam(dis.parameters(), lr = lr, betas=(0, 0.99))

    latent = torch.randn((2,512), dtype = torch.float32)

    w = mapping(latent)
    x_fake = gen(w)

    y_pred = dis(x_fake)

    y_true = torch.zeros([2,1])

    adv_loss = criterion(y_pred, y_true)
    
    optim_map.zero_grad()
    optim_gen.zero_grad()
    optim_dis.zero_grad()

    adv_loss.backward()

    optim_map.step()
    optim_gen.step()
    optim_dis.step()


