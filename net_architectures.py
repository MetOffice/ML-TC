#This module contains the different GAN architectures used in the project
#Each network class contains a subclass for the Generator and a subclass for the Discriminator
#Throughout the file nz represents the length of the random vector passed to the Generator, and nc represents the number of channels used

import torch
import torch.nn as nn

#https://discuss.pytorch.org/t/how-do-i-print-output-of-each-layer-in-sequential/5773/3
class PrintLayer(nn.Module):
    def __init__(self,f):
        self.f = f
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        if self.f:
            print(x.shape)
        return x

#256 by 256 network
class Large_Net:
    class Discriminator(nn.Module):
        def __init__(self, ngpu, nc, f=False):
            super(Large_Net.Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                PrintLayer(f),
                # input is (nc) x 256 x 256
                nn.Conv2d(nc, 8, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d(2,1),
                PrintLayer(f),
                # state size. 8 x 127 x 127
                nn.Conv2d(8,16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d(2,1),
                PrintLayer(f),
                # state size. 16 x 62 x 62
                nn.Conv2d(16,32, 4, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d(2,1),
                PrintLayer(f),
                # state size. 32 x 30 x 30
                nn.Conv2d(32,64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d(2,1),
                PrintLayer(f),
                # state size. 64 x 14 x 14
                nn.Conv2d(64,128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d(2,1),
                PrintLayer(f),
                # state size. 128 x 6 x 6
                nn.Conv2d(128,256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d(2,1),
                PrintLayer(f),
                # state size. 256 x 2 x 2
                nn.Flatten(),
                nn.Linear(1024,128, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                PrintLayer(f),
                # state size. 128
                nn.Linear(128,1, bias=False),
                nn.Sigmoid(),
                PrintLayer(f)
                # state size. 1
            )

        def forward(self, input):
            return self.main(input)

    class Generator(nn.Module):
        def __init__(self, ngpu,nc, nz,f=False):
            super(Large_Net.Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                PrintLayer(f),
                # input is 100 x 1 x 1, going into a convolution
                nn.ConvTranspose2d( nz, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                PrintLayer(f),
                # state size. (512) x 4 x 4
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                PrintLayer(f),
                # state size. (256) x 8 x 8
                nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                PrintLayer(f),
                # state size. (128) x 16 x 16
                nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                PrintLayer(f),
                # state size. (64) x 32 x 32
                nn.ConvTranspose2d( 64,32, 4, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                PrintLayer(f),
                # state size. 32 x 64 x 64
                nn.ConvTranspose2d( 32,16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                PrintLayer(f),
                # state size. 16 x 128 x 128
                nn.ConvTranspose2d( 16,8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                PrintLayer(f),
                # state size. 8 x 256 x 256
                nn.Conv2d( 8,nc, 3, 1, 1, bias=False),
                nn.Tanh(),
                PrintLayer(f)
                # state size. 1 x 256 x 256
            )
        def forward(self, input):
            return self.main(input)

#64 by 64 network, based on https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Small_Net:
    class Discriminator(nn.Module):
        def __init__(self, ngpu, nc, f=False):
            super(Small_Net.Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                PrintLayer(f),
                # input is 1 x 64 x 64
                nn.Conv2d(nc, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                PrintLayer(f),
                # state size. 64 x 32 x 32
                nn.Conv2d(64, 64*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.LeakyReLU(0.2, inplace=True),
                PrintLayer(f),
                # state size. 128 x 16 x 16
                nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 4),
                nn.LeakyReLU(0.2, inplace=True),
                PrintLayer(f),
                # state size. 256 x 8 x 8
                nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 8),
                nn.LeakyReLU(0.2, inplace=True),
                PrintLayer(f),
                # state size. 512 x 4 x 4
                nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
                PrintLayer(f)
                # state size. 1
            )

        def forward(self, input):
            return self.main(input)

    class Generator(nn.Module):
        def __init__(self, ngpu, nc, nz, f=False):
            super(Small_Net.Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                PrintLayer(f),
                # input is 100 x 1 x 1, going into a convolution
                nn.ConvTranspose2d( nz, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                PrintLayer(f),
                # state size. 512 x 4 x 4
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                PrintLayer(f),
                # state size. 256 x 8 x 8
                nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                PrintLayer(f),
                # state size. 128 x 16 x 16
                nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                PrintLayer(f),
                # state size. 64 x 32 x 32
                nn.ConvTranspose2d( 64,nc, 4, 2, 1, bias=False),
                nn.Tanh(),
                PrintLayer(f)
                # state size. 1 x 64 x 64
            )

        def forward(self, input):
            return self.main(input)

