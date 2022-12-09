'''
主にnn.Moduleを継承したネットワークを定義
'''
import torch
import torch.nn as nn
import activations

class CAE(nn.Module):
    def __init__(self, image_size=512):
        super(CAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = ConvBlock(3,16)
        self.layer2 = ConvBlock(16,64)
        self.layer3 = ConvBlock(64,128)
        self.layer4 = ConvBlock(128,256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = DeconvBlock(256, 128)
        self.layer2 = DeconvBlock(128, 64)
        self.layer3 = DeconvBlock(64, 16)
        self.layer4 = DeconvBlock(16, 3)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.tanhexp = activations.TanhExp()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.tanhexp(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.tanhexp = activations.TanhExp()
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.tanhexp(x)
        return x

