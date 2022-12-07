import torch
import torch.nn as nn

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
        self.layers = [ConvBlock(3, 16), 
                        ConvBlock(16, 64),
                        ConvBlock(64, 128),
                        ConvBlock(128, 256)]
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.maxpool(x)
        return x
class Decoder(nn.Module):
    def __init__(self):
        self.layers = [DeconvBlock(256, 128), 
                        DeconvBlock(128, 64),
                        DeconvBlock(64, 16),
                        DeconvBlock(16, 3)]
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='same'),
        self.bn = nn.BatchNorm2d(out_channels),
        self.tanhexp = TanhExp()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.tanhexp(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, padding_mode='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.tanhexp = TanhExp()
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.tanhexp(x)
        return x

class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size = 3, padding=1, padding_mode='same')
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.tanhexp = TanhExp()
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size = 3, padding=1, padding_mode='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.tanhexp(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanhexp(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x


class UNet_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 64, 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)
        self.TCB6 = TwoConvBlock(1024, 512, 512)
        self.TCB7 = TwoConvBlock(512, 256, 256)
        self.TCB8 = TwoConvBlock(256, 128, 128)
        self.TCB9 = TwoConvBlock(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4= UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 4, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return x


class TanhExp(nn.Module):
    def __init__(self):
        super(TanhExp, self).__init__()
    def formard(x):
        return x*torch.tanh(torch.exp(x))