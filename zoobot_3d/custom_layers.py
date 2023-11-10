# import torch
import torch.nn as nn

# Downsampling block for the Encoder
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, 3,
                              stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# Upsampling Block for the Decoder
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4,
                                stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

# Conv Block for ResNet
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, 3,
                              stride=1, padding='same')
        self.batchnorm = nn.BatchNorm2d(out_channels)

        self.block = nn.Sequential(self.conv,
                                   self.batchnorm)

    def forward(self, x):

        return self.block(x)

# Resnet Block
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.block1 = ConvBlock(in_channels, out_channels)
        self.act1 = nn.Mish()
        self.block2 = ConvBlock(out_channels, out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding='same') if in_channels != out_channels else nn.Identity()
        self.act2 = nn.Mish()

    def forward(self, x):
        h = self.block1(x)
        h = self.act1(h)
        h = self.block2(h)
        return self.act2(h + self.res_conv(x))
