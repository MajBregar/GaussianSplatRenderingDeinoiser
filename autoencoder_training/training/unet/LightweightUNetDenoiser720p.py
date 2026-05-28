import torch
import torch.nn as nn


class SepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        norm_groups = min(groups, out_channels)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.block = SepConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.block = SepConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class LightweightUNetDenoiser720p(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_channels=24):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.input = SepConvBlock(in_channels, c1)

        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)

        self.bottleneck = SepConvBlock(c4, c4)

        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.output = nn.Sequential(
            nn.Conv2d(c1, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s1 = self.input(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)

        x = self.down3(s3)
        x = self.bottleneck(x)

        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        return self.output(x)