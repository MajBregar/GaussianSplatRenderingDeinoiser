import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()

        groups = min(groups, out_channels)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimpleAutoencoder720p(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()

        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)

        self.encoder2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)

        self.encoder3 = ConvBlock(base_channels * 4, base_channels * 4)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1)

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1)
        self.decoder3 = ConvBlock(base_channels * 4, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.decoder2 = ConvBlock(base_channels * 2, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.decoder1 = ConvBlock(base_channels, base_channels)

        self.output = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder1(x)
        x = self.down1(x)

        x = self.encoder2(x)
        x = self.down2(x)

        x = self.encoder3(x)
        x = self.down3(x)

        x = self.bottleneck(x)

        x = self.up3(x)
        x = self.decoder3(x)

        x = self.up2(x)
        x = self.decoder2(x)

        x = self.up1(x)
        x = self.decoder1(x)

        return self.output(x)