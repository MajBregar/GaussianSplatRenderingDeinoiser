
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path



def _norm(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, num_channels)



class ConvNormRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            _norm(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RecurrentBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvNormRelu(in_ch,           out_ch)
        self.conv2 = ConvNormRelu(out_ch + out_ch, out_ch)
        self.conv3 = ConvNormRelu(out_ch,          out_ch)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        f     = self.conv1(x)
        f     = torch.cat([f, h_prev], dim=1)
        f     = self.conv2(f)
        h_new = self.conv3(f)
        return h_new


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvNormRelu(in_ch, out_ch)
        self.rcnn = RecurrentBlock(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor):
        f      = self.conv(x)
        h_new  = self.rcnn(f, h_prev)
        skip   = h_new
        pooled = self.pool(h_new)
        return skip, pooled, h_new


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvNormRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvNormRelu(out_ch,          out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class RecurrentDenoisingAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 3, base: int = 32):
        super().__init__()
        b = base

        self.enc1 = EncoderStage(in_channels, b)
        self.enc2 = EncoderStage(b,           b * 2) 
        self.enc3 = EncoderStage(b * 2,       b * 4)
        self.enc4 = EncoderStage(b * 4,       b * 8)

        self.bottleneck = nn.Sequential(
            ConvNormRelu(b * 8, b * 8),
            ConvNormRelu(b * 8, b * 8),
        )

        self.dec4 = DecoderStage(b * 8, b * 8, b * 4)
        self.dec3 = DecoderStage(b * 4, b * 4, b * 2)
        self.dec2 = DecoderStage(b * 2, b * 2, b)
        self.dec1 = DecoderStage(b,     b,      b)

        self.output_conv = nn.Conv2d(b, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:  torch.Tensor,
        h1: torch.Tensor,
        h2: torch.Tensor,
        h3: torch.Tensor,
        h4: torch.Tensor,
    ):
        skip1, x, h1_out = self.enc1(x,  h1)
        skip2, x, h2_out = self.enc2(x,  h2)
        skip3, x, h3_out = self.enc3(x,  h3)
        skip4, x, h4_out = self.enc4(x,  h4)

        x = self.bottleneck(x)
        
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        output = torch.sigmoid(self.output_conv(x))

        return output, h1_out, h2_out, h3_out, h4_out