import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, num_channels)


def _make_channels(base: int, n_stages: int) -> list[int]:
    return [max(base, int(base * (4/3)**i)) for i in range(n_stages)]


class ConvNormRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            _norm(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RecurrentBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvNormRelu(in_ch, out_ch)
        self.conv2 = ConvNormRelu(out_ch + out_ch, out_ch)
        self.conv3 = ConvNormRelu(out_ch, out_ch)

    def forward(self, x, h_prev):
        f = self.conv1(x)
        f = torch.cat([f, h_prev], dim=1)
        f = self.conv2(f)
        return self.conv3(f)


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvNormRelu(in_ch, out_ch)
        self.rcnn = RecurrentBlock(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, h_prev):
        f     = self.conv(x)
        h_new = self.rcnn(f, h_prev)
        return h_new, self.pool(h_new), h_new


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvNormRelu(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RecurrentDenoisingAutoencoderLW(nn.Module):
    """
    3-stage variant. base=24 gives [24, 32, 42] — enough capacity
    to learn full mapping without residual. pad to multiples of 8.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, base: int = 24):
        super().__init__()
        C = _make_channels(base, 3)

        self.enc1 = EncoderStage(in_channels, C[0])
        self.enc2 = EncoderStage(C[0], C[1])
        self.enc3 = EncoderStage(C[1], C[2])

        self.bottleneck = ConvNormRelu(C[2], C[2])

        self.dec3 = DecoderStage(C[2], C[2], C[1])
        self.dec2 = DecoderStage(C[1], C[1], C[0])
        self.dec1 = DecoderStage(C[0], C[0], C[0])

        self.output_conv = nn.Conv2d(C[0], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, h1, h2, h3):
        B, C, H, W = x.shape
        pH = (H + 7) // 8 * 8
        pW = (W + 7) // 8 * 8
        x = F.pad(x, (0, pW - W, 0, pH - H))

        skip1, x, h1_out = self.enc1(x, h1)
        skip2, x, h2_out = self.enc2(x, h2)
        skip3, x, h3_out = self.enc3(x, h3)

        x = self.bottleneck(x)

        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        output = torch.sigmoid(self.output_conv(x))
        output = output[:, :, :H, :W]

        return output, h1_out, h2_out, h3_out