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
        self.conv1 = ConvNormRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvNormRelu(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class RecurrentDenoisingAutoencoderUpsampling(nn.Module):
    """
    Receives 360p input, outputs 720p.
    Network runs at 360p, final output is bilinearly upsampled 2x.
    hidden states are sized to 360p (the input resolution).
    pad to multiples of 32 at input resolution.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, base: int = 32):
        super().__init__()
        C = _make_channels(base, 5)

        self.enc1 = EncoderStage(in_channels, C[0])
        self.enc2 = EncoderStage(C[0], C[1])
        self.enc3 = EncoderStage(C[1], C[2])
        self.enc4 = EncoderStage(C[2], C[3])
        self.enc5 = EncoderStage(C[3], C[4])

        self.bottleneck = nn.Sequential(
            ConvNormRelu(C[4], C[4]),
            ConvNormRelu(C[4], C[4]),
        )

        self.dec5 = DecoderStage(C[4], C[4], C[3])
        self.dec4 = DecoderStage(C[3], C[3], C[2])
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

    def forward(self, x, h1, h2, h3, h4, h5):
        B, C, H, W = x.shape

        pH = (H + 31) // 32 * 32
        pW = (W + 31) // 32 * 32
        x = F.pad(x, (0, pW - W, 0, pH - H))

        skip1, x, h1_out = self.enc1(x, h1)
        skip2, x, h2_out = self.enc2(x, h2)
        skip3, x, h3_out = self.enc3(x, h3)
        skip4, x, h4_out = self.enc4(x, h4)
        skip5, x, h5_out = self.enc5(x, h5)

        x = self.bottleneck(x)

        x = self.dec5(x, skip5)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        output = torch.sigmoid(self.output_conv(x))
        output = output[:, :, :H*2, :W*2]

        return output, h1_out, h2_out, h3_out, h4_out, h5_out