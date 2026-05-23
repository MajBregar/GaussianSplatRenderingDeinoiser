import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, num_channels)


def _make_channels(base: int, n_stages: int) -> list[int]:
    return [max(base, int(base * (4/3)**i)) for i in range(n_stages)]


def _icnr_init(tensor: torch.Tensor, scale: int = 2) -> None:
    """
    ICNR initialization for pixel shuffle convolutions.
    Initializes weights so pixel shuffle starts behaving like nearest-neighbor
    upsampling, then learns from there — eliminates checkerboard artifacts.
    """
    out_ch, in_ch, h, w = tensor.shape
    sub = torch.zeros(out_ch // (scale ** 2), in_ch, h, w)
    nn.init.kaiming_normal_(sub, a=0.1, nonlinearity='leaky_relu')
    sub = sub.repeat_interleave(scale ** 2, dim=0)
    with torch.no_grad():
        tensor.copy_(sub)


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            _norm(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class RecurrentBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch)
        self.conv2 = ConvNormAct(out_ch + out_ch, out_ch)
        self.conv3 = ConvNormAct(out_ch, out_ch)

    def forward(self, x, h_prev):
        f = self.conv1(x)
        f = torch.cat([f, h_prev], dim=1)
        f = self.conv2(f)
        return self.conv3(f)


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvNormAct(in_ch, out_ch)
        self.rcnn = RecurrentBlock(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, h_prev):
        f     = self.conv(x)
        h_new = self.rcnn(f, h_prev)
        return h_new, self.pool(h_new), h_new


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch + skip_ch, out_ch)
        self.conv2 = ConvNormAct(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, growth, 3, padding=1, bias=False)
        self.act  = nn.GELU()

    def forward(self, x):
        return torch.cat([x, self.act(self.conv(x))], dim=1)


class MiniRRDB(nn.Module):
    """
    Tiny 3-layer dense block before upsample.
    Gives the network dedicated capacity for detail recovery
    before the resolution jump — adds ~150K params at 360p.
    """
    def __init__(self, ch: int, growth: int = 16):
        super().__init__()
        self.d1   = DenseLayer(ch,           growth)
        self.d2   = DenseLayer(ch + growth,  growth)
        self.d3   = DenseLayer(ch + growth*2, growth)
        self.fuse = nn.Conv2d(ch + growth*3, ch, 1, bias=False)

    def forward(self, x):
        return x + self.fuse(self.d3(self.d2(self.d1(x)))) * 0.2


class PixelShuffleUpsample(nn.Module):
    """
    Pixel shuffle with ICNR initialization + refinement conv.
    ICNR eliminates checkerboard artifacts by starting from a
    nearest-neighbor baseline. Refinement conv cleans up shuffle artifacts.
    """
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.conv          = nn.Conv2d(in_ch, out_ch * scale * scale, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.refine        = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        _icnr_init(self.conv.weight, scale)

    def forward(self, x):
        return self.refine(self.pixel_shuffle(self.conv(x)))


class RecurrentDenoisingAutoencoderUpsampling(nn.Module):
    """
    360p input → 720p output.
    Denoises and temporally accumulates at 360p, then upsamples via:
      - 2x MiniRRDB blocks for detail preparation
      - ICNR pixel shuffle for checkerboard-free upsampling
      - refinement conv to clean shuffle artifacts
    Pad to multiples of 32. Hidden states at 360p resolution.
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
            ConvNormAct(C[4], C[4]),
            ConvNormAct(C[4], C[4]),
        )

        self.dec5 = DecoderStage(C[4], C[4], C[3])
        self.dec4 = DecoderStage(C[3], C[3], C[2])
        self.dec3 = DecoderStage(C[2], C[2], C[1])
        self.dec2 = DecoderStage(C[1], C[1], C[0])
        self.dec1 = DecoderStage(C[0], C[0], C[0])

        self.pre_upsample = nn.Sequential(
            MiniRRDB(C[0]),
            MiniRRDB(C[0]),
        )
        self.upsample = PixelShuffleUpsample(C[0], out_channels, scale=2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not self.upsample.conv.weight:
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

        x = self.pre_upsample(x)
        output = (torch.tanh(self.upsample(x)) + 1) / 2
        output = output[:, :, :H*2, :W*2]

        return output, h1_out, h2_out, h3_out, h4_out, h5_out