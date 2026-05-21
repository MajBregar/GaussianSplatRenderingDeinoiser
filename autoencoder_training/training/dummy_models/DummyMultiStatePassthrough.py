import torch
import torch.nn as nn
from pathlib import Path


ONNX_OUTPUT_PATH = Path("state_passthrough.onnx")


class DummyMultiStatePassthrough(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_channels=32):
        super().__init__()

        if in_channels == out_channels:
            self.output = nn.Identity()
        else:
            self.output = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, h1, h2, h3, h4):
        y = self.output(x)
        y = torch.sigmoid(y)
        y = 1.0 - y

        z = x[:, :1, :1, :1].sum() * 0.0

        h1_out = h1 + z
        h2_out = h2 + z
        h3_out = h3 + z
        h4_out = h4 + z

        return y, h1_out, h2_out, h3_out, h4_out


if __name__ == "__main__":
    model = DummyMultiStatePassthrough(
        in_channels=4,
        out_channels=3,
        base_channels=32,
    ).eval()

    B, H, W = 1, 720, 1280
    base = 32

    dummy = (
        torch.randn(B, 4, H, W),
        torch.zeros(B, base,     H,      W),
        torch.zeros(B, base * 2, H // 2, W // 2),
        torch.zeros(B, base * 4, H // 4, W // 4),
        torch.zeros(B, base * 8, H // 8, W // 8),
    )

    torch.onnx.export(
        model,
        dummy,
        ONNX_OUTPUT_PATH.as_posix(),
        input_names=[
            "input",
            "h1_in",
            "h2_in",
            "h3_in",
            "h4_in",
        ],
        output_names=[
            "output",
            "h1_out",
            "h2_out",
            "h3_out",
            "h4_out",
        ],
        opset_version=18,
        external_data=False,
        do_constant_folding=False,
    )