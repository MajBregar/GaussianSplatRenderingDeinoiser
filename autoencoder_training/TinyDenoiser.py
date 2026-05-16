import torch
import torch.nn as nn


class InvertPassthroughDenoiser(nn.Module):
    def forward(self, x):
        # x shape: [1, 4, H, W]
        # channels: R, G, B, Depth
        rgb = x[:, 0:3, :, :]

        # Invert RGB, ignore depth.
        return 1.0 - rgb


model = InvertPassthroughDenoiser()
model.eval()

dummy = torch.randn(1, 4, 720, 1280)

torch.onnx.export(
    model,
    dummy,
    "../public/models/tiny_denoiser.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    external_data=False,
)