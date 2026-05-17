import torch
import torch.nn as nn


class DummySingeFrameAutoencoder(nn.Module):
    def forward(self, x):
        rgb = x[:, 0:3, :, :]
        return 1.0 - rgb 


if __name__ == "__main__":
    model = DummySingeFrameAutoencoder()
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