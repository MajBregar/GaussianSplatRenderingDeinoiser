from pathlib import Path
import torch

from UNetDenoiser720p import UNetDenoiser720p

CHECKPOINT_PATH = Path("model_output_garden_C24/autoencoder_best.pt")
ONNX_OUTPUT_PATH = Path("../../../public/models/UNetDenoiser720p_garden_C24.onnx")

IN_CHANNELS = 4
OUT_CHANNELS = 3
BASE_CHANNELS = 24
HEIGHT = 720
WIDTH = 1280


def load_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = UNetDenoiser720p(
        in_channels=checkpoint.get("in_channels", IN_CHANNELS),
        out_channels=checkpoint.get("out_channels", OUT_CHANNELS),
        base=checkpoint.get("base_channels", BASE_CHANNELS),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ONNX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, device)

    dummy = torch.randn(
        1,
        IN_CHANNELS,
        HEIGHT,
        WIDTH,
        device=device,
        dtype=torch.float32,
    )

    torch.onnx.export(
        model,
        dummy,
        ONNX_OUTPUT_PATH.as_posix(),
        input_names    = ["input"],
        output_names   = ["output"],
        dynamic_axes   = {
            "input"  : {0: "batch", 2: "height", 3: "width"},
            "output" : {0: "batch", 2: "height", 3: "width"},
        },
        opset_version  = 18,
        external_data  = False,
    )

    print(f"Exported ONNX model to: {ONNX_OUTPUT_PATH}")