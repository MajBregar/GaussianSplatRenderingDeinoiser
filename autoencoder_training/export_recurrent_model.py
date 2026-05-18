from pathlib import Path

import torch

from models.RecurrentDenoisingAutoencoder import RecurrentDenoisingAutoencoder

CHECKPOINT_PATH  = Path("model_output_recurrent/autoencoder_best.pt")
ONNX_OUTPUT_PATH = Path("../public/models/RecurrentDenoisingAutoencoder.onnx")

IN_CHANNELS  = 4
OUT_CHANNELS = 3
BASE         = 32
HEIGHT       = 720
WIDTH        = 1280


def load_model(checkpoint_path: Path, device: str) -> RecurrentDenoisingAutoencoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = RecurrentDenoisingAutoencoder(
        in_channels  = checkpoint.get("in_channels",  IN_CHANNELS),
        out_channels = checkpoint.get("out_channels", OUT_CHANNELS),
        base         = checkpoint.get("base_channels", BASE),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"  train_loss = {checkpoint.get('train_loss', '?'):.6f}")
    print(f"  eval_loss  = {checkpoint.get('eval_loss',  '?'):.6f}")

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ONNX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, device)

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n:,}")

    dummy = (
        torch.randn(1, IN_CHANNELS, HEIGHT, WIDTH, device=device),
        torch.zeros(1, BASE,        HEIGHT,        WIDTH,        device=device),
        torch.zeros(1, BASE * 2,    HEIGHT // 2,   WIDTH // 2,   device=device),
        torch.zeros(1, BASE * 4,    HEIGHT // 4,   WIDTH // 4,   device=device),
        torch.zeros(1, BASE * 8,    HEIGHT // 8,   WIDTH // 8,   device=device),
    )

    with torch.no_grad():
        out, h1, h2, h3, h4 = model(*dummy)

    print(f"output: {tuple(out.shape)}")
    print(f"h1:     {tuple(h1.shape)}")
    print(f"h2:     {tuple(h2.shape)}")
    print(f"h3:     {tuple(h3.shape)}")
    print(f"h4:     {tuple(h4.shape)}")

    print(f"\nExporting to {ONNX_OUTPUT_PATH} ...")

    torch.onnx.export(
        model,
        dummy,
        ONNX_OUTPUT_PATH.as_posix(),
        input_names  = ["input", "h1_in", "h2_in", "h3_in", "h4_in"],
        output_names = ["output", "h1_out", "h2_out", "h3_out", "h4_out"],
        dynamic_axes = {
            "input"  : {0: "batch", 2: "height", 3: "width"},
            "h1_in"  : {0: "batch", 2: "height", 3: "width"},
            "h2_in"  : {0: "batch", 2: "h2_height", 3: "h2_width"},
            "h3_in"  : {0: "batch", 2: "h3_height", 3: "h3_width"},
            "h4_in"  : {0: "batch", 2: "h4_height", 3: "h4_width"},
            "output" : {0: "batch", 2: "height", 3: "width"},
            "h1_out" : {0: "batch", 2: "height", 3: "width"},
            "h2_out" : {0: "batch", 2: "h2_height", 3: "h2_width"},
            "h3_out" : {0: "batch", 2: "h3_height", 3: "h3_width"},
            "h4_out" : {0: "batch", 2: "h4_height", 3: "h4_width"},
        },
        opset_version       = 18,
        external_data       = False,
        do_constant_folding = True,
    )

    print(f"Exported ONNX model to: {ONNX_OUTPUT_PATH}")