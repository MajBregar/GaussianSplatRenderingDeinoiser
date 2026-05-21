from pathlib import Path

import torch

from training.recurrent_lw.RecurrentDenoisingAutoencoderLW import RecurrentDenoisingAutoencoderLW, _make_channels

CHECKPOINT_PATH  = Path("model_output_recurrent_lw/autoencoder_best.pt")
ONNX_OUTPUT_PATH = Path("../public/models/RecurrentDenoisingAutoencoderLW.onnx")

IN_CHANNELS  = 4
OUT_CHANNELS = 3
BASE         = 24
HEIGHT       = 720
WIDTH        = 1280


def load_model(checkpoint_path: Path, device: str) -> RecurrentDenoisingAutoencoderLW:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RecurrentDenoisingAutoencoderLW(
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


def infer_hidden_shapes(model, in_channels, height, width, device):
    C = _make_channels(BASE, 3)
    pH = (height + 7) // 8 * 8
    pW = (width  + 7) // 8 * 8

    h_init = [
        torch.zeros(1, C[0], pH,      pW,      device=device),
        torch.zeros(1, C[1], pH >> 1, pW >> 1, device=device),
        torch.zeros(1, C[2], pH >> 2, pW >> 2, device=device),
    ]

    with torch.no_grad():
        x = torch.zeros(1, in_channels, height, width, device=device)
        _, h1, h2, h3 = model(x, *h_init)

    return [h.shape for h in (h1, h2, h3)]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ONNX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    hidden_shapes = infer_hidden_shapes(model, IN_CHANNELS, HEIGHT, WIDTH, device)
    for i, s in enumerate(hidden_shapes):
        print(f"h{i+1} shape: {tuple(s)}")

    dummy = (
        torch.randn(1, IN_CHANNELS, HEIGHT, WIDTH, device=device),
        *[torch.zeros(*s, device=device) for s in hidden_shapes],
    )

    with torch.no_grad():
        out, h1, h2, h3 = model(*dummy)

    print(f"output: {tuple(out.shape)}")
    print(f"h1:     {tuple(h1.shape)}")
    print(f"h2:     {tuple(h2.shape)}")
    print(f"h3:     {tuple(h3.shape)}")

    print(f"\nExporting to {ONNX_OUTPUT_PATH} ...")

    torch.onnx.export(
        model,
        dummy,
        ONNX_OUTPUT_PATH.as_posix(),
        input_names  = ["input", "h1_in", "h2_in", "h3_in"],
        output_names = ["output", "h1_out", "h2_out", "h3_out"],
        dynamic_axes = {
            "input"  : {0: "batch", 2: "height",    3: "width"},
            "h1_in"  : {0: "batch", 2: "h1_height", 3: "h1_width"},
            "h2_in"  : {0: "batch", 2: "h2_height", 3: "h2_width"},
            "h3_in"  : {0: "batch", 2: "h3_height", 3: "h3_width"},
            "output" : {0: "batch", 2: "height",    3: "width"},
            "h1_out" : {0: "batch", 2: "h1_height", 3: "h1_width"},
            "h2_out" : {0: "batch", 2: "h2_height", 3: "h2_width"},
            "h3_out" : {0: "batch", 2: "h3_height", 3: "h3_width"},
        },
        opset_version       = 18,
        external_data       = False,
        do_constant_folding = True,
        dynamo              = False,
    )

    print(f"Exported ONNX model to: {ONNX_OUTPUT_PATH}")