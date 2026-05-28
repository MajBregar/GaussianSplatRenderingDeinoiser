from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from PIL import Image

from LightweightUNetDenoiser720p import LightweightUNetDenoiser720p


CHECKPOINT_PATH = Path("model_output/autoencoder_best.pt")
DATASET_PATH    = Path("../../dataset_unet")
OUTPUT_DIR      = Path("inference_vis_output")
SAMPLE_START    = 0
SAMPLE_END      = 30

IN_CHANNELS   = 5
OUT_CHANNELS  = 3
BASE_CHANNELS = 32
TARGET_SIZE   = (720, 1280)


def load_sample(dataset_dir: Path, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    name = f"{index:06d}"

    color = np.asarray(Image.open(dataset_dir / "input" / f"{name}.png").convert("RGB"), dtype=np.float32) / 255.0
    depth = np.load(dataset_dir / "depth" / f"{name}.npy").astype(np.float32)
    target = np.asarray(Image.open(dataset_dir / "target" / f"{name}.png").convert("RGB"), dtype=np.float32) / 255.0

    color_t  = torch.from_numpy(color).permute(2, 0, 1)
    depth_t  = torch.from_numpy(np.clip(depth, 0.0, 1.0)).unsqueeze(0)
    target_t = torch.from_numpy(target).permute(2, 0, 1)

    x = torch.cat([color_t, depth_t], dim=0)
    return x, target_t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    in_channels   = ckpt.get("in_channels",   IN_CHANNELS)
    out_channels  = ckpt.get("out_channels",  OUT_CHANNELS)
    base_channels = ckpt.get("base_channels", BASE_CHANNELS)

    model = LightweightUNetDenoiser720p(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")

    eval_dir = DATASET_PATH / "eval"
    samples  = sorted((eval_dir / "input").glob("*.png"))
    total    = len(samples)
    print(f"Eval dataset has {total} samples")

    if SAMPLE_START >= total:
        raise ValueError(f"SAMPLE_START {SAMPLE_START} is out of range (0–{total - 1})")

    sample_end = min(SAMPLE_END, total)
    print(f"Running inference on samples {SAMPLE_START}–{sample_end - 1}...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(SAMPLE_START, sample_end):
        x, y = load_sample(eval_dir, i)
        x = x.unsqueeze(0).to(device)

        pred = model(x)

        tensor_to_pil(x[:, :3]).save(OUTPUT_DIR / f"frame_{i:06d}_input.png")
        tensor_to_pil(pred).save(    OUTPUT_DIR / f"frame_{i:06d}_output.png")
        tensor_to_pil(y).save(       OUTPUT_DIR / f"frame_{i:06d}_target.png")
        print(f"  Saved sample {i:06d}")

    print(f"\nDone. {sample_end - SAMPLE_START} samples written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()