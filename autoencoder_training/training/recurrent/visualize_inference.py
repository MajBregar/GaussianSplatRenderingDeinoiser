from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from PIL import Image

from RecurrentDenoisingAutoencoder import RecurrentDenoisingAutoencoder, _make_channels


CHECKPOINT_PATH = Path("model_output_recurrent_no_conf_garden_C24/autoencoder_best.pt")
DATASET_PATH = Path("../../dataset_evaluation_kitchen_history/")
OUTPUT_DIR = Path("rec_no_conf_train_garden_eval_kitchen")

SEQ_INDEX = 0
INFER_FRAMES = 10

IN_CHANNELS = 4
OUT_CHANNELS = 3
BASE = 24


def zero_hidden(batch_size: int, base: int, h: int, w: int, device: torch.device):
    C = _make_channels(base, 5)
    return (
        torch.zeros(batch_size, C[0], h,      w,      device=device),
        torch.zeros(batch_size, C[1], h >> 1, w >> 1, device=device),
        torch.zeros(batch_size, C[2], h >> 2, w >> 2, device=device),
        torch.zeros(batch_size, C[3], h >> 3, w >> 3, device=device),
        torch.zeros(batch_size, C[4], h >> 4, w >> 4, device=device),
    )


def load_frame(seq_dir: Path, frame_idx: int) -> torch.Tensor:
    frame = f"{frame_idx:04d}"

    color = np.asarray(
        Image.open(seq_dir / "input" / f"{frame}.png").convert("RGB"),
        dtype=np.float32,
    ) / 255.0

    depth = np.load(seq_dir / "depth" / f"{frame}.npy").astype(np.float32)

    color_t = torch.from_numpy(color).permute(2, 0, 1)
    depth_t = torch.from_numpy(depth).unsqueeze(0)

    return torch.cat([color_t, depth_t], dim=0)


def load_target(seq_dir: Path, frame_idx: int) -> torch.Tensor:
    frame = f"{frame_idx:04d}"

    color = np.asarray(
        Image.open(seq_dir / "target" / f"{frame}.png").convert("RGB"),
        dtype=np.float32,
    ) / 255.0

    return torch.from_numpy(color).permute(2, 0, 1)


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

    in_channels = ckpt.get("in_channels", IN_CHANNELS)
    out_channels = ckpt.get("out_channels", OUT_CHANNELS)
    base = ckpt.get("base_channels", BASE)

    print(f"Checkpoint input channels: {in_channels}")

    if in_channels != 4:
        raise ValueError(
            f"This script is for the no-confidence model with 4 input channels, "
            f"but checkpoint expects {in_channels} channels."
        )

    model = RecurrentDenoisingAutoencoder(
        in_channels=in_channels,
        out_channels=out_channels,
        base=base,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")

    seq_root = DATASET_PATH
    seq_dirs = sorted(seq_root.glob("seq_*"))
    total = len(seq_dirs)

    print(f"Eval sequence root: {seq_root}")
    print(f"Found {total} sequences")

    if total == 0:
        raise ValueError(f"No seq_* folders found in: {seq_root}")

    if SEQ_INDEX >= total:
        raise ValueError(f"SEQ_INDEX {SEQ_INDEX} is out of range (0–{total - 1})")

    seq_dir = seq_dirs[SEQ_INDEX]
    frames = sorted((seq_dir / "input").glob("*.png"))
    T = len(frames)

    if T == 0:
        raise ValueError(f"No input frames found in: {seq_dir / 'input'}")

    num_frames = min(INFER_FRAMES, T)

    print(f"Sequence: {seq_dir.name} ({T} frames)")
    print(f"Visualizing first {num_frames} frame(s)")

    sample_input = load_frame(seq_dir, 0)
    _, H, W = sample_input.shape

    pH = (H + 31) // 32 * 32
    pW = (W + 31) // 32 * 32

    h1, h2, h3, h4, h5 = zero_hidden(1, base, pH, pW, device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running inference at full resolution ({H}x{W})...")

    for t in range(num_frames):
        x = load_frame(seq_dir, t).unsqueeze(0).to(device)
        y = load_target(seq_dir, t)

        pred, h1, h2, h3, h4, h5 = model(x, h1, h2, h3, h4, h5)

        tensor_to_pil(x[:, :3]).save(OUTPUT_DIR / f"frame_{t:04d}_input.png")
        tensor_to_pil(pred).save(OUTPUT_DIR / f"frame_{t:04d}_output.png")
        tensor_to_pil(y).save(OUTPUT_DIR / f"frame_{t:04d}_target.png")

        print(f"  Saved frame {t:04d}")

    print(f"\nDone. {num_frames} frames written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()