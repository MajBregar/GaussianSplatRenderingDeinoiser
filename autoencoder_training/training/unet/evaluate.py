from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from UNetDenoiser720p import UNetDenoiser720p
from load_evaluation_dataset import load_eval_dataset

CHECKPOINT_PATH = Path("model_output_garden_C24/autoencoder_best.pt")
EVAL_DATASET    = '../../dataset_evaluation_kitchen_history'

IN_CHANNELS   = 4
OUT_CHANNELS  = 3
BASE_CHANNELS = 24
TARGET_SIZE   = (720, 1280)


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()

    total_l1    = 0.0
    frame_count = 0

    progress = tqdm(loader, desc="Evaluating", unit="frame")

    for x, target in progress:
        x      = x.to(device)
        target = target.to(device)

        pred = model(x)

        frame_l1 = F.l1_loss(pred, target).item()
        total_l1    += frame_l1
        frame_count += 1

        progress.set_postfix({
            "frame_l1": f"{frame_l1:.6f}",
            "mean_l1":  f"{total_l1 / frame_count:.6f}",
        })

    mean_l1 = total_l1 / max(frame_count, 1)
    print(f"\nEvaluation complete: {frame_count} frames")
    print(f"  Mean L1: {mean_l1:.6f}")
    return mean_l1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    model = UNetDenoiser720p(
        in_channels  = ckpt.get("in_channels",   IN_CHANNELS),
        out_channels = ckpt.get("out_channels",  OUT_CHANNELS),
        base         = ckpt.get("base_channels", BASE_CHANNELS),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Loading eval dataset...")
    eval_loader = load_eval_dataset(
        eval_folder=EVAL_DATASET,
        target_size=TARGET_SIZE,
    )

    run_evaluation(model, eval_loader, device)