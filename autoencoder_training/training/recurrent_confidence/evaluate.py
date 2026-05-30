from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from RecurrentDenoisingAutoencoder import RecurrentDenoisingAutoencoder, _make_channels
from load_evaluation_dataset import load_eval_dataset

CHECKPOINT_PATH = Path("model_output_recurrent_garden_C24/autoencoder_best.pt")
EVAL_DATASET    = '../../dataset_evaluation_kitchen_history'

IN_CHANNELS  = 5
OUT_CHANNELS = 3
BASE         = 24
PAD_MULTIPLE = 32


def pad_to_multiple(x: torch.Tensor, multiple: int):
    _, _, H, W = x.shape
    pH = (multiple - H % multiple) % multiple
    pW = (multiple - W % multiple) % multiple
    if pH == 0 and pW == 0:
        return x, (0, 0)
    return F.pad(x, (0, pW, 0, pH), mode='reflect'), (pH, pW)


def zero_hidden(base: int, h: int, w: int, device: torch.device):
    C = _make_channels(base, 5)
    return (
        torch.zeros(1, C[0], h,      w,      device=device),
        torch.zeros(1, C[1], h >> 1, w >> 1, device=device),
        torch.zeros(1, C[2], h >> 2, w >> 2, device=device),
        torch.zeros(1, C[3], h >> 3, w >> 3, device=device),
        torch.zeros(1, C[4], h >> 4, w >> 4, device=device),
    )


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()

    h1, h2, h3, h4, h5 = None, None, None, None, None
    total_l1    = 0.0
    frame_count = 0

    progress = tqdm(loader, desc="Evaluating", unit="frame")

    for x, target, _conf in progress:
        x      = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        x_padded, (pH, pW) = pad_to_multiple(x, PAD_MULTIPLE)
        _, _, Hp, Wp = x_padded.shape

        if h1 is None:
            print(f"[Eval] input size {x.shape[-2]}x{x.shape[-1]} -> padded {Hp}x{Wp} (pad +{pH},+{pW})")
            h1, h2, h3, h4, h5 = zero_hidden(BASE, Hp, Wp, device)

        pred_padded, h1, h2, h3, h4, h5 = model(x_padded, h1, h2, h3, h4, h5)

        H, W = x.shape[-2], x.shape[-1]
        pred = pred_padded[..., :H, :W]

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

    model = RecurrentDenoisingAutoencoder(
        in_channels  = ckpt.get("in_channels",   IN_CHANNELS),
        out_channels = ckpt.get("out_channels",  OUT_CHANNELS),
        base         = ckpt.get("base_channels", BASE),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Loading eval dataset...")
    eval_loader = load_eval_dataset(
        eval_folder=EVAL_DATASET,
        target_size=(720, 1280),
        num_workers=4,
    )

    run_evaluation(model, eval_loader, device)