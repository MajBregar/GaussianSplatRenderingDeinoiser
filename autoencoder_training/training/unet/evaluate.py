from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm

from UNetDenoiser720p import UNetDenoiser720p
from load_evaluation_dataset import load_eval_dataset

CHECKPOINT_PATH = Path("model_output_garden_C24/autoencoder_best.pt")
EVAL_DATASET    = '../../dataset_evaluation_kitchen_history'
CSV_OUTPUT_PATH = Path("evaluation_metrics_unet_garden.csv")

IN_CHANNELS   = 4
OUT_CHANNELS  = 3
BASE_CHANNELS = 24
TARGET_SIZE   = (720, 1280)

W_SPATIAL  = 0.8
W_GRADIENT = 0.1
W_TEMPORAL = 0.1

_LOG_CACHE: dict = {}
def _log_kernel(device: torch.device) -> torch.Tensor:
    log = torch.tensor([
        [ 0,  0, -1,  0,  0],
        [ 0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [ 0, -1, -2, -1,  0],
        [ 0,  0, -1,  0,  0],
    ], dtype=torch.float32, device=device) / 16.0
    return log.view(1, 1, 5, 5)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _, C, _, _ = pred.shape
    device = pred.device

    if device not in _LOG_CACHE:
        _LOG_CACHE[device] = _log_kernel(device)

    kernel = _LOG_CACHE[device].expand(C, 1, 5, 5)

    pred_edges = F.conv2d(pred, kernel, padding=2, groups=C)
    tgt_edges  = F.conv2d(target, kernel, padding=2, groups=C)

    return F.l1_loss(pred_edges, tgt_edges)


def temporal_delta_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev_pred: torch.Tensor,
    prev_target: torch.Tensor,
) -> torch.Tensor:
    return F.l1_loss(
        pred - prev_pred,
        target - prev_target,
    )

def write_metrics_csv(metrics: dict, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "checkpoint_path",
        "eval_dataset",
        "checkpoint_epoch",
        "num_frames",
        "num_temporal_pairs",
        "mean_l1",
        "mean_gradient_l1",
        "mean_temporal_delta_l1",
        "mean_sequence_loss",
    ]

    row = {key: metrics.get(key, "") for key in fieldnames}

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f"  Wrote CSV: {csv_path}")


@torch.no_grad()
def run_evaluation(model, loader, device, checkpoint_epoch=None):
    model.eval()

    total_l1   = 0.0
    total_grad = 0.0
    total_temp = 0.0
    total_seq  = 0.0

    frame_count = 0
    temp_count  = 0

    prev_pred   = None
    prev_target = None

    progress = tqdm(loader, desc="Evaluating", unit="frame")

    for x, target in progress:
        x      = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        pred = model(x)

        frame_l1   = F.l1_loss(pred, target)
        frame_grad = gradient_loss(pred, target)

        if prev_pred is None:
            frame_temp = torch.tensor(0.0, device=device)
        else:
            frame_temp = temporal_delta_loss(
                pred,
                target,
                prev_pred,
                prev_target,
            )
            total_temp += frame_temp.item()
            temp_count += 1

        frame_seq = (
            W_SPATIAL  * frame_l1
            + W_GRADIENT * frame_grad
            + W_TEMPORAL * frame_temp
        )

        total_l1   += frame_l1.item()
        total_grad += frame_grad.item()
        total_seq  += frame_seq.item()
        frame_count += 1

        prev_pred   = pred.detach()
        prev_target = target.detach()

        progress.set_postfix({
            "l1":      f"{frame_l1.item():.6f}",
            "grad":    f"{frame_grad.item():.6f}",
            "temp":    f"{frame_temp.item():.6f}",
            "mean_l1": f"{total_l1 / frame_count:.6f}",
            "seq":     f"{frame_seq.item():.6f}",
        })

    mean_l1   = total_l1 / max(frame_count, 1)
    mean_grad = total_grad / max(frame_count, 1)
    mean_temp = total_temp / max(temp_count, 1)
    mean_seq  = total_seq / max(frame_count, 1)

    print(f"\nEvaluation complete: {frame_count} frames")
    print(f"  Mean L1:              {mean_l1:.6f}")
    print(f"  Mean gradient L1:     {mean_grad:.6f}")
    print(f"  Mean temporal Δ L1:   {mean_temp:.6f}")
    print(f"  Mean sequence loss:   {mean_seq:.6f}")

    return {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "eval_dataset": str(EVAL_DATASET),
        "checkpoint_epoch": checkpoint_epoch if checkpoint_epoch is not None else "",
        "mean_l1": mean_l1,
        "mean_gradient_l1": mean_grad,
        "mean_temporal_delta_l1": mean_temp,
        "mean_sequence_loss": mean_seq,
        "num_frames": frame_count,
        "num_temporal_pairs": temp_count,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    checkpoint_epoch = ckpt.get("epoch", "")
    print(f"Loaded checkpoint from epoch {checkpoint_epoch}")

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

    metrics = run_evaluation(
        model,
        eval_loader,
        device,
        checkpoint_epoch=checkpoint_epoch,
    )

    write_metrics_csv(metrics, CSV_OUTPUT_PATH)
