from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm

from RecurrentDenoisingAutoencoder import RecurrentDenoisingAutoencoder, _make_channels
from load_evaluation_dataset import load_eval_dataset

CHECKPOINT_PATH = Path("model_output_recurrent_no_conf_closed_rooms_C24/autoencoder_best.pt")
EVAL_DATASET    = '../../dataset_evaluation_kitchen_history/'
CSV_OUTPUT_PATH = Path("evaluation_metrics_recurrent_no_conf_train_closed_rooms_eval_kitchen.csv")

IN_CHANNELS  = 4
OUT_CHANNELS = 3
BASE         = 24
PAD_MULTIPLE = 32
SEQ_LEN      = 7

W_SPATIAL  = 0.8
W_GRADIENT = 0.1
W_TEMPORAL = 0.1

_LOG_CACHE: dict = {}


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


def _gaussian_frame_weights(n: int) -> torch.Tensor:
    t = torch.linspace(-2.5, 0.0, n)
    w = torch.exp(-t ** 2)
    return w / w.max()


def spatial_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


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


def temporal_loss(preds: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
    if len(preds) < 2:
        return torch.tensor(0.0, device=preds[0].device)

    loss = sum(
        F.l1_loss(preds[i] - preds[i - 1], targets[i] - targets[i - 1])
        for i in range(1, len(preds))
    )

    return loss / (len(preds) - 1)


def sequence_loss(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    frame_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Exact training-style sequence loss over one sliding window.

    preds, targets:
        lists of length SEQ_LEN, each tensor [B, C, H, W]

    frame_weights:
        tensor [SEQ_LEN], usually _gaussian_frame_weights(SEQ_LEN)
    """
    L_s = sum(
        frame_weights[t] * spatial_loss(pred, tgt)
        for t, (pred, tgt) in enumerate(zip(preds, targets))
    ) / len(preds)

    L_g = sum(
        frame_weights[t] * gradient_loss(pred, tgt)
        for t, (pred, tgt) in enumerate(zip(preds, targets))
    ) / len(preds)

    L_t = temporal_loss(preds, targets)

    return W_SPATIAL * L_s + W_GRADIENT * L_g + W_TEMPORAL * L_t


def write_metrics_csv(metrics: dict, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "checkpoint_path",
        "eval_dataset",
        "checkpoint_epoch",
        "num_frames",
        "num_temporal_pairs",
        "num_seq7_windows",
        "mean_l1",
        "mean_gradient_l1",
        "mean_temporal_delta_l1",
        "mean_sliding_seq7_loss",
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

    h1, h2, h3, h4, h5 = None, None, None, None, None

    total_l1   = 0.0
    total_grad = 0.0
    total_temp = 0.0
    total_seq7 = 0.0

    frame_count = 0
    temp_count  = 0
    seq7_count  = 0

    prev_pred   = None
    prev_target = None

    recent_preds: list[torch.Tensor] = []
    recent_targets: list[torch.Tensor] = []
    frame_weights = _gaussian_frame_weights(SEQ_LEN).to(device)

    progress = tqdm(loader, desc="Evaluating", unit="frame")

    for x, target in progress:
        x      = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        x_padded, (pH, pW) = pad_to_multiple(x, PAD_MULTIPLE)
        _, _, Hp, Wp = x_padded.shape

        if h1 is None:
            print(
                f"[Eval] input size {x.shape[-2]}x{x.shape[-1]} "
                f"-> padded {Hp}x{Wp} (pad +{pH},+{pW})"
            )
            h1, h2, h3, h4, h5 = zero_hidden(BASE, Hp, Wp, device)

        pred_padded, h1, h2, h3, h4, h5 = model(
            x_padded,
            h1, h2, h3, h4, h5,
        )

        H, W = x.shape[-2], x.shape[-1]
        pred = pred_padded[..., :H, :W]

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

        total_l1   += frame_l1.item()
        total_grad += frame_grad.item()
        frame_count += 1

        pred_detached = pred.detach()
        target_detached = target.detach()

        recent_preds.append(pred_detached)
        recent_targets.append(target_detached)

        if len(recent_preds) > SEQ_LEN:
            recent_preds.pop(0)
            recent_targets.pop(0)

        if len(recent_preds) == SEQ_LEN:
            seq7_loss = sequence_loss(
                recent_preds,
                recent_targets,
                frame_weights,
            )
            total_seq7 += seq7_loss.item()
            seq7_count += 1
        else:
            seq7_loss = torch.tensor(0.0, device=device)

        prev_pred   = pred_detached
        prev_target = target_detached

        progress.set_postfix({
            "l1":       f"{frame_l1.item():.6f}",
            "grad":     f"{frame_grad.item():.6f}",
            "temp":     f"{frame_temp.item():.6f}",
            "mean_l1":  f"{total_l1 / frame_count:.6f}",
            "seq7":     f"{seq7_loss.item():.6f}",
        })

    mean_l1   = total_l1 / max(frame_count, 1)
    mean_grad = total_grad / max(frame_count, 1)
    mean_temp = total_temp / max(temp_count, 1)
    mean_seq7 = total_seq7 / max(seq7_count, 1)

    print(f"\nEvaluation complete: {frame_count} frames")
    print(f"  Mean L1:                 {mean_l1:.6f}")
    print(f"  Mean gradient L1:        {mean_grad:.6f}")
    print(f"  Mean temporal Δ L1:      {mean_temp:.6f}")
    print(f"  Mean sliding seq-7 loss: {mean_seq7:.6f}")
    print(f"  Seq-7 windows:           {seq7_count}")

    return {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "eval_dataset": str(EVAL_DATASET),
        "checkpoint_epoch": checkpoint_epoch if checkpoint_epoch is not None else "",
        "mean_l1": mean_l1,
        "mean_gradient_l1": mean_grad,
        "mean_temporal_delta_l1": mean_temp,
        "mean_sliding_seq7_loss": mean_seq7,
        "num_frames": frame_count,
        "num_temporal_pairs": temp_count,
        "num_seq7_windows": seq7_count,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    checkpoint_epoch = ckpt.get("epoch", "")
    print(f"Loaded checkpoint from epoch {checkpoint_epoch}")

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

    metrics = run_evaluation(
        model,
        eval_loader,
        device,
        checkpoint_epoch=checkpoint_epoch,
    )

    write_metrics_csv(metrics, CSV_OUTPUT_PATH)
