from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from RecurrentDenoisingAutoencoder import RecurrentDenoisingAutoencoder, _make_channels
from load_dataset import load_sequence_dataset


TRAINING_EPOCHS = 200
SEQ_LEN         = 7
PATCH_SIZE      = 128
BATCH_SIZE      = 1
LR              = 1e-3
LR_MIN          = 5e-6

IN_CHANNELS  = 5
OUT_CHANNELS = 3
BASE         = 24

W_SPATIAL  = 0.8
W_GRADIENT = 0.1
W_TEMPORAL = 0.1

DATASET_PATH = '../../dataset_seq_confidence_closed_rooms'
MODEL_OUTPUT_DIR = Path("model_output_recurrent_closed_rooms_C24")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = MODEL_OUTPUT_DIR / "training_log.json"


def _gaussian_frame_weights(n: int) -> torch.Tensor:
    t = torch.linspace(-2.5, 0.0, n)
    w = torch.exp(-t ** 2)
    return w / w.max()


def spatial_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


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
    B, C, H, W = pred.shape
    device = pred.device
    if device not in _LOG_CACHE:
        _LOG_CACHE[device] = _log_kernel(device)
    kernel = _LOG_CACHE[device].expand(C, 1, 5, 5)
    return F.l1_loss(
        F.conv2d(pred,   kernel, padding=2, groups=C),
        F.conv2d(target, kernel, padding=2, groups=C),
    )


def temporal_loss(preds: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
    if len(preds) < 2:
        return torch.tensor(0.0, device=preds[0].device)
    loss = sum(
        F.l1_loss(preds[i] - preds[i-1], targets[i] - targets[i-1])
        for i in range(1, len(preds))
    )
    return loss / (len(preds) - 1)


def sequence_loss(
    preds:         list[torch.Tensor],
    targets:       list[torch.Tensor],
    frame_weights: torch.Tensor,
) -> torch.Tensor:
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


def zero_hidden(batch_size: int, base: int, patch_h: int, patch_w: int, device: torch.device):
    C = _make_channels(base, 5)
    return (
        torch.zeros(batch_size, C[0], patch_h,      patch_w,      device=device),
        torch.zeros(batch_size, C[1], patch_h >> 1, patch_w >> 1, device=device),
        torch.zeros(batch_size, C[2], patch_h >> 2, patch_w >> 2, device=device),
        torch.zeros(batch_size, C[3], patch_h >> 3, patch_w >> 3, device=device),
        torch.zeros(batch_size, C[4], patch_h >> 4, patch_w >> 4, device=device),
    )


def build_scheduler(optimizer: AdamW, start_epoch: int) -> CosineAnnealingLR:
    return CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_EPOCHS,
        eta_min=LR_MIN,
        last_epoch=start_epoch - 1,
    )


def load_training_log() -> list[dict]:
    if LOG_PATH.exists():
        with open(LOG_PATH, "r") as f:
            log = json.load(f)
        print(f"Loaded training log ({len(log)} epochs)")
        return log
    return []


def append_training_log(log: list[dict], epoch: int, train_loss: float, eval_loss: float, lr: float):
    log.append({
        "epoch":      epoch + 1,
        "train_loss": round(train_loss, 8),
        "eval_loss":  round(eval_loss,  8),
        "lr":         lr,
    })
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, eval_loss, path):
    torch.save({
        "epoch":                epoch + 1,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss":           train_loss,
        "eval_loss":            eval_loss,
        "in_channels":          IN_CHANNELS,
        "out_channels":         OUT_CHANNELS,
        "base_channels":        BASE,
        "patch_size":           PATCH_SIZE,
        "seq_len":              SEQ_LEN,
        "lr":                   LR,
        "lr_min":               LR_MIN,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"]

    scheduler = build_scheduler(optimizer, start_epoch)
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    print(f"Resumed from {path} (epoch {start_epoch})")
    return start_epoch, scheduler


@torch.no_grad()
def evaluate(model, loader, frame_weights, device) -> float:
    model.eval()
    total = 0.0
    frame_weights = frame_weights.to(device)

    for xs, ys, cs in tqdm(loader, desc="  Eval", leave=False):
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        cs = cs.to(device, non_blocking=True)

        B, T, _, pH, pW = xs.shape
        h1, h2, h3, h4, h5 = zero_hidden(B, BASE, pH, pW, device)

        preds, targets = [], []
        for t in range(T):
            pred, h1, h2, h3, h4, h5 = model(xs[:, t], h1, h2, h3, h4, h5)
            preds.append(pred)
            targets.append(ys[:, t])

        total += sequence_loss(preds, targets, frame_weights).item()

    return total / max(len(loader), 1)


def train_one_epoch(model, loader, optimizer, frame_weights, device, epoch):
    model.train()
    running = 0.0
    frame_weights = frame_weights.to(device)

    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{TRAINING_EPOCHS}", unit="batch")

    for xs, ys, cs in progress:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        cs = cs.to(device, non_blocking=True)

        B, T, _, pH, pW = xs.shape
        h1, h2, h3, h4, h5 = zero_hidden(B, BASE, pH, pW, device)

        preds, targets = [], []
        for t in range(T):
            pred, h1, h2, h3, h4, h5 = model(xs[:, t], h1, h2, h3, h4, h5)
            h1, h2, h3, h4, h5 = (
                h1.detach(), h2.detach(), h3.detach(), h4.detach(), h5.detach()
            )
            preds.append(pred)
            targets.append(ys[:, t])

        loss = sequence_loss(preds, targets, frame_weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running += loss.item()
        progress.set_postfix({
            "loss": f"{loss.item():.6f}",
            "avg":  f"{running / (progress.n + 1):.6f}",
        })

    return running / max(len(loader), 1)







if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading datasets...")
    train_loader, eval_loader = load_sequence_dataset(
        train_folder=f"{DATASET_PATH}/train",
        eval_folder=f"{DATASET_PATH}/eval",
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        target_size=(720, 1280),
        patch_size=PATCH_SIZE,
        num_workers=10,
    )

    model = RecurrentDenoisingAutoencoder(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        base=BASE,
    ).to(device)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=1e-4)

    frame_weights = _gaussian_frame_weights(SEQ_LEN)
    print(f"Frame weights: {[f'{w:.3f}' for w in frame_weights.tolist()]}")

    start_epoch = 0
    resume_path = MODEL_OUTPUT_DIR / "autoencoder_latest.pt"

    if resume_path.exists():
        start_epoch, scheduler = load_checkpoint(resume_path, model, optimizer, device)
    else:
        scheduler = build_scheduler(optimizer, start_epoch)

    training_log = load_training_log()

    best_eval_loss = float("inf")
    if training_log:
        best_eval_loss = min(entry["eval_loss"] for entry in training_log)
        print(f"Best eval loss so far: {best_eval_loss:.6f}")

    at_epoch = start_epoch

    print("Beginning training...")
    try:
        for epoch in range(start_epoch, TRAINING_EPOCHS):
            at_epoch = epoch

            train_loss = train_one_epoch(
                model, train_loader, optimizer, frame_weights, device, epoch
            )

            scheduler.step()

            eval_loss = evaluate(model, eval_loader, frame_weights, device)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1:03d}/{TRAINING_EPOCHS}  "
                f"train={train_loss:.6f}  eval={eval_loss:.6f}  "
                f"lr={current_lr:.2e}"
            )

            append_training_log(training_log, epoch, train_loss, eval_loss, current_lr)

            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, eval_loss,
                MODEL_OUTPUT_DIR / "autoencoder_latest.pt",
            )

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    train_loss, eval_loss,
                    MODEL_OUTPUT_DIR / "autoencoder_best.pt",
                )
                print(f"  ✓ New best eval loss: {best_eval_loss:.6f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        save_checkpoint(
            model, optimizer, scheduler, at_epoch,
            0.0, 0.0,
            MODEL_OUTPUT_DIR / "autoencoder_interrupted.pt",
        )
        print(f"Saved checkpoint at epoch {at_epoch + 1}")

    finally:
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("Done.")