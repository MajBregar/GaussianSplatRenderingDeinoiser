from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.RecurrentDenoisingAutoencoder import RecurrentDenoisingAutoencoder
from utils.dataset_loading import load_sequence_dataset


TRAINING_EPOCHS = 50
SEQ_LEN         = 7          # frames per training sequence (paper uses 7)
PATCH_SIZE      = 128        # spatial crop size (paper uses 128×128)
BATCH_SIZE      = 1          # keep at 1 for 720p patches on a single GPU
LR              = 1e-3       # initial learning rate (paper: Adam 0.001)
LR_WARMUP       = 10         # epochs for geometric warmup (paper §5.3)
GAMMA_COMPRESS  = 0.2        # loss gamma exponent (paper: 0.2)

IN_CHANNELS  = 4
OUT_CHANNELS = 3
BASE         = 32

# loss weights from paper
W_SPATIAL  = 0.8
W_GRADIENT = 0.1
W_TEMPORAL = 0.1

MODEL_OUTPUT_DIR = Path("model_output_recurrent")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _gaussian_frame_weights(n: int) -> torch.Tensor:
    t = torch.linspace(-2.5, 0.0, n)
    w = torch.exp(-t ** 2)
    w = w / w.max()
    return w

def gamma_compress(x: torch.Tensor, gamma: float = 0.2) -> torch.Tensor:
    return x.clamp(min=0.0).pow(gamma)


def spatial_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 0.2) -> torch.Tensor:
    return F.l1_loss(gamma_compress(pred, gamma), gamma_compress(target, gamma))


def _log_kernel(device: torch.device) -> torch.Tensor:
    log = torch.tensor([
        [ 0,  0, -1,  0,  0],
        [ 0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [ 0, -1, -2, -1,  0],
        [ 0,  0, -1,  0,  0],
    ], dtype=torch.float32, device=device) / 16.0
    return log.view(1, 1, 5, 5)


_LOG_CACHE: dict = {}


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    B, C, H, W = pred.shape
    device = pred.device

    if device not in _LOG_CACHE:
        _LOG_CACHE[device] = _log_kernel(device)
    kernel = _LOG_CACHE[device]
    kernel = kernel.expand(C, 1, 5, 5)

    pred_g   = F.conv2d(pred,   kernel, padding=2, groups=C)
    target_g = F.conv2d(target, kernel, padding=2, groups=C)

    return F.l1_loss(pred_g, target_g)


def temporal_loss(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> torch.Tensor:
    if len(preds) < 2:
        return torch.tensor(0.0, device=preds[0].device)

    loss = torch.tensor(0.0, device=preds[0].device)
    n = 0
    for i in range(1, len(preds)):
        dpred   = preds[i]   - preds[i-1]
        dtarget = targets[i] - targets[i-1]
        loss = loss + F.l1_loss(dpred, dtarget)
        n += 1
    return loss / n


def sequence_loss(
    preds:   list[torch.Tensor],
    targets: list[torch.Tensor],
    frame_weights: torch.Tensor,
    gamma: float = GAMMA_COMPRESS,
    w_s: float   = W_SPATIAL,
    w_g: float   = W_GRADIENT,
    w_t: float   = W_TEMPORAL,
) -> torch.Tensor:
    L_s = torch.tensor(0.0, device=preds[0].device)
    L_g = torch.tensor(0.0, device=preds[0].device)

    for t, (pred, tgt) in enumerate(zip(preds, targets)):
        w = frame_weights[t]
        L_s = L_s + w * spatial_loss(pred, tgt, gamma)
        L_g = L_g + w * gradient_loss(pred, tgt)

    L_s = L_s / len(preds)
    L_g = L_g / len(preds)
    L_t = temporal_loss(preds, targets)

    return w_s * L_s + w_g * L_g + w_t * L_t


def zero_hidden(batch_size: int, base: int, patch: int, device: torch.device):
    h = patch
    w = patch
    return (
        torch.zeros(batch_size, base,     h,      w,      device=device),
        torch.zeros(batch_size, base * 2, h // 2, w // 2, device=device),
        torch.zeros(batch_size, base * 4, h // 4, w // 4, device=device),
        torch.zeros(batch_size, base * 8, h // 8, w // 8, device=device),
    )


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
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"]
    print(f"Resumed from {path} (epoch {start_epoch})")
    return start_epoch


@torch.no_grad()
def evaluate(model, loader, frame_weights, device) -> float:
    model.eval()
    total = 0.0
    frame_weights = frame_weights.to(device)

    for xs, ys in tqdm(loader, desc="  Eval", leave=False):
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)

        B, T, _, pH, pW = xs.shape
        h1, h2, h3, h4 = zero_hidden(B, BASE, pH, device)

        preds   = []
        targets = []

        for t in range(T):
            x_t = xs[:, t]
            y_t = ys[:, t]
            pred, h1, h2, h3, h4 = model(x_t, h1, h2, h3, h4)
            preds.append(pred)
            targets.append(y_t)

        loss = sequence_loss(preds, targets, frame_weights)
        total += loss.item()

    return total / max(len(loader), 1)



def train_one_epoch(model, loader, optimizer, frame_weights, device, epoch, total_epochs):
    model.train()
    running = 0.0
    frame_weights = frame_weights.to(device)

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{total_epochs}",
        unit="batch",
    )

    for xs, ys in progress:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)

        B, T, _, pH, pW = xs.shape

        h1, h2, h3, h4 = zero_hidden(B, BASE, pH, device)

        preds   = []
        targets = []

        for t in range(T):
            x_t = xs[:, t]
            y_t = ys[:, t]
            pred, h1, h2, h3, h4 = model(x_t, h1, h2, h3, h4)
            preds.append(pred)
            targets.append(y_t)

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
        train_folder="dataset_recurrent/train",
        eval_folder="dataset_recurrent/eval",
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        target_size=(720, 1280),
        patch_size=PATCH_SIZE,
        num_workers=4,
    )

    model = RecurrentDenoisingAutoencoder(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        base=BASE,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=1e-4)
    warmup_factor = 10.0 ** (1.0 / LR_WARMUP)

    def warmup_lambda(epoch):
        if epoch < LR_WARMUP:
            return warmup_factor ** (epoch - LR_WARMUP)
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=TRAINING_EPOCHS - LR_WARMUP, eta_min=1e-6)

    frame_weights = _gaussian_frame_weights(SEQ_LEN)
    print(f"Frame weights: {frame_weights.tolist()}")

    start_epoch = 0
    resume_path = MODEL_OUTPUT_DIR / "autoencoder_latest.pt"
    if resume_path.exists():
        start_epoch = load_checkpoint(resume_path, model, optimizer, warmup_scheduler, device)

    best_eval_loss = float("inf")
    at_epoch = start_epoch

    print("Beginning training...")

    try:
        for epoch in range(start_epoch, TRAINING_EPOCHS):
            at_epoch = epoch

            train_loss = train_one_epoch(
                model, train_loader, optimizer, frame_weights, device, epoch, TRAINING_EPOCHS
            )

            if epoch < LR_WARMUP:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            eval_loss = evaluate(model, eval_loader, frame_weights, device)

            print(
                f"Epoch {epoch + 1:03d}/{TRAINING_EPOCHS}  "
                f"train={train_loss:.6f}  "
                f"eval={eval_loss:.6f}  "
                f"lr={current_lr:.2e}"
            )

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(
                    model, optimizer, warmup_scheduler, epoch,
                    train_loss, eval_loss,
                    MODEL_OUTPUT_DIR / "autoencoder_best.pt",
                )
                print(f"  ✓ New best eval loss: {best_eval_loss:.6f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        save_checkpoint(
            model, optimizer, warmup_scheduler, at_epoch,
            0.0, 0.0,
            MODEL_OUTPUT_DIR / "autoencoder_interrupted.pt",
        )
        print(f"Saved checkpoint at epoch {at_epoch + 1}")

    finally:
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("Done.")