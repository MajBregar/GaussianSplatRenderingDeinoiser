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
EVAL_DATASET    = "../../dataset_evaluation_garden_history"
CSV_OUTPUT_PATH = Path("evaluation_metrics_unet_train_garden_eval_garden.csv")

IN_CHANNELS   = 4
OUT_CHANNELS  = 3
BASE_CHANNELS = 24
TARGET_SIZE   = (720, 1280)


_SSIM_KERNEL_CACHE: dict[tuple[torch.device, int, int], torch.Tensor] = {}


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(pred, target))


def _gaussian_kernel_1d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    coords -= kernel_size // 2

    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    return kernel


def _ssim_kernel(
    channels: int,
    kernel_size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    cache_key = (device, channels, kernel_size)

    if cache_key in _SSIM_KERNEL_CACHE:
        return _SSIM_KERNEL_CACHE[cache_key]

    kernel_1d = _gaussian_kernel_1d(kernel_size, sigma, device)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()

    _SSIM_KERNEL_CACHE[cache_key] = kernel
    return kernel


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    _, channels, _, _ = pred.shape
    device = pred.device

    kernel = _ssim_kernel(
        channels=channels,
        kernel_size=kernel_size,
        sigma=sigma,
        device=device,
    )

    padding = kernel_size // 2

    mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_tgt  = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu_pred_sq  = mu_pred ** 2
    mu_tgt_sq   = mu_tgt ** 2
    mu_pred_tgt = mu_pred * mu_tgt

    sigma_pred_sq = (
        F.conv2d(pred * pred, kernel, padding=padding, groups=channels)
        - mu_pred_sq
    )

    sigma_tgt_sq = (
        F.conv2d(target * target, kernel, padding=padding, groups=channels)
        - mu_tgt_sq
    )

    sigma_pred_tgt = (
        F.conv2d(pred * target, kernel, padding=padding, groups=channels)
        - mu_pred_tgt
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = (
        (2 * mu_pred_tgt + c1) * (2 * sigma_pred_tgt + c2)
    ) / (
        (mu_pred_sq + mu_tgt_sq + c1)
        * (sigma_pred_sq + sigma_tgt_sq + c2)
    )

    return ssim_map.mean()


def write_metrics_csv(metrics: dict, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "checkpoint_path",
        "eval_dataset",
        "checkpoint_epoch",
        "num_frames",
        "mean_rmse",
        "mean_ssim",
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

    total_rmse = 0.0
    total_ssim = 0.0
    frame_count = 0

    progress = tqdm(loader, desc="Evaluating", unit="frame")

    for x, target in progress:
        x      = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        pred = model(x)

        frame_rmse = rmse(pred, target)
        frame_ssim = ssim(pred, target, data_range=1.0)

        total_rmse += frame_rmse.item()
        total_ssim += frame_ssim.item()
        frame_count += 1

        progress.set_postfix({
            "rmse":      f"{frame_rmse.item():.6f}",
            "ssim":      f"{frame_ssim.item():.6f}",
            "mean_rmse": f"{total_rmse / frame_count:.6f}",
            "mean_ssim": f"{total_ssim / frame_count:.6f}",
        })

    mean_rmse = total_rmse / max(frame_count, 1)
    mean_ssim = total_ssim / max(frame_count, 1)

    print(f"\nEvaluation complete: {frame_count} frames")
    print(f"  Mean RMSE: {mean_rmse:.6f}")
    print(f"  Mean SSIM: {mean_ssim:.6f}")

    return {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "eval_dataset": str(EVAL_DATASET),
        "checkpoint_epoch": checkpoint_epoch if checkpoint_epoch is not None else "",
        "num_frames": frame_count,
        "mean_rmse": mean_rmse,
        "mean_ssim": mean_ssim,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    checkpoint_epoch = ckpt.get("epoch", "")
    print(f"Loaded checkpoint from epoch {checkpoint_epoch}")

    model = UNetDenoiser720p(
        in_channels=ckpt.get("in_channels", IN_CHANNELS),
        out_channels=ckpt.get("out_channels", OUT_CHANNELS),
        base=ckpt.get("base_channels", BASE_CHANNELS),
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