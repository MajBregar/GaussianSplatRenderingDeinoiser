from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from load_dataset import load_dataset
from UNetDenoiser720p import UNetDenoiser720p


TRAINING_EPOCHS  = 150
PATCH_SIZE       = 128
BATCH_SIZE       = 4
NUM_WORKERS      = 8

MODEL_OUTPUT_DIR = Path("model_output_garden_C24")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = '../../dataset_unet'

IN_CHANNELS   = 4
OUT_CHANNELS  = 3
BASE_CHANNELS = 24
TARGET_SIZE   = (720, 1280)


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch) -> float:
    model.eval()
    total = 0.0
    for x, y in tqdm(loader, desc=f"Eval {epoch + 1}/{TRAINING_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        total += criterion(model(x), y).item()
    return total / max(len(loader), 1)


def save_model(model, optimizer, save_path, epoch, train_loss, eval_loss):
    torch.save(
        {
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss":           train_loss,
            "eval_loss":            eval_loss,
            "in_channels":          IN_CHANNELS,
            "out_channels":         OUT_CHANNELS,
            "base_channels":        BASE_CHANNELS,
            "target_size":          TARGET_SIZE,
        },
        save_path,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading Dataset")
    train_loader, eval_loader = load_dataset(
        train_folder=f"{DATASET_PATH}/train",
        eval_folder=f"{DATASET_PATH}/eval",
        batch_size=BATCH_SIZE,
        target_size=TARGET_SIZE,
        patch_size=PATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    print("Dataset Loaded")

    model = UNetDenoiser720p(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        base=BASE_CHANNELS,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    best_eval_loss = float("inf")
    train_loss     = 0.0
    eval_loss      = 0.0

    print("Beginning Training Loop")
    at_epoch = 0

    try:
        for epoch in range(TRAINING_EPOCHS):
            at_epoch = epoch
            model.train()
            running_loss = 0.0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{TRAINING_EPOCHS}",
                unit="batch",
            )

            for x, y in progress:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                pred = model(x)
                loss = criterion(pred, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "avg":  f"{running_loss / (progress.n + 1):.6f}",
                })

            train_loss = running_loss / len(train_loader)

            eval_loss = evaluate(model, eval_loader, criterion, device, epoch)

            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.6f}, "
                f"eval_loss={eval_loss:.6f}"
            )

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_path = MODEL_OUTPUT_DIR / "autoencoder_best.pt"
                save_model(model, optimizer, best_path, epoch, train_loss, eval_loss)
                print(f"Saved best model: {best_path}")

        final_path = MODEL_OUTPUT_DIR / "autoencoder_final.pt"
        save_model(model, optimizer, final_path, TRAINING_EPOCHS - 1, train_loss, eval_loss)
        print(f"Saved final model: {final_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        interrupted_path = MODEL_OUTPUT_DIR / "autoencoder_interrupted.pt"
        save_model(model, optimizer, interrupted_path, at_epoch, train_loss, eval_loss)
        print(f"Saved interrupted checkpoint: {interrupted_path}")

    finally:
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("Cleanup complete.")