from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from utils.dataset_loading import load_dataset, load_dataset_no_depth
from utils.evaluation import evaluate_model

#from models.SimpleAutoencoder720p_with_depth import SimpleAutoencoder720p_with_depth
from training.unet.LightweightUNetDenoiser720p import LightweightUNetDenoiser720p

TRAINING_EPOCHS = 3

MODEL_OUTPUT_DIR = Path("model_output")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CHANNELS = 4
OUT_CHANNELS = 3
BASE_CHANNELS = 32
TARGET_SIZE = (720, 1280)


def save_model(model, optimizer, save_path, epoch):
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "in_channels": IN_CHANNELS,
            "out_channels": OUT_CHANNELS,
            "base_channels": BASE_CHANNELS,
            "target_size": TARGET_SIZE,
        },
        save_path,
    )

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading Dataset")
    train_loader, eval_loader = load_dataset(
        train_folder="dataset/train",
        eval_folder="dataset/eval",
        batch_size=1,
        target_size=TARGET_SIZE,
    )
    print("Dataset Loaded")

    model = LightweightUNetDenoiser720p(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        base_channels=BASE_CHANNELS,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    best_eval_loss = float("inf")

    print("Beginning Training Loop")

    at_epoch = -0
    
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
                    "avg": f"{running_loss / (progress.n + 1):.6f}",
                })

            train_loss = running_loss / len(train_loader)

            eval_loss = evaluate_model(
                model=model,
                data_loader=eval_loader,
                criterion=criterion,
                device=device,
                desc=f"Eval {epoch + 1}/{TRAINING_EPOCHS}",
            )

            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.6f}, "
                f"eval_loss={eval_loss:.6f}"
            )

            checkpoint_path = MODEL_OUTPUT_DIR / f"autoencoder_epoch_{epoch + 1:03d}.pt"

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss

                best_path = MODEL_OUTPUT_DIR / "autoencoder_best.pt"
                save_model(model, optimizer, best_path, epoch)
                print(f"Saved best model: {best_path}")

        final_path = MODEL_OUTPUT_DIR / "autoencoder_final.pt"
        save_model(model, optimizer, final_path, TRAINING_EPOCHS)
        print(f"Saved final model: {final_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

        interrupted_path = MODEL_OUTPUT_DIR / "autoencoder_interrupted.pt"
        save_model(model, optimizer, interrupted_path, at_epoch)
        print(f"Saved interrupted checkpoint: {interrupted_path}")

    finally:
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        print("Cleanup complete.")