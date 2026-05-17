from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from utils.dataset_loading import load_dataset, load_dataset_no_depth
from utils.evaluation import evaluate_model
from models.SimpleAutoencoder720p import SimpleAutoencoder720p


MODEL_OUTPUT_DIR = Path("model_output")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading Dataset")
    train_loader, eval_loader = load_dataset_no_depth(
        train_folder="dataset/train",
        eval_folder="dataset/eval",
        batch_size=1,
        target_size=(720, 1280),
    )
    print("Dataset Loaded")

    model = SimpleAutoencoder720p(
        in_channels=3,
        out_channels=3,
        base_channels=32,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    best_eval_loss = float("inf")

    print("Beginning Training Loop")

    try:
        for epoch in range(10):
            model.train()
            running_loss = 0.0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/10",
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
                desc=f"Eval {epoch + 1}/10",
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

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        "in_channels": 3,
                        "out_channels": 3,
                        "base_channels": 32,
                        "target_size": (720, 1280),
                    },
                    best_path,
                )

                print(f"Saved best model: {best_path}")

        final_path = MODEL_OUTPUT_DIR / "autoencoder_final.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_eval_loss": best_eval_loss,
                "in_channels": 3,
                "out_channels": 3,
                "base_channels": 32,
                "target_size": (720, 1280),
            },
            final_path,
        )

        print(f"Saved final model: {final_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

        interrupted_path = MODEL_OUTPUT_DIR / "autoencoder_interrupted.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_eval_loss": best_eval_loss,
                "in_channels": 4,
                "out_channels": 3,
                "base_channels": 32,
                "target_size": (720, 1280),
            },
            interrupted_path,
        )

        print(f"Saved interrupted checkpoint: {interrupted_path}")

    finally:
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        print("Cleanup complete.")