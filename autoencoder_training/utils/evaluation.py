import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(model, data_loader, criterion, device, desc="Evaluation"):
    model.eval()

    running_loss = 0.0
    num_batches = 0

    progress = tqdm(
        data_loader,
        desc=desc,
        unit="batch",
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = criterion(pred, y)

        running_loss += loss.item()
        num_batches += 1

        progress.set_postfix({
            "loss": f"{loss.item():.6f}",
            "avg": f"{running_loss / num_batches:.6f}",
        })

    return running_loss / max(num_batches, 1)