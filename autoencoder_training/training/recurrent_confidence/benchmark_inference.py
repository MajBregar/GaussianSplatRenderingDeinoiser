from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import torch
import numpy as np
from PIL import Image


from RecurrentDenoisingAutoencoder import RecurrentDenoisingAutoencoder, _make_channels


CHECKPOINT_PATH = Path("../../final_models/recurrent_confidence_closed_rooms_C24/autoencoder_best.pt")
BENCHMARK_SEQUENCE_PATH = Path("../../benchmark_sequence")

IN_CHANNELS  = 5
OUT_CHANNELS = 3
BASE         = 24

WARMUP_RUNS  = 5


def zero_hidden(batch_size, base, h, w, device):
    C = _make_channels(base, 5)
    return (
        torch.zeros(batch_size, C[0], h,      w,      device=device),
        torch.zeros(batch_size, C[1], h >> 1, w >> 1, device=device),
        torch.zeros(batch_size, C[2], h >> 2, w >> 2, device=device),
        torch.zeros(batch_size, C[3], h >> 3, w >> 3, device=device),
        torch.zeros(batch_size, C[4], h >> 4, w >> 4, device=device),
    )


def load_frame(seq_dir, frame_idx):
    frame = f"{frame_idx:04d}"
    color = np.asarray(Image.open(seq_dir / "input" / f"{frame}.png").convert("RGB"), dtype=np.float32) / 255.0
    depth = np.load(seq_dir / "depth" / f"{frame}.npy")
    conf_path = seq_dir / "confidence" / f"{frame}.npy"
    conf = np.load(conf_path) if conf_path.exists() else np.zeros(depth.shape, dtype=np.float32)

    color_t = torch.from_numpy(color).permute(2, 0, 1)
    depth_t = torch.from_numpy(depth).unsqueeze(0)
    conf_t  = torch.from_numpy(conf).unsqueeze(0)
    return torch.cat([color_t, depth_t, conf_t], dim=0)


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    in_channels  = ckpt.get("in_channels",   IN_CHANNELS)
    out_channels = ckpt.get("out_channels",  OUT_CHANNELS)
    base         = ckpt.get("base_channels", BASE)

    model = RecurrentDenoisingAutoencoder(
        in_channels=in_channels,
        out_channels=out_channels,
        base=base,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")

    seq_dir = BENCHMARK_SEQUENCE_PATH
    frames  = sorted((seq_dir / "input").glob("*.png"))
    T       = len(frames)
    print(f"Sequence: {seq_dir.name} ({T} frames)")

    print("Preloading frames to GPU...")
    t0 = time.perf_counter()
    preloaded = [load_frame(seq_dir, t).unsqueeze(0).to(device) for t in range(T)]
    load_time = time.perf_counter() - t0
    print(f"  Preloaded {T} frames in {load_time:.3f}s")

    _, _, H, W = preloaded[0].shape
    pH = (H + 31) // 32 * 32
    pW = (W + 31) // 32 * 32
    print(f"  Resolution: {H}x{W} (padded to {pH}x{pW})")

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"\nWarmup ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        h1, h2, h3, h4, h5 = zero_hidden(1, base, pH, pW, device)
        for x in preloaded:
            _, h1, h2, h3, h4, h5 = model(x, h1, h2, h3, h4, h5)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("Benchmarking...")
    h1, h2, h3, h4, h5 = zero_hidden(1, base, pH, pW, device)
    frame_times = []

    for t, x in enumerate(preloaded):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _, h1, h2, h3, h4, h5 = model(x, h1, h2, h3, h4, h5)

        if device.type == "cuda":
            torch.cuda.synchronize()
        frame_times.append(time.perf_counter() - t0)

    times = np.array(frame_times)
    total = times.sum()

    print(f"\n{'='*40}")
    print(f"  Frames     : {T}")
    print(f"  Total time : {total*1000:.1f} ms")
    print(f"  Mean       : {times.mean()*1000:.2f} ms/frame")
    print(f"  Median     : {np.median(times)*1000:.2f} ms/frame")
    print(f"  Std        : {times.std()*1000:.2f} ms")
    print(f"  Min        : {times.min()*1000:.2f} ms")
    print(f"  Max        : {times.max()*1000:.2f} ms")
    print(f"  Throughput : {T/total:.1f} FPS")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()