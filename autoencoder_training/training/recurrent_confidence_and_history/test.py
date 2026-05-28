from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from RecurrentDenoisingAutoencoder import RecurrentDenoisingAutoencoder, _make_channels
from load_dataset import load_sequence_dataset

CHECKPOINT_PATH  = Path("model_output_recurrent_history/autoencoder_best.pt")
GIF_OUTPUT_DIR   = Path("eval_gifs")

IN_CHANNELS  = 9
OUT_CHANNELS = 3
BASE         = 32
SEQ_LEN      = 7
PATCH_SIZE   = None
FPS          = 8
MAX_SEQS     = 10


def load_model(checkpoint_path: Path, device: str) -> RecurrentDenoisingAutoencoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"  in_channels  = {checkpoint.get('in_channels', '?')}")
    print(f"  out_channels = {checkpoint.get('out_channels', '?')}")
    print(f"  base_channels= {checkpoint.get('base_channels', '?')}")
    print(f"  epoch        = {checkpoint.get('epoch', '?')}")
    print(f"  eval_loss    = {checkpoint.get('eval_loss', '?')}")
    model = RecurrentDenoisingAutoencoder(
        in_channels  = checkpoint.get("in_channels",  IN_CHANNELS),
        out_channels = checkpoint.get("out_channels", OUT_CHANNELS),
        base         = checkpoint.get("base_channels", BASE),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded successfully")
    return model


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def zero_hidden(batch_size: int, base: int, h: int, w: int, device):
    pH = (h + 31) // 32 * 32
    pW = (w + 31) // 32 * 32
    C = _make_channels(base, 5)
    return (
        torch.zeros(batch_size, C[0], pH,      pW,      device=device),
        torch.zeros(batch_size, C[1], pH >> 1, pW >> 1, device=device),
        torch.zeros(batch_size, C[2], pH >> 2, pW >> 2, device=device),
        torch.zeros(batch_size, C[3], pH >> 3, pW >> 3, device=device),
        torch.zeros(batch_size, C[4], pH >> 4, pW >> 4, device=device),
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    GIF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    _, eval_loader = load_sequence_dataset(
        train_folder="../../dataset_test_history/train",
        eval_folder ="../../dataset_test_history/eval",
        seq_len      = SEQ_LEN,
        batch_size   = 1,
        target_size  = (720, 1280),
        patch_size   = PATCH_SIZE,
        num_workers  = 0,
        use_history  = True,
    )

    with torch.inference_mode():
        for seq_idx, (xs, ys, cs) in enumerate(tqdm(eval_loader, desc="Sequences")):
            if seq_idx >= MAX_SEQS:
                print(f"Reached MAX_SEQS={MAX_SEQS}, stopping.")
                break

            print(f"\n=== Sequence {seq_idx} ===")
            print(f"  xs shape: {xs.shape}  dtype: {xs.dtype}")
            print(f"  ys shape: {ys.shape}  dtype: {ys.dtype}")
            print(f"  xs min={xs.min():.3f} max={xs.max():.3f} mean={xs.mean():.3f}")

            xs = xs.to(device)
            ys = ys.to(device)

            B, T, _, pH, pW = xs.shape
            h1, h2, h3, h4, h5 = zero_hidden(B, BASE, pH, pW, device)
            print(f"  hidden h1 shape: {h1.shape}")

            pred_frames  = []
            input_frames = []
            gt_frames    = []

            for t in range(T):
                pred, h1, h2, h3, h4, h5 = model(xs[:, t], h1, h2, h3, h4, h5)

                if t == 0:
                    print(f"  [t=0] input ch0(R)={xs[0,t,0].mean():.3f} ch1(G)={xs[0,t,1].mean():.3f} ch2(B)={xs[0,t,2].mean():.3f}")
                    print(f"  [t=0] input ch3(depth)={xs[0,t,3].mean():.3f} ch4(conf)={xs[0,t,4].mean():.3f}")
                    if xs.shape[2] > 5:
                        print(f"  [t=0] input ch5(hR)={xs[0,t,5].mean():.3f} ch6(hG)={xs[0,t,6].mean():.3f} ch7(hB)={xs[0,t,7].mean():.3f} ch8(hD)={xs[0,t,8].mean():.3f}")
                    print(f"  [t=0] pred shape: {pred.shape}")
                    print(f"  [t=0] pred ch0(R)={pred[0,0].mean():.3f} ch1(G)={pred[0,1].mean():.3f} ch2(B)={pred[0,2].mean():.3f}")
                    print(f"  [t=0] pred min={pred.min():.3f} max={pred.max():.3f}")
                    print(f"  [t=0] gt   ch0(R)={ys[0,t,0].mean():.3f} ch1(G)={ys[0,t,1].mean():.3f} ch2(B)={ys[0,t,2].mean():.3f}")

                pred_frames.append(tensor_to_pil(pred))
                input_frames.append(tensor_to_pil(xs[:, t, :3]))
                gt_frames.append(tensor_to_pil(ys[:, t]))

            frame_duration = int(1000 / FPS)

            def save_gif(frames, path):
                frames[0].save(
                    path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=frame_duration,
                    loop=0,
                )

            save_gif(pred_frames,  GIF_OUTPUT_DIR / f"seq_{seq_idx:03d}_pred.gif")
            save_gif(input_frames, GIF_OUTPUT_DIR / f"seq_{seq_idx:03d}_input.gif")
            save_gif(gt_frames,    GIF_OUTPUT_DIR / f"seq_{seq_idx:03d}_gt.gif")
            print(f"  Saved GIFs for seq {seq_idx}")

    print(f"\nGIFs saved to: {GIF_OUTPUT_DIR}")