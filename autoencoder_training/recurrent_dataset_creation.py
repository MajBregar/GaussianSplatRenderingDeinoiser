import argparse
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def depth_rgb_to_f32(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32)
    depth = (
        rgb[..., 0] / 255.0
        + rgb[..., 1] / (255.0 * 255.0)
        + rgb[..., 2] / (255.0 * 255.0 * 255.0)
    )
    return depth.astype(np.float32)


def save_depth_npy(input_png: str | Path, output_npy: str | Path) -> None:
    np.save(output_npy, depth_rgb_to_f32(input_png))

def _parse_filename(path: Path) -> dict | None:
    match = re.match(r"(.+?)_(noise|gt)_(color|depth)_(\d+)\.png$", path.name)
    if not match:
        return None
    model_name, image_name, kind, sample_id = match.groups()
    return {
        "model_name": model_name,
        "image_name": image_name,
        "kind":       kind,
        "sample_id":  sample_id,
        "path":       path,
    }


# ─── main ────────────────────────────────────────────────────────────────────

def generate_sequence_dataset(
    image_folder:  str | Path,
    output_folder: str | Path,
    seq_stride:    int   = 50,
    eval_every:    int   = 5,
    min_seq_len:   int   = 7,
) -> None:
    image_folder  = Path(image_folder)
    output_folder = Path(output_folder)

    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    samples: dict[str, dict[str, Path]] = {}

    for path in sorted(image_folder.glob("*.png")):
        parsed = _parse_filename(path)
        if parsed is None:
            continue

        sid = parsed["sample_id"]
        key = f"{parsed['image_name']}_{parsed['kind']}"

        if sid not in samples:
            samples[sid] = {}
        samples[sid][key] = path

    required = {"noise_color", "noise_depth", "gt_color"}
    valid: list[tuple[str, dict]] = []

    for sid, files in samples.items():
        missing = required - files.keys()
        if missing:
            print(f"[Skip] sample {sid}: missing {sorted(missing)}")
            continue
        valid.append((sid, files))

    valid.sort(key=lambda x: int(x[0]))

    print(f"Found {len(valid)} complete frames")

    if not valid:
        raise RuntimeError("No valid frames found — check filename format.")

    sequences: list[list[tuple[str, dict]]] = []

    for start in range(0, len(valid), seq_stride):
        chunk = valid[start : start + seq_stride]
        if len(chunk) < min_seq_len:
            print(f"[Skip] last chunk has only {len(chunk)} frames "
                  f"(min_seq_len={min_seq_len}), discarding")
            continue
        sequences.append(chunk)

    print(f"Formed {len(sequences)} sequences "
          f"(stride={seq_stride}, eval_every={eval_every})")

    train_root = output_folder / "train"
    eval_root  = output_folder / "eval"

    train_seq_idx = 0
    eval_seq_idx  = 0
    train_count   = 0
    eval_count    = 0

    for seq_idx, chunk in enumerate(sequences):
        is_eval = (seq_idx % eval_every == 0)

        if is_eval:
            seq_dir = eval_root / f"seq_{eval_seq_idx:03d}"
            eval_seq_idx += 1
        else:
            seq_dir = train_root / f"seq_{train_seq_idx:03d}"
            train_seq_idx += 1

        (seq_dir / "input").mkdir(parents=True, exist_ok=True)
        (seq_dir / "depth").mkdir(parents=True, exist_ok=True)
        (seq_dir / "target").mkdir(parents=True, exist_ok=True)

        for frame_idx, (sid, files) in enumerate(chunk):
            frame_name = f"{frame_idx:04d}"

            shutil.copy2(
                files["noise_color"],
                seq_dir / "input" / f"{frame_name}.png",
            )

            shutil.copy2(
                files["gt_color"],
                seq_dir / "target" / f"{frame_name}.png",
            )
            
            save_depth_npy(
                files["noise_depth"],
                seq_dir / "depth" / f"{frame_name}.npy",
            )

        if is_eval:
            eval_count += len(chunk)
        else:
            train_count += len(chunk)

    print(f"\nDataset written to: {output_folder}")
    print(f"  Train: {train_seq_idx} sequences, {train_count} frames")
    print(f"  Eval:  {eval_seq_idx}  sequences, {eval_count}  frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert captured frames into sequence folders for recurrent denoiser training."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Folder containing the raw captured PNG files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset_recurrent"),
        help="Root output folder (default: dataset/)",
    )
    parser.add_argument(
        "--seq_stride",
        type=int,
        default=50,
        help="Number of consecutive frames per sequence (default: 50)",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Every N-th sequence goes to eval (default: 5, so 20%% eval)",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=7,
        help="Discard sequences shorter than this (default: 7, must match seq_len in training)",
    )

    args = parser.parse_args()

    generate_sequence_dataset(
        image_folder  = args.images,
        output_folder = args.out,
        seq_stride    = args.seq_stride,
        eval_every    = args.eval_every,
        min_seq_len   = args.min_seq_len,
    )