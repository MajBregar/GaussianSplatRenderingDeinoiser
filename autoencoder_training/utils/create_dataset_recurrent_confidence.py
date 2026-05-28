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


def confidence_rgb_to_f32(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32)
    return (rgb[..., 0] / 255.0).astype(np.float32)


def save_depth_npy(input_png: str | Path, output_npy: str | Path) -> None:
    np.save(output_npy, depth_rgb_to_f32(input_png))


def save_confidence_npy(input_png: str | Path, output_npy: str | Path) -> None:
    np.save(output_npy, confidence_rgb_to_f32(input_png))


def _parse_filename(path: Path) -> dict | None:
    match = re.match(r"(.+?)_(noise|gt)_(color|depth)_(\d+)\.png$", path.name)
    if match:
        model_name, image_name, kind, sample_id = match.groups()
        return {
            "model_name": model_name,
            "image_name": image_name,
            "kind":       f"{image_name}_{kind}",
            "sample_id":  sample_id,
            "path":       path,
        }

    match = re.match(r"(.+?)_confidence_(\d+)\.png$", path.name)
    if match:
        model_name, sample_id = match.groups()
        return {
            "model_name": model_name,
            "image_name": "confidence",
            "kind":       "confidence",
            "sample_id":  sample_id,
            "path":       path,
        }

    return None


def _is_image_valid(path: Path) -> bool:
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


def _load_sequences_from_folder(
    image_folder:       Path,
    seq_stride:         int,
    min_seq_len:        int,
    require_confidence: bool,
) -> list[list[tuple[int, dict]]]:
    samples: dict[str, dict[str, Path]] = {}

    for path in sorted(image_folder.glob("*.png")):
        parsed = _parse_filename(path)
        if parsed is None:
            continue
        sid = parsed["sample_id"]
        key = parsed["kind"]
        if sid not in samples:
            samples[sid] = {}
        samples[sid][key] = path

    required = {"noise_color", "noise_depth", "gt_color"}
    if require_confidence:
        required.add("confidence")

    valid: list[tuple[int, dict]] = []
    corrupted_count = 0

    for sid, files in samples.items():
        missing = required - files.keys()
        if missing:
            print(f"  [Skip] sample {sid}: missing {sorted(missing)}")
            continue
        corrupt = [k for k in required if not _is_image_valid(files[k])]
        if corrupt:
            print(f"  [Skip] sample {sid}: corrupted {sorted(corrupt)}")
            corrupted_count += 1
            continue
        valid.append((int(sid), files))

    valid.sort(key=lambda x: x[0])
    print(f"  Found {len(valid)} complete frames ({corrupted_count} corrupted discarded)")

    if not valid:
        return []

    runs: list[list[tuple[int, dict]]] = []
    current_run: list[tuple[int, dict]] = [valid[0]]

    for prev, curr in zip(valid, valid[1:]):
        gap = curr[0] - prev[0]
        if gap <= 2:
            current_run.append(curr)
        else:
            runs.append(current_run)
            current_run = [curr]
    runs.append(current_run)

    print(f"  Formed {len(runs)} contiguous run(s)")

    sequences: list[list[tuple[int, dict]]] = []
    for run in runs:
        for start in range(0, len(run), seq_stride):
            chunk = run[start : start + seq_stride]
            if len(chunk) < min_seq_len:
                print(f"  [Skip] chunk starting at {chunk[0][0]} has only {len(chunk)} frames, discarding")
                continue
            sequences.append(chunk)

    return sequences


def generate_sequence_dataset(
    image_folders:      list[Path],
    output_folder:      Path,
    seq_stride:         int  = 50,
    eval_every:         int  = 5,
    min_seq_len:        int  = 7,
    require_confidence: bool = True,
) -> None:
    output_folder = Path(output_folder)

    all_sequences: list[list[tuple[int, dict]]] = []

    for folder in image_folders:
        if not folder.exists():
            raise FileNotFoundError(f"Image folder not found: {folder}")
        print(f"\nLoading scene: {folder}")
        seqs = _load_sequences_from_folder(folder, seq_stride, min_seq_len, require_confidence)
        print(f"  -> {len(seqs)} sequences")
        all_sequences.extend(seqs)

    print(f"\nTotal sequences across all scenes: {len(all_sequences)}")
    print(f"(stride={seq_stride}, eval_every={eval_every})")

    train_root = output_folder / "train"
    eval_root  = output_folder / "eval"

    train_seq_idx = 0
    eval_seq_idx  = 0
    train_count   = 0
    eval_count    = 0

    for seq_idx, chunk in enumerate(all_sequences):
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
        if require_confidence:
            (seq_dir / "confidence").mkdir(parents=True, exist_ok=True)

        for frame_idx, (sid, files) in enumerate(chunk):
            frame_name = f"{frame_idx:04d}"
            shutil.copy2(files["noise_color"], seq_dir / "input"  / f"{frame_name}.png")
            shutil.copy2(files["gt_color"],    seq_dir / "target" / f"{frame_name}.png")
            save_depth_npy(files["noise_depth"], seq_dir / "depth" / f"{frame_name}.npy")
            if require_confidence and "confidence" in files:
                save_confidence_npy(files["confidence"], seq_dir / "confidence" / f"{frame_name}.npy")

        if is_eval:
            eval_count += len(chunk)
        else:
            train_count += len(chunk)

    print(f"\nDataset written to: {output_folder}")
    print(f"  Train: {train_seq_idx} sequences, {train_count} frames")
    print(f"  Eval:  {eval_seq_idx}  sequences, {eval_count}  frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert captured frames from multiple scenes into sequence folders for recurrent denoiser training."
    )
    parser.add_argument("--images",      type=Path, nargs="+", required=True,
                        help="One or more image folders (one per scene)")
    parser.add_argument("--out",         type=Path, default=Path("dataset_recurrent_orbit_cam"))
    parser.add_argument("--seq_stride",  type=int,  default=30)
    parser.add_argument("--eval_every",  type=int,  default=5)
    parser.add_argument("--min_seq_len", type=int,  default=10)
    parser.add_argument("--no_confidence", action="store_true")

    args = parser.parse_args()

    generate_sequence_dataset(
        image_folders      = args.images,
        output_folder      = args.out,
        seq_stride         = args.seq_stride,
        eval_every         = args.eval_every,
        min_seq_len        = args.min_seq_len,
        require_confidence = not args.no_confidence,
    )