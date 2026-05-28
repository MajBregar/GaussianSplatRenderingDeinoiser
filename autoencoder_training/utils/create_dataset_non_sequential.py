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


def _is_image_valid(path: Path) -> bool:
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


def generate_flat_dataset(
    image_folders: list[Path],
    output_folder: Path,
    eval_every:    int = 5,
) -> None:
    output_folder = Path(output_folder)

    all_valid: list[tuple[int, dict, str]] = []
    corrupted_count = 0

    for folder in image_folders:
        if not folder.exists():
            raise FileNotFoundError(f"Image folder not found: {folder}")
        print(f"\nScanning: {folder}")

        samples: dict[str, dict[str, Path]] = {}
        for path in sorted(folder.glob("*.png")):
            parsed = _parse_filename(path)
            if parsed is None:
                continue
            sid = parsed["sample_id"]
            key = f"{parsed['image_name']}_{parsed['kind']}"
            if sid not in samples:
                samples[sid] = {}
            samples[sid][key] = path

        required = {"noise_color", "noise_depth", "gt_color"}
        folder_valid = []

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
            folder_valid.append((int(sid), files, str(folder)))

        folder_valid.sort(key=lambda x: x[0])
        print(f"  Found {len(folder_valid)} valid frames")
        all_valid.extend(folder_valid)

    print(f"\nTotal: {len(all_valid)} frames ({corrupted_count} corrupted discarded)")

    if not all_valid:
        raise RuntimeError("No valid frames found — check filename format.")

    train_root = output_folder / "train"
    eval_root  = output_folder / "eval"

    (train_root / "input").mkdir(parents=True, exist_ok=True)
    (train_root / "depth").mkdir(parents=True, exist_ok=True)
    (train_root / "target").mkdir(parents=True, exist_ok=True)
    (eval_root  / "input").mkdir(parents=True, exist_ok=True)
    (eval_root  / "depth").mkdir(parents=True, exist_ok=True)
    (eval_root  / "target").mkdir(parents=True, exist_ok=True)

    train_idx = 0
    eval_idx  = 0

    for global_idx, (sid, files, _) in enumerate(all_valid):
        is_eval = (global_idx % eval_every == 0)

        if is_eval:
            name = f"{eval_idx:06d}"
            dst  = eval_root
            eval_idx += 1
        else:
            name = f"{train_idx:06d}"
            dst  = train_root
            train_idx += 1

        shutil.copy2(files["noise_color"], dst / "input"  / f"{name}.png")
        shutil.copy2(files["gt_color"],    dst / "target" / f"{name}.png")
        save_depth_npy(files["noise_depth"], dst / "depth" / f"{name}.npy")

    print(f"\nDataset written to: {output_folder}")
    print(f"  Train: {train_idx} frames")
    print(f"  Eval:  {eval_idx}  frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert captured frames into a flat folder dataset for UNet training."
    )
    parser.add_argument("--images",     type=Path, nargs="+", required=True,
                        help="One or more image folders")
    parser.add_argument("--out",        type=Path, default=Path("dataset_unet"))
    parser.add_argument("--eval_every", type=int,  default=5)

    args = parser.parse_args()

    generate_flat_dataset(
        image_folders = args.images,
        output_folder = args.out,
        eval_every    = args.eval_every,
    )