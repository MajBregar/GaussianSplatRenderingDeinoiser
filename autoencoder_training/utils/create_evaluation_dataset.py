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
    match = re.match(r"(.+?)_(noise|history)_(color|depth)_(\d+)\.png$", path.name)
    if match:
        model_name, image_name, kind, sample_id = match.groups()
        return {
            "model_name": model_name,
            "kind":       f"{image_name}_{kind}",
            "sample_id":  sample_id,
            "path":       path,
        }

    match = re.match(r"(.+?)_gt_color_(\d+)\.png$", path.name)
    if match:
        model_name, sample_id = match.groups()
        return {
            "model_name": model_name,
            "kind":       "gt_color",
            "sample_id":  sample_id,
            "path":       path,
        }

    match = re.match(r"(.+?)_confidence_(\d+)\.png$", path.name)
    if match:
        model_name, sample_id = match.groups()
        return {
            "model_name": model_name,
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


def generate_eval_dataset(
    image_folder:       str | Path,
    output_folder:      str | Path,
    require_confidence: bool = True,
    require_history:    bool = True,
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
        key = parsed["kind"]
        if sid not in samples:
            samples[sid] = {}
        samples[sid][key] = path

    required = {"noise_color", "noise_depth", "gt_color"}
    if require_confidence:
        required.add("confidence")
    if require_history:
        required.add("history_color")
        required.add("history_depth")

    valid: list[tuple[int, dict]] = []
    corrupted_count = 0

    for sid, files in samples.items():
        missing = required - files.keys()
        if missing:
            print(f"[Skip] sample {sid}: missing {sorted(missing)}")
            continue

        corrupt = [k for k in required if not _is_image_valid(files[k])]
        if corrupt:
            print(f"[Skip] sample {sid}: corrupted {sorted(corrupt)}")
            corrupted_count += 1
            continue

        valid.append((int(sid), files))

    valid.sort(key=lambda x: x[0])

    print(f"Found {len(valid)} complete frames ({corrupted_count} corrupted discarded)")

    if not valid:
        raise RuntimeError("No valid frames found — check filename format.")

    seq_dir = output_folder / "seq_000"

    (seq_dir / "input").mkdir(parents=True, exist_ok=True)
    (seq_dir / "depth").mkdir(parents=True, exist_ok=True)
    (seq_dir / "target").mkdir(parents=True, exist_ok=True)
    if require_confidence:
        (seq_dir / "confidence").mkdir(parents=True, exist_ok=True)
    if require_history:
        (seq_dir / "history_color").mkdir(parents=True, exist_ok=True)
        (seq_dir / "history_depth").mkdir(parents=True, exist_ok=True)

    for frame_idx, (sid, files) in enumerate(valid):
        frame_name = f"{frame_idx:04d}"

        shutil.copy2(files["noise_color"], seq_dir / "input"  / f"{frame_name}.png")
        shutil.copy2(files["gt_color"],    seq_dir / "target" / f"{frame_name}.png")
        save_depth_npy(files["noise_depth"], seq_dir / "depth" / f"{frame_name}.npy")

        if require_confidence and "confidence" in files:
            save_confidence_npy(files["confidence"], seq_dir / "confidence" / f"{frame_name}.npy")

        if require_history:
            if "history_color" in files:
                shutil.copy2(files["history_color"], seq_dir / "history_color" / f"{frame_name}.png")
            if "history_depth" in files:
                save_depth_npy(files["history_depth"], seq_dir / "history_depth" / f"{frame_name}.npy")

    print(f"\nEval dataset written to: {output_folder}")
    print(f"  Sequence: seq_000, {len(valid)} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert captured frames into a single evaluation sequence."
    )
    parser.add_argument("--images",        type=Path, required=True)
    parser.add_argument("--out",           type=Path, default=Path("dataset_eval"))
    parser.add_argument("--no_confidence", action="store_true")
    parser.add_argument("--no_history",    action="store_true")

    args = parser.parse_args()

    generate_eval_dataset(
        image_folder       = args.images,
        output_folder      = args.out,
        require_confidence = not args.no_confidence,
        require_history    = not args.no_history,
    )