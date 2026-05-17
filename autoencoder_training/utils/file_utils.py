from pathlib import Path
import shutil
import re

import numpy as np
from PIL import Image


def depth_rgb_to_f32(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32)

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    depth = (
        r / 255.0 +
        g / (255.0 * 255.0) +
        b / (255.0 * 255.0 * 255.0)
    )

    return depth.astype(np.float32)


def save_depth_npy(input_png: str | Path, output_npy: str | Path) -> None:
    depth = depth_rgb_to_f32(input_png)
    np.save(output_npy, depth)


def _parse_sample_file(path: Path):
    """
    Expected format:
        {model_name}_{image_name}_{kind}_{id}.png

    Examples:
        nike_noise_color_000000.png
        nike_noise_depth_000000.png
        nike_gt_color_000000.png
        nike_gt_depth_000000.png
    """
    match = re.match(r"(.+)_(noise|gt)_(color|depth)_(\d+)\.png$", path.name)

    if not match:
        return None

    model_name, image_name, kind, sample_id = match.groups()

    return {
        "model_name": model_name,
        "image_name": image_name,
        "kind": kind,
        "sample_id": sample_id,
        "path": path,
    }


def generate_dataset(
    image_folder: str | Path,
    dataset_train_folder: str | Path,
    dataset_eval_folder: str | Path,
    train_eval_split: float,
) -> None:
    """
    Creates PyTorch-ready folders:

        train/
            input/
            target/
            depth/

        eval/
            input/
            target/
            depth/

    Mapping:
        *_noise_color_XXXXXX.png -> input image
        *_gt_color_XXXXXX.png    -> target image
        *_noise_depth_XXXXXX.png -> depth .npy

    train_eval_split:
        0.1 -> every 10th sample goes to eval
        0.5 -> every 2nd sample goes to eval
    """

    image_folder = Path(image_folder)
    dataset_train_folder = Path(dataset_train_folder)
    dataset_eval_folder = Path(dataset_eval_folder)

    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

    if not (0.0 < train_eval_split < 1.0):
        raise ValueError("train_eval_split must be between 0 and 1, e.g. 0.1 or 0.5")

    eval_every_n = round(1.0 / train_eval_split)

    for root in [dataset_train_folder, dataset_eval_folder]:
        (root / "input").mkdir(parents=True, exist_ok=True)
        (root / "target").mkdir(parents=True, exist_ok=True)
        (root / "depth").mkdir(parents=True, exist_ok=True)

    parsed_files = []

    for path in sorted(image_folder.glob("*.png")):
        parsed = _parse_sample_file(path)

        if parsed is not None:
            parsed_files.append(parsed)

    samples = {}

    for item in parsed_files:
        key = (item["model_name"], item["sample_id"])

        if key not in samples:
            samples[key] = {}

        name = item["image_name"]
        kind = item["kind"]

        samples[key][f"{name}_{kind}"] = item["path"]

    valid_samples = []

    for key, files in samples.items():
        required = {
            "noise_color",
            "noise_depth",
            "gt_color",
        }

        missing = required - files.keys()

        if missing:
            print(f"Skipping {key}: missing {sorted(missing)}")
            continue

        valid_samples.append((key, files))

    valid_samples.sort(key=lambda x: x[0][1])

    for index, ((model_name, sample_id), files) in enumerate(valid_samples):
        is_eval = index % eval_every_n == 0
        output_root = dataset_eval_folder if is_eval else dataset_train_folder

        base_name = f"{model_name}_{sample_id}"

        input_out = output_root / "input" / f"{base_name}.png"
        target_out = output_root / "target" / f"{base_name}.png"
        depth_out = output_root / "depth" / f"{base_name}.npy"

        shutil.copy2(files["noise_color"], input_out)
        shutil.copy2(files["gt_color"], target_out)

        save_depth_npy(files["noise_depth"], depth_out)

    print(f"Generated dataset from {len(valid_samples)} valid samples.")
    print(f"Train folder: {dataset_train_folder}")
    print(f"Eval folder:  {dataset_eval_folder}")