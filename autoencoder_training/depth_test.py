from pathlib import Path
import numpy as np
from PIL import Image


def decode_depth24_png(path: str | Path) -> np.ndarray:
    """
    Decodes depth encoded with:

        bitShift = vec3(256*256, 256, 1)
        res = fract(depth * bitShift)
        res -= res.xxy * vec3(0, 1/256, 1/256)

    Returns:
        depth: float32 array, shape (H, W), values in [0, 1]
    """
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32) / 255.0

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    depth = r / (256.0 * 256.0) + g / 256.0 + b
    return depth.astype(np.float32)


def save_depth_npy(input_png: str | Path, output_npy: str | Path) -> None:
    depth = decode_depth24_png(input_png)
    np.save(output_npy, depth)


def save_depth_visualization(input_png: str | Path, output_png: str | Path) -> None:
    depth = decode_depth24_png(input_png)

    # Invert for easier viewing: near = bright, far = dark
    vis = (1.0 - depth) * 255.0
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    Image.fromarray(vis, mode="L").save(output_png)


if __name__ == "__main__":
    input_png = "samples_1_depth_000000.png"

    save_depth_npy(input_png, "depth_000000.npy")
    save_depth_visualization(input_png, "depth_000000_visual.png")