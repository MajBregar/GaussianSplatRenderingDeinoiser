from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


def _load_rgb(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size[1], size[0]):
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _load_depth(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    depth = np.load(path).astype(np.float32)
    if depth.shape != (size[0], size[1]):
        img = Image.fromarray(depth)
        img = img.resize((size[1], size[0]), Image.BILINEAR)
        depth = np.asarray(img, dtype=np.float32)
    return torch.from_numpy(np.clip(depth, 0.0, 1.0)[None, :, :])

class EvalDenoisingDataset(Dataset):
    def __init__(
        self,
        root:        str | Path,
        target_size: Tuple[int, int] = (720, 1280),
    ):
        self.root        = Path(root)
        self.target_size = target_size

        seq_dirs = sorted(d for d in self.root.iterdir() if d.is_dir() and (d / "input").exists())
        if not seq_dirs:
            raise RuntimeError(f"No valid sequences found under {self.root}")

        self._samples: List[Tuple[Path, str]] = []
        for seq_dir in seq_dirs:
            stems = sorted(p.stem for p in (seq_dir / "input").glob("*.png"))
            for stem in stems:
                self._samples.append((seq_dir, stem))

        print(f"[EvalDataset] {self.root.name}: "
              f"{len(seq_dirs)} sequence(s), {len(self._samples)} frames total")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        seq_dir, stem = self._samples[idx]

        rgb   = _load_rgb(seq_dir / "input"  / f"{stem}.png", self.target_size)
        depth = _load_depth(seq_dir / "depth" / f"{stem}.npy", self.target_size)
        tgt   = _load_rgb(seq_dir / "target" / f"{stem}.png", self.target_size)

        x = torch.cat([rgb, depth], dim=0)
        return x, tgt

def load_eval_dataset(
    eval_folder: str | Path,
    target_size: Tuple[int, int] = (720, 1280),
) -> DataLoader:
    ds = EvalDenoisingDataset(eval_folder, target_size=target_size)
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )