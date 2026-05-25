from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional


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
    depth = np.clip(depth, 0.0, 1.0)
    return torch.from_numpy(depth[None, :, :])


def _load_confidence(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    conf = np.load(path).astype(np.float32)
    if conf.shape != (size[0], size[1]):
        img = Image.fromarray(conf)
        img = img.resize((size[1], size[0]), Image.BILINEAR)
        conf = np.asarray(img, dtype=np.float32)
    conf = np.clip(conf, 0.0, 1.0)
    return torch.from_numpy(conf[None, :, :])


class SequenceDenoisingDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        seq_len: int = 7,
        target_size: Tuple[int, int] = (720, 1280),
        patch_size: Optional[int] = 128,
        augment: bool = True,
        use_confidence: bool = True,
    ):
        self.root           = Path(root)
        self.seq_len        = seq_len
        self.target_size    = target_size
        self.patch_size     = patch_size
        self.augment        = augment
        self.use_confidence = use_confidence

        self._index: List[Tuple[Path, List[str], bool]] = []

        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue
            input_dir = seq_dir / "input"
            if not input_dir.exists():
                continue
            stems = sorted(p.stem for p in input_dir.glob("*.png"))
            if len(stems) < seq_len:
                print(f"[Dataset] Skipping {seq_dir.name}: "
                      f"only {len(stems)} frames, need {seq_len}")
                continue
            has_confidence = (seq_dir / "confidence").exists() and use_confidence
            self._index.append((seq_dir, stems, has_confidence))

        if not self._index:
            raise RuntimeError(f"No valid sequences found under {self.root}")

        self._samples: List[Tuple[Path, List[str], int, bool]] = []
        for seq_dir, stems, has_confidence in self._index:
            n = len(stems)
            for start in range(n - seq_len + 1):
                self._samples.append((seq_dir, stems, start, has_confidence))

        n_with_conf = sum(1 for *_, hc in self._samples if hc)
        print(f"[Dataset] {self.root.name}: "
              f"{len(self._index)} sequences, "
              f"{len(self._samples)} samples "
              f"(seq_len={seq_len}, patch={patch_size}, "
              f"confidence={n_with_conf}/{len(self._samples)})")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        seq_dir, stems, start, has_confidence = self._samples[idx]

        frames_x = []
        frames_y = []
        frames_c = []

        for i in range(self.seq_len):
            stem  = stems[start + i]
            rgb   = _load_rgb(seq_dir / "input"  / f"{stem}.png", self.target_size)
            depth = _load_depth(seq_dir / "depth" / f"{stem}.npy", self.target_size)
            tgt   = _load_rgb(seq_dir / "target" / f"{stem}.png", self.target_size)

            if has_confidence:
                conf = _load_confidence(
                    seq_dir / "confidence" / f"{stem}.npy", self.target_size
                )
            else:
                conf = torch.ones(1, self.target_size[0], self.target_size[1])

            frames_x.append(torch.cat([rgb, depth, conf], dim=0))  # 5ch
            frames_y.append(tgt)
            frames_c.append(conf)

        xs = torch.stack(frames_x)
        ys = torch.stack(frames_y)
        cs = torch.stack(frames_c)

        if self.patch_size is not None:
            xs, ys, cs = self._random_crop(xs, ys, cs)

        if self.augment:
            xs, ys, cs = self._augment(xs, ys, cs)

        return xs, ys, cs

    def _random_crop(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        cs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, H, W = xs.shape
        p    = self.patch_size
        top  = torch.randint(0, H - p + 1, (1,)).item()
        left = torch.randint(0, W - p + 1, (1,)).item()
        xs = xs[:, :, top:top+p, left:left+p]
        ys = ys[:, :, top:top+p, left:left+p]
        cs = cs[:, :, top:top+p, left:left+p]
        return xs, ys, cs

    def _augment(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        cs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            xs = torch.rot90(xs, k, dims=[-2, -1])
            ys = torch.rot90(ys, k, dims=[-2, -1])
            cs = torch.rot90(cs, k, dims=[-2, -1])
        return xs, ys, cs


def load_sequence_dataset(
    train_folder:   str | Path,
    eval_folder:    str | Path,
    seq_len:        int = 7,
    batch_size:     int = 1,
    target_size:    Tuple[int, int] = (720, 1280),
    patch_size:     Optional[int] = 128,
    num_workers:    int = 4,
    use_confidence: bool = True,
):
    train_ds = SequenceDenoisingDataset(
        train_folder,
        seq_len=seq_len,
        target_size=target_size,
        patch_size=patch_size,
        augment=True,
        use_confidence=use_confidence,
    )
    eval_ds = SequenceDenoisingDataset(
        eval_folder,
        seq_len=seq_len,
        target_size=target_size,
        patch_size=patch_size,
        augment=False,
        use_confidence=use_confidence,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    print("Confidence based sets loaded")
    return train_loader, eval_loader