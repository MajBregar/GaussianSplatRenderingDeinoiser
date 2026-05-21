from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

class DenoisingDataset(Dataset):
    def __init__(self, dataset_folder, target_size=(720, 1280)):
        self.dataset_folder = Path(dataset_folder)
        self.input_folder = self.dataset_folder / "input"
        self.depth_folder = self.dataset_folder / "depth"
        self.target_folder = self.dataset_folder / "target"

        self.target_size = target_size

        self.samples = sorted(self.input_folder.glob("*.png"))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.input_folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path = self.samples[index]
        name = input_path.stem

        target_path = self.target_folder / f"{name}.png"
        depth_path = self.depth_folder / f"{name}.npy"

        input_rgb = self._load_rgb(input_path)
        target_rgb = self._load_rgb(target_path)
        depth = self._load_depth(depth_path)

        x = torch.cat([input_rgb, depth], dim=0)
        y = target_rgb

        return x, y

    def _load_rgb(self, path):
        image = Image.open(path).convert("RGB")
        image = image.resize((self.target_size[1], self.target_size[0]))

        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))

        return torch.from_numpy(arr)

    def _load_depth(self, path):
        depth = np.load(path).astype(np.float32)

        if depth.shape != self.target_size:
            image = Image.fromarray(depth)
            image = image.resize(
                (self.target_size[1], self.target_size[0]),
                resample=Image.BILINEAR,
            )
            depth = np.asarray(image, dtype=np.float32)

        depth = np.clip(depth, 0.0, 1.0)
        depth = depth[None, :, :]

        return torch.from_numpy(depth)


def load_dataset(
    train_folder,
    eval_folder,
    batch_size=1,
    target_size=(720, 1280),
    num_workers=4,
):
    train_dataset = DenoisingDataset(
        train_folder,
        target_size=target_size,
    )

    eval_dataset = DenoisingDataset(
        eval_folder,
        target_size=target_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, eval_loader




class DenoisingDatasetNoDepth(Dataset):
    def __init__(self, dataset_folder, target_size=(720, 1280)):
        self.dataset_folder = Path(dataset_folder)
        self.input_folder = self.dataset_folder / "input"
        self.target_folder = self.dataset_folder / "target"

        self.target_size = target_size

        self.samples = sorted(self.input_folder.glob("*.png"))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.input_folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path = self.samples[index]
        name = input_path.stem

        target_path = self.target_folder / f"{name}.png"

        x = self._load_rgb(input_path)
        y = self._load_rgb(target_path)

        return x, y

    def _load_rgb(self, path):
        image = Image.open(path).convert("RGB")
        image = image.resize((self.target_size[1], self.target_size[0]))

        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))

        return torch.from_numpy(arr)


def load_dataset_no_depth(
    train_folder,
    eval_folder,
    batch_size=1,
    target_size=(720, 1280),
    num_workers=4,
):
    train_dataset = DenoisingDatasetNoDepth(
        train_folder,
        target_size=target_size,
    )

    eval_dataset = DenoisingDatasetNoDepth(
        eval_folder,
        target_size=target_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, eval_loader



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


class SequenceDenoisingDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        seq_len: int = 7,
        target_size: Tuple[int, int] = (720, 1280),
        patch_size: Optional[int] = 128,
        augment: bool = True,
    ):
        self.root        = Path(root)
        self.seq_len     = seq_len
        self.target_size = target_size
        self.patch_size  = patch_size
        self.augment     = augment
 
        self._index: List[Tuple[Path, List[str]]] = []
 
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
            self._index.append((seq_dir, stems))
 
        if not self._index:
            raise RuntimeError(f"No valid sequences found under {self.root}")
 
        self._samples: List[Tuple[Path, List[str], int]] = []
        for seq_dir, stems in self._index:
            n = len(stems)
            for start in range(n - seq_len + 1):
                self._samples.append((seq_dir, stems, start))
 
        print(f"[Dataset] {self.root.name}: "
              f"{len(self._index)} sequences, "
              f"{len(self._samples)} samples "
              f"(seq_len={seq_len}, patch={patch_size})")
 
    def __len__(self) -> int:
        return len(self._samples)
 
    def __getitem__(self, idx: int):
        seq_dir, stems, start = self._samples[idx]
 
        frames_x = []
        frames_y = []
 
        for i in range(self.seq_len):
            stem = stems[start + i]
            rgb   = _load_rgb(seq_dir / "input"  / f"{stem}.png", self.target_size)
            depth = _load_depth(seq_dir / "depth" / f"{stem}.npy",  self.target_size)
            tgt   = _load_rgb(seq_dir / "target" / f"{stem}.png", self.target_size)
 
            frames_x.append(torch.cat([rgb, depth], dim=0))
            frames_y.append(tgt)
 
        xs = torch.stack(frames_x)
        ys = torch.stack(frames_y)
 
        if self.patch_size is not None:
            xs, ys = self._random_crop(xs, ys)
 
        if self.augment:
            xs, ys = self._augment(xs, ys)
 
        return xs, ys
 
    def _random_crop(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, H, W = xs.shape
        p = self.patch_size
        top  = torch.randint(0, H - p + 1, (1,)).item()
        left = torch.randint(0, W - p + 1, (1,)).item()
        xs = xs[:, :, top:top+p, left:left+p]
        ys = ys[:, :, top:top+p, left:left+p]
        return xs, ys
 
    def _augment(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            xs = torch.rot90(xs, k, dims=[-2, -1])
            ys = torch.rot90(ys, k, dims=[-2, -1])
        return xs, ys
 
  
def load_sequence_dataset(
    train_folder: str | Path,
    eval_folder:  str | Path,
    seq_len:      int = 7,
    batch_size:   int = 1,
    target_size:  Tuple[int, int] = (720, 1280),
    patch_size:   Optional[int] = 128,
    num_workers:  int = 4,
):
    train_ds = SequenceDenoisingDataset(
        train_folder,
        seq_len=seq_len,
        target_size=target_size,
        patch_size=patch_size,
        augment=True,
    )
    eval_ds = SequenceDenoisingDataset(
        eval_folder,
        seq_len=seq_len,
        target_size=target_size,
        patch_size=patch_size,
        augment=False,
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
 
    return train_loader, eval_loader











class DownsampledSequenceDenoisingDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        seq_len: int = 7,
        input_size: Tuple[int, int] = (720, 1280),
        target_size: Tuple[int, int] = (720, 1280),
        patch_size: Optional[int] = 128,
        augment: bool = True,
    ):
        self.root        = Path(root)
        self.seq_len     = seq_len
        self.input_size  = input_size
        self.target_size = target_size
        self.patch_size  = patch_size
        self.augment     = augment

        self._index: List[Tuple[Path, List[str]]] = []

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
            self._index.append((seq_dir, stems))

        if not self._index:
            raise RuntimeError(f"No valid sequences found under {self.root}")

        self._samples: List[Tuple[Path, List[str], int]] = []
        for seq_dir, stems in self._index:
            n = len(stems)
            for start in range(n - seq_len + 1):
                self._samples.append((seq_dir, stems, start))

        print(f"[Dataset] {self.root.name}: "
              f"{len(self._index)} sequences, "
              f"{len(self._samples)} samples "
              f"(seq_len={seq_len}, patch={patch_size}, "
              f"input={input_size}, target={target_size})")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        seq_dir, stems, start = self._samples[idx]

        frames_x = []
        frames_y = []

        for i in range(self.seq_len):
            stem  = stems[start + i]
            rgb   = _load_rgb(seq_dir / "input"  / f"{stem}.png", self.input_size)
            depth = _load_depth(seq_dir / "depth" / f"{stem}.npy", self.input_size)
            tgt   = _load_rgb(seq_dir / "target" / f"{stem}.png", self.target_size)

            frames_x.append(torch.cat([rgb, depth], dim=0))
            frames_y.append(tgt)

        xs = torch.stack(frames_x)
        ys = torch.stack(frames_y)

        if self.patch_size is not None:
            xs, ys = self._random_crop(xs, ys)

        if self.augment:
            xs, ys = self._augment(xs, ys)

        return xs, ys

    def _random_crop(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, xH, xW = xs.shape
        _, _, yH, yW = ys.shape
        p = self.patch_size

        # crop xs at input resolution
        top_x  = torch.randint(0, xH - p + 1, (1,)).item()
        left_x = torch.randint(0, xW - p + 1, (1,)).item()
        xs = xs[:, :, top_x:top_x+p, left_x:left_x+p]

        # scale crop coords to target resolution and crop ys with 2x patch
        scale_h = yH / xH
        scale_w = yW / xW
        top_y  = int(top_x  * scale_h)
        left_y = int(left_x * scale_w)
        py = int(p * scale_h)
        px = int(p * scale_w)
        ys = ys[:, :, top_y:top_y+py, left_y:left_y+px]

        return xs, ys

    def _augment(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            xs = torch.rot90(xs, k, dims=[-2, -1])
            ys = torch.rot90(ys, k, dims=[-2, -1])
        return xs, ys


def load_downsampled_sequence_dataset(
    train_folder: str | Path,
    eval_folder:  str | Path,
    seq_len:      int = 7,
    batch_size:   int = 1,
    input_size:   Tuple[int, int] = (720, 1280),
    target_size:  Tuple[int, int] = (720, 1280),
    patch_size:   Optional[int] = 128,
    num_workers:  int = 4,
):
    train_ds = DownsampledSequenceDenoisingDataset(
        train_folder,
        seq_len=seq_len,
        input_size=input_size,
        target_size=target_size,
        patch_size=patch_size,
        augment=True,
    )
    eval_ds = DownsampledSequenceDenoisingDataset(
        eval_folder,
        seq_len=seq_len,
        input_size=input_size,
        target_size=target_size,
        patch_size=patch_size,
        augment=False,
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

    return train_loader, eval_loader