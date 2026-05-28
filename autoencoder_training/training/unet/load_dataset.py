from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class DenoisingDataset(Dataset):
    def __init__(self, dataset_folder, target_size=(720, 1280), patch_size=None):
        self.dataset_folder = Path(dataset_folder)
        self.input_folder   = self.dataset_folder / "input"
        self.depth_folder   = self.dataset_folder / "depth"
        self.target_folder  = self.dataset_folder / "target"
        self.target_size    = target_size
        self.patch_size     = patch_size

        self.samples = sorted(self.input_folder.glob("*.png"))
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.input_folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path  = self.samples[index]
        name        = input_path.stem
        target_path = self.target_folder / f"{name}.png"
        depth_path  = self.depth_folder  / f"{name}.npy"

        input_rgb  = self._load_rgb(input_path)
        target_rgb = self._load_rgb(target_path)
        depth      = self._load_depth(depth_path)

        x = torch.cat([input_rgb, depth], dim=0)
        y = target_rgb

        if self.patch_size is not None:
            x, y = self._random_patch(x, y)

        return x, y

    def _random_patch(self, x, y):
        _, H, W = x.shape
        p = self.patch_size
        top  = torch.randint(0, H - p + 1, (1,)).item()
        left = torch.randint(0, W - p + 1, (1,)).item()
        return x[:, top:top+p, left:left+p], y[:, top:top+p, left:left+p]

    def _load_rgb(self, path):
        image = Image.open(path).convert("RGB")
        image = image.resize((self.target_size[1], self.target_size[0]))
        arr   = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(np.transpose(arr, (2, 0, 1)))

    def _load_depth(self, path):
        depth = np.load(path).astype(np.float32)
        if depth.shape != self.target_size:
            image = Image.fromarray(depth)
            image = image.resize((self.target_size[1], self.target_size[0]), resample=Image.BILINEAR)
            depth = np.asarray(image, dtype=np.float32)
        depth = np.clip(depth, 0.0, 1.0)
        return torch.from_numpy(depth[None, :, :])


def load_dataset(
    train_folder,
    eval_folder,
    batch_size=1,
    target_size=(720, 1280),
    patch_size=None,
    num_workers=4,
):
    train_dataset = DenoisingDataset(train_folder, target_size=target_size, patch_size=patch_size)
    eval_dataset  = DenoisingDataset(eval_folder,  target_size=target_size, patch_size=patch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    eval_loader  = DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, eval_loader

