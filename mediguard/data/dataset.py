"""Finds, validates and loads an evaluation dataset.

Expected layout: one sub-folder per class, each holding that class's images.
A zip that wraps this in extra folders, contains a test/ split, or includes
junk such as __MACOSX is handled automatically.
"""
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DatasetError(Exception):
    """Raised with a message that is safe to show to the user."""


@dataclass
class EvalData:
    root: Path
    class_names: list
    samples: list        # (image path, class index)
    class_counts: dict


def _is_image(p):
    return p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS


def _images_in(folder):
    return sorted((p for p in folder.iterdir() if _is_image(p)), key=lambda p: p.name)


def _class_dirs(folder):
    dirs = [d for d in folder.iterdir()
            if d.is_dir() and not d.name.startswith(("__", ".")) and any(_is_image(p) for p in d.iterdir())]
    return sorted(dirs, key=lambda d: d.name)


def resolve_eval_root(root):
    """Walks down to the folder whose sub-folders are the classes."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"Dataset folder not found: {root}")
    for _ in range(4):
        if len(_class_dirs(root)) >= 2:
            return root
        if (root / "test").is_dir():
            root = root / "test"
            continue
        subs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(("__", "."))]
        if len(subs) == 1:
            root = subs[0]
            continue
        break
    raise DatasetError("No class folders found. Use one sub-folder per class, each containing images, and at least two classes.")


def collect_samples(root, max_images=None, seed=1234):
    """Lists the images and labels. max_images keeps a class-balanced random subset."""
    root = resolve_eval_root(root)
    dirs = _class_dirs(root)
    files = [_images_in(d) for d in dirs]
    total = sum(len(f) for f in files)
    if max_images and total > max_images:
        rng = random.Random(seed)
        files = [sorted(rng.sample(f, max(1, round(len(f) * max_images / total))), key=lambda p: p.name) for f in files]
    samples = [(p, i) for i, f in enumerate(files) for p in f]
    return EvalData(root, [d.name for d in dirs], samples,
                    {d.name: len(f) for d, f in zip(dirs, files)})


class FolderDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label


def make_loader(samples, transform, batch_size=32, num_workers=2):
    return DataLoader(FolderDataset(samples, transform), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)
