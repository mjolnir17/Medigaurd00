"""Paths and constants shared across the Mediguard backend."""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo"
STORAGE_DIR = ROOT / "storage"
UPLOADED_MODELS_DIR = STORAGE_DIR / "models"
DATASETS_DIR = STORAGE_DIR / "datasets"
RESULTS_DIR = STORAGE_DIR / "results"
SAMPLES_DIR = STORAGE_DIR / "samples"

DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED = 1234  # fixed so random disturbances (noise) are identical on every run


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
