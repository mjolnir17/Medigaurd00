from torchvision import transforms as T

from .base import Disturbance, Level

_FACTORS = [("low_dark", 0.85), ("medium_dark", 0.65), ("high_dark", 0.4),
            ("low_bright", 1.15), ("medium_bright", 1.35), ("high_bright", 1.6)]

DISTURBANCE = Disturbance(
    "brightness", "Brightness", "pil",
    tuple(Level(name, f, f"factor {f}") for name, f in _FACTORS),
    lambda f: T.ColorJitter(brightness=(f, f)),
)
