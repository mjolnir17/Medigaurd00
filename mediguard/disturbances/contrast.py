from torchvision import transforms as T

from .base import Disturbance, Level

_FACTORS = [("low_flat", 0.85), ("medium_flat", 0.6), ("high_flat", 0.35),
            ("low_harsh", 1.15), ("medium_harsh", 1.4), ("high_harsh", 1.7)]

DISTURBANCE = Disturbance(
    "contrast", "Contrast", "pil",
    tuple(Level(name, f, f"factor {f}") for name, f in _FACTORS),
    lambda f: T.ColorJitter(contrast=(f, f)),
)
