"""Shared description of a disturbance and its severity levels."""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Level:
    label: str
    param: Any
    description: str


@dataclass(frozen=True)
class Disturbance:
    key: str
    display: str
    stage: str        # "pil": applied to the image before tensor conversion; "tensor": after normalization
    levels: tuple
    factory: Callable  # param -> transform
