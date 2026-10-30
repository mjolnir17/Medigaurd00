import torch

from .base import Disturbance, Level


class AddGaussianNoise:
    """Adds Gaussian noise to a normalized image tensor."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.sigma


DISTURBANCE = Disturbance(
    "noise", "Gaussian noise", "tensor",
    (Level("low", 0.05, "sigma 0.05"), Level("medium", 0.10, "sigma 0.1"), Level("high", 0.20, "sigma 0.2")),
    AddGaussianNoise,
)
