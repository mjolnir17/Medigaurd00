from torchvision import transforms as T

from .base import Disturbance, Level

DISTURBANCE = Disturbance(
    "blur", "Gaussian blur", "pil",
    (Level("low", (3, 1.0), "kernel 3, sigma 1.0"),
     Level("medium", (7, 2.5), "kernel 7, sigma 2.5"),
     Level("high", (15, 5.0), "kernel 15, sigma 5.0")),
    lambda p: T.GaussianBlur(kernel_size=p[0], sigma=p[1]),
)
