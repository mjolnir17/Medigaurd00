"""All available disturbances, and the image pipeline that applies one of them."""
from torchvision import transforms as T

from . import blur, brightness, contrast, noise

ALL = {d.key: d for d in (noise.DISTURBANCE, blur.DISTURBANCE, brightness.DISTURBANCE, contrast.DISTURBANCE)}


def build_transform(image_size, mean, std, disturbance=None, param=None):
    """Resize -> [pil disturbance] -> tensor -> normalize -> [tensor disturbance]."""
    steps = [T.Resize((image_size, image_size))]
    if disturbance and disturbance.stage == "pil":
        steps.append(disturbance.factory(param))
    steps += [T.ToTensor(), T.Normalize(mean, std)]
    if disturbance and disturbance.stage == "tensor":
        steps.append(disturbance.factory(param))
    return T.Compose(steps)
