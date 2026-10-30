"""Loads a user's model from disk and checks it is usable.

Two formats are accepted:
  torchscript : a file saved with torch.jit.save. Works for any architecture.
  weights     : a state_dict saved with torch.save(model.state_dict(), ...),
                loaded into an architecture chosen from the registry.
Weights are read with weights_only=True, so no arbitrary code is executed.
"""
import torch

from .architectures import ARCHITECTURES


class ModelLoadError(Exception):
    """Raised with a message that is safe to show to the user."""


def _unwrap(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if isinstance(state.get(key), dict):
                state = state[key]
    if not isinstance(state, dict):
        raise ModelLoadError("This file does not contain a weights dictionary (state_dict).")
    return {k.removeprefix("module."): v for k, v in state.items()}


def load_model(path, kind, architecture=None, num_classes=None, device=None):
    device = device or torch.device("cpu")

    if kind == "torchscript":
        try:
            model = torch.jit.load(str(path), map_location=device)
        except Exception as e:
            raise ModelLoadError(f"Could not read the TorchScript model: {e}") from e

    elif kind == "weights":
        if architecture not in ARCHITECTURES:
            raise ModelLoadError(f"Unknown architecture '{architecture}'. Choose one of: {list(ARCHITECTURES)}")
        if not num_classes:
            raise ModelLoadError("The number of classes is required for a weights file.")
        model = ARCHITECTURES[architecture]["build"](num_classes)
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as e:
            raise ModelLoadError(f"Could not read the weights file: {e}") from e
        try:
            model.load_state_dict(_unwrap(state))
        except RuntimeError as e:
            raise ModelLoadError(
                f"These weights do not fit '{architecture}' with {num_classes} classes. "
                f"Check the architecture and the class count."
            ) from e
    else:
        raise ModelLoadError(f"Unknown model format '{kind}'.")

    return model.to(device).eval()


def validate_model(model, num_classes, image_size, device):
    """Runs one blank image through the model and checks the output size."""
    try:
        with torch.no_grad():
            out = model(torch.zeros(1, 3, image_size, image_size, device=device))
    except Exception as e:
        raise ModelLoadError(f"The model failed on a {image_size}x{image_size} RGB image: {e}") from e
    if out.ndim != 2 or out.shape[1] != num_classes:
        raise ModelLoadError(f"The model outputs {tuple(out.shape)} but the dataset has {num_classes} classes.")
