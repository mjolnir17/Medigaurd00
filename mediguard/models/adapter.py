"""Uniform prediction interface: the rest of Mediguard only calls predict()."""
from typing import NamedTuple

import numpy as np
import torch


class Predictions(NamedTuple):
    y_true: np.ndarray      # true class index per image
    y_pred: np.ndarray      # predicted class index per image
    confidence: np.ndarray  # highest softmax probability per image
    probs: np.ndarray       # full probability matrix (images x classes)


class ModelAdapter:
    def __init__(self, model, device, name="model"):
        self.model = model.to(device).eval()
        self.device = device
        self.name = name

    @torch.no_grad()
    def predict(self, loader):
        labels, probs = [], []
        for images, y in loader:
            logits = self.model(images.to(self.device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.numpy())
        probs = np.concatenate(probs)
        return Predictions(np.concatenate(labels), probs.argmax(axis=1), probs.max(axis=1), probs)
