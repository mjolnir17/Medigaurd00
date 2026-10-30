"""Known architectures a user can pick when uploading a weights file.

Each entry builds an untrained network with a final layer sized for
`num_classes`, so a saved state_dict can be loaded into it.
"""
import torch.nn as nn
from torchvision import models as tv


class SmallCNN(nn.Module):
    """The custom CNN used in the demo. Layer order matches its saved weights."""

    def __init__(self, num_classes=2):
        super().__init__()
        blocks = []
        in_ch = 3
        for out_ch in (32, 64, 128, 128):
            blocks += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 14 * 14, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(256, num_classes),
        )  # the 14x14 map assumes 224x224 input

    def forward(self, x):
        return self.classifier(self.features(x))


def _resnet(fn):
    def build(n):
        m = fn(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n)
        return m
    return build


def _classifier_last(fn, index=None):
    """Replace the last classifier layer (torchvision VGG / MobileNet / EfficientNet)."""
    def build(n):
        m = fn(weights=None)
        m.classifier[index] = nn.Linear(m.classifier[index].in_features, n)
        return m
    return build


def _densenet(n):
    m = tv.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, n)
    return m


ARCHITECTURES = {
    "smallcnn": {"label": "SmallCNN (custom, 224x224 only)", "build": SmallCNN},
    "resnet18": {"label": "ResNet18", "build": _resnet(tv.resnet18)},
    "resnet34": {"label": "ResNet34", "build": _resnet(tv.resnet34)},
    "resnet50": {"label": "ResNet50", "build": _resnet(tv.resnet50)},
    "vgg16": {"label": "VGG16", "build": _classifier_last(tv.vgg16, 6)},
    "densenet121": {"label": "DenseNet121", "build": _densenet},
    "mobilenet_v2": {"label": "MobileNetV2", "build": _classifier_last(tv.mobilenet_v2, 1)},
    "efficientnet_b0": {"label": "EfficientNet-B0", "build": _classifier_last(tv.efficientnet_b0, 1)},
}
