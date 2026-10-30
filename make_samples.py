# make_samples.py
# Generates example images for the dashboard's "original vs disturbed"
# comparison: one NORMAL and one PNEUMONIA test X-ray, each shown clean and
# with every disturbance at every severity used in the evaluation.
#
# Output: results/samples/*.png and results/samples/manifest.json
# The images do not depend on any model, so this only needs to run once.
#
# Run with:
#   python make_samples.py

import json
import os

import torch
from PIL import Image
from torchvision import transforms

from dataset_loader import DATA_ROOT, IMAGE_SIZE
from evaluate_blur import BLUR_SEVERITIES
from evaluate_brightness import BRIGHTNESS_SEVERITIES
from evaluate_contrast import CONTRAST_SEVERITIES
from engine import NOISE_SEVERITIES

OUT_DIR = os.path.join("results", "samples")
SEED = 1234
CLASSES = ["NORMAL", "PNEUMONIA"]

# Same normalization the models use. Noise is added in this normalized space,
# exactly as in the evaluation, then converted back to a viewable image.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def first_image(class_name):
    folder = os.path.join(DATA_ROOT, "test", class_name)
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return os.path.join(folder, files[0])


def add_noise(img, sigma):
    t = transforms.ToTensor()(img)
    t = (t - MEAN) / STD
    t = t + torch.randn_like(t) * sigma
    t = (t * STD + MEAN).clamp(0, 1)
    return transforms.ToPILImage()(t)


def build_conditions():
    """category -> list of (label, description, function image -> image)."""
    cats = {}
    cats["noise"] = [
        (label, f"sigma {v}", lambda img, v=v: add_noise(img, v))
        for label, v in NOISE_SEVERITIES.items()
    ]
    cats["blur"] = [
        (label, f"kernel {k}, sigma {s}",
         lambda img, k=k, s=s: transforms.GaussianBlur(kernel_size=k, sigma=s)(img))
        for label, (k, s) in zip(["low", "medium", "high"], BLUR_SEVERITIES.values())
    ]
    cats["brightness"] = [
        (label, f"factor {f}",
         lambda img, f=f: transforms.ColorJitter(brightness=(f, f))(img))
        for label, f in BRIGHTNESS_SEVERITIES.items()
    ]
    cats["contrast"] = [
        (label, f"factor {f}",
         lambda img, f=f: transforms.ColorJitter(contrast=(f, f))(img))
        for label, f in CONTRAST_SEVERITIES.items()
    ]
    return cats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    manifest = {"image_size": IMAGE_SIZE, "classes": {}}
    count = 0

    for class_name in CLASSES:
        src = first_image(class_name)
        base = Image.open(src).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

        original_file = f"{class_name}_original.png"
        base.save(os.path.join(OUT_DIR, original_file))
        count += 1

        entry = {"original": original_file}
        for cat_key, conditions in build_conditions().items():
            items = []
            for label, description, fn in conditions:
                file_name = f"{class_name}_{cat_key}_{label}.png"
                fn(base).save(os.path.join(OUT_DIR, file_name))
                items.append({"label": label, "description": description, "file": file_name})
                count += 1
            entry[cat_key] = items
        manifest["classes"][class_name] = entry

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {count} images and manifest.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
