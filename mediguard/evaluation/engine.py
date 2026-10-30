"""run_evaluation(): one model + one dataset -> the full Mediguard report."""
import json
import time

import torch

from ..config import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESULTS_DIR, SEED, get_device
from ..data.dataset import collect_samples, make_loader
from ..disturbances.registry import ALL, build_transform
from ..models.adapter import ModelAdapter
from ..models.loader import validate_model
from .metrics import compute_metrics
from .risk import make_recommendations, make_risk_flags
from .scoring import category_summary, overall_scores

DELTA_KEYS = ("accuracy", "precision", "sensitivity", "specificity", "f1", "mean_confidence")


def run_evaluation(model, dataset_root, key, display_name, *, dataset_name=None,
                   image_size=DEFAULT_IMAGE_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                   max_images=None, positive_class=None, batch_size=32, num_workers=2, progress=None):
    """progress(done, total) is called after every pass over the dataset."""
    device = get_device()
    data = collect_samples(dataset_root, max_images, SEED)
    names = data.class_names
    validate_model(model, len(names), image_size, device)
    positive = names.index(positive_class) if positive_class else len(names) - 1
    adapter = ModelAdapter(model, device, key)

    total = 1 + sum(len(d.levels) for d in ALL.values())
    done = 0

    def run(transform):
        nonlocal done
        loader = make_loader(data.samples, transform, batch_size, num_workers)
        torch.manual_seed(SEED)  # identical random noise on every run
        preds = adapter.predict(loader)
        done += 1
        if progress:
            progress(done, total)
        return preds

    base = run(build_transform(image_size, mean, std))
    baseline = compute_metrics(base.y_true, base.y_pred, base.confidence, names, positive)

    categories = {}
    for dist in ALL.values():
        conditions = []
        for level in dist.levels:
            p = run(build_transform(image_size, mean, std, dist, level.param))
            m = compute_metrics(base.y_true, p.y_pred, p.confidence, names, positive)
            conditions.append({
                "label": level.label,
                "description": level.description,
                "metrics": m,
                "delta": {k: round(m[k] - baseline[k], 4) for k in DELTA_KEYS},
                "prediction_change_rate": round(float((p.y_pred != base.y_pred).mean()), 4),
                "confidence_drop": round(baseline["mean_confidence"] - m["mean_confidence"], 4),
            })
        categories[dist.key] = category_summary(dist.display, conditions, baseline["accuracy"])

    flags = make_risk_flags(baseline, categories, names)
    return {
        "meta": {
            "model_key": key, "display_name": display_name,
            "dataset": dataset_name or str(data.root), "n_images": int(len(base.y_true)),
            "class_names": names, "class_counts": data.class_counts,
            "positive_class": names[positive] if len(names) == 2 else None,
            "image_size": image_size, "seed": SEED,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "baseline": baseline,
        "categories": categories,
        "scores": overall_scores(categories),
        "risk_flags": flags,
        "recommendations": make_recommendations(categories, flags),
    }


def save_result(result, results_dir=RESULTS_DIR):
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{result['meta']['model_key']}.json"
    path.write_text(json.dumps(result, indent=2))
    return path
