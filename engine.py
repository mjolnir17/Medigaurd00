# engine.py
# Mediguard evaluation engine.
#
# Runs one model through the clean baseline and every disturbance, and
# RETURNS structured results (also saved as JSON in results/<model>.json
# for the backend and dashboard).
#
# Per condition it records: accuracy, precision, sensitivity, specificity,
# F1, FP/FN counts, mean confidence, change vs baseline, prediction change
# rate and confidence drop. It also computes the average robustness score,
# a worst-case score, rule-based clinical risk flags and recommendations.
#
# Assumes binary classification with class index 1 = positive (PNEUMONIA).
#
# Run with:
#   python engine.py            (all registered models)
#   python engine.py resnet18   (one model)

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from dataset_loader import get_dataloaders
from model_adapter import ModelAdapter
from robustness_report import MODEL_REGISTRY
from evaluate_noise import get_noisy_test_loader
from evaluate_blur import get_blurred_test_loader, BLUR_SEVERITIES
from evaluate_brightness import get_brightness_test_loader, BRIGHTNESS_SEVERITIES
from evaluate_contrast import get_contrast_test_loader, CONTRAST_SEVERITIES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "results"
SEED = 1234  # fixed so the random noise is identical on every run
NOISE_SEVERITIES = {"low": 0.05, "medium": 0.10, "high": 0.20}
DELTA_KEYS = ["accuracy", "precision", "sensitivity", "specificity", "f1", "mean_confidence"]

# Shown on the dashboard when a category is weak
RECOMMENDATIONS = {
    "noise": [
        "Add denoising (e.g. median or non-local means filtering) before inference",
        "Retrain with Gaussian-noise augmentation",
    ],
    "blur": [
        "Add a sharpness check that rejects or flags blurry scans",
        "Retrain with blur augmentation",
    ],
    "brightness": [
        "Normalize exposure before inference (e.g. histogram equalization)",
        "Retrain with brightness augmentation, especially darker images",
    ],
    "contrast": [
        "Apply contrast enhancement (e.g. CLAHE) before inference",
        "Retrain with contrast augmentation, especially low-contrast images",
    ],
}


# ---------- inference and metrics ----------

def infer(adapter, loader):
    """Returns (y_true, y_pred, confidence) as numpy arrays.
    Confidence = highest softmax probability for each image."""
    adapter.model.eval()
    y_true, y_pred, conf = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(adapter.device)
            probs = torch.softmax(adapter.model(images), dim=1).cpu()
            c, p = probs.max(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(p.tolist())
            conf.extend(c.tolist())
    return np.array(y_true), np.array(y_pred), np.array(conf)


def compute_metrics(y_true, y_pred, conf):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) else 0.0)
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ---------- conditions to test ----------

def build_conditions():
    """category -> display name + list of (label, description, loader factory)."""
    cats = {}

    cats["noise"] = {
        "display": "Gaussian noise",
        "items": [
            (label, f"sigma {v}", lambda v=v: get_noisy_test_loader(v)[0])
            for label, v in NOISE_SEVERITIES.items()
        ],
    }

    cats["blur"] = {
        "display": "Gaussian blur",
        "items": [
            (label, f"kernel {k}, sigma {s}",
             lambda k=k, s=s: get_blurred_test_loader(k, s)[0])
            for label, (_, (k, s)) in zip(["low", "medium", "high"], BLUR_SEVERITIES.items())
        ],
    }

    cats["brightness"] = {
        "display": "Brightness",
        "items": [
            (label, f"factor {f}", lambda f=f: get_brightness_test_loader(f)[0])
            for label, f in BRIGHTNESS_SEVERITIES.items()
        ],
    }

    cats["contrast"] = {
        "display": "Contrast",
        "items": [
            (label, f"factor {f}", lambda f=f: get_contrast_test_loader(f)[0])
            for label, f in CONTRAST_SEVERITIES.items()
        ],
    }
    return cats


# ---------- scoring, risk, recommendations ----------

def rating_for_score(score):
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Moderate"
    elif score >= 40:
        return "Weak"
    return "Poor"


def make_risk_flags(baseline, categories):
    flags = []
    if baseline["specificity"] < 0.6:
        flags.append({
            "level": "medium", "condition": "Clean images",
            "message": f"Even on clean images the model flags "
                       f"{round((1 - baseline['specificity']) * 100)}% of healthy patients "
                       f"as diseased (false alarms).",
        })
    if baseline["sensitivity"] < 0.9:
        flags.append({
            "level": "high", "condition": "Clean images",
            "message": f"Even on clean images the model misses "
                       f"{round((1 - baseline['sensitivity']) * 100)}% of diseased patients.",
        })

    for cat in categories.values():
        for c in cat["conditions"]:
            m = c["metrics"]
            name = f"{cat['display']} ({c['label']})"
            if m["specificity"] < 0.1 and m["sensitivity"] > 0.95:
                flags.append({
                    "level": "high", "condition": name,
                    "message": "Model collapses to predicting disease for almost every "
                               "patient and can no longer identify healthy cases.",
                })
            elif m["sensitivity"] < 0.1 and m["specificity"] > 0.95:
                flags.append({
                    "level": "high", "condition": name,
                    "message": "Model collapses to predicting healthy for almost every "
                               "patient and misses nearly all disease.",
                })
            elif m["sensitivity"] < baseline["sensitivity"] - 0.10:
                flags.append({
                    "level": "high", "condition": name,
                    "message": f"Missed diagnoses increase: sensitivity falls from "
                               f"{round(baseline['sensitivity'] * 100)}% to "
                               f"{round(m['sensitivity'] * 100)}%.",
                })
            elif m["specificity"] < baseline["specificity"] - 0.20:
                flags.append({
                    "level": "medium", "condition": name,
                    "message": f"False alarms rise: specificity falls from "
                               f"{round(baseline['specificity'] * 100)}% to "
                               f"{round(m['specificity'] * 100)}%.",
                })
    return flags


def make_recommendations(categories, flags):
    recs = []
    for key, cat in categories.items():
        if cat["retention"] < 0.90 or cat["worst_case_retention"] < 0.80:
            recs.append({
                "category": cat["display"],
                "priority": "high" if cat["retention"] < 0.85 else "medium",
                "actions": RECOMMENDATIONS[key],
            })
    if any("collapses" in f["message"] for f in flags):
        recs.append({
            "category": "Input quality control",
            "priority": "high",
            "actions": [
                "Add an image-quality gate that rejects very dark, flat or blurry scans "
                "instead of letting the model guess",
            ],
        })
    return recs


# ---------- main evaluation ----------

def load_model(model_key):
    cfg = MODEL_REGISTRY[model_key]
    model = cfg["build_fn"](num_classes=2).to(device)
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
    return ModelAdapter(model, device, name=model_key), cfg


def run_evaluation(model_key):
    adapter, cfg = load_model(model_key)
    _, _, clean_loader, class_names = get_dataloaders()

    torch.manual_seed(SEED)
    y_true, base_pred, base_conf = infer(adapter, clean_loader)
    baseline = compute_metrics(y_true, base_pred, base_conf)

    categories = {}
    for cat_key, cat in build_conditions().items():
        print(f"  [{model_key}] {cat['display']}...")
        conditions = []
        for label, description, make_loader in cat["items"]:
            loader = make_loader()
            torch.manual_seed(SEED)
            _, pred, conf = infer(adapter, loader)
            m = compute_metrics(y_true, pred, conf)
            conditions.append({
                "label": label,
                "description": description,
                "metrics": m,
                "delta": {k: round(m[k] - baseline[k], 4) for k in DELTA_KEYS},
                "prediction_change_rate": round(float((pred != base_pred).mean()), 4),
                "confidence_drop": round(baseline["mean_confidence"] - m["mean_confidence"], 4),
            })

        accs = [c["metrics"]["accuracy"] for c in conditions]
        avg_acc, worst_acc = sum(accs) / len(accs), min(accs)
        categories[cat_key] = {
            "display": cat["display"],
            "conditions": conditions,
            "avg_accuracy": round(avg_acc, 4),
            "worst_accuracy": round(worst_acc, 4),
            "retention": round(min(avg_acc / baseline["accuracy"], 1.0), 4),
            "worst_case_retention": round(min(worst_acc / baseline["accuracy"], 1.0), 4),
        }

    score = sum(c["retention"] for c in categories.values()) / len(categories) * 100
    worst_score = sum(c["worst_case_retention"] for c in categories.values()) / len(categories) * 100

    flags = make_risk_flags(baseline, categories)
    recs = make_recommendations(categories, flags)

    result = {
        "meta": {
            "model_key": model_key,
            "display_name": cfg["display_name"],
            "dataset": "Chest X-Ray Images (Pneumonia), test split",
            "n_images": int(len(y_true)),
            "class_names": class_names,
            "class_counts": {class_names[i]: int((y_true == i).sum()) for i in range(len(class_names))},
            "seed": SEED,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "baseline": baseline,
        "categories": categories,
        "scores": {
            "robustness_score": round(score, 2),
            "rating": rating_for_score(score),
            "worst_case_score": round(worst_score, 2),
            "worst_case_rating": rating_for_score(worst_score),
        },
        "risk_flags": flags,
        "recommendations": recs,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {path}")
    return result


def main():
    keys = list(MODEL_REGISTRY.keys())
    if len(sys.argv) > 1:
        key = sys.argv[1].lower()
        if key not in MODEL_REGISTRY:
            print(f"Unknown model '{key}'. Available: {keys}")
            sys.exit(1)
        keys = [key]

    for key in keys:
        print(f"\nEvaluating {key}...")
        r = run_evaluation(key)
        print(f"  Clean accuracy:    {r['baseline']['accuracy']:.4f}")
        print(f"  Robustness score:  {r['scores']['robustness_score']} ({r['scores']['rating']})")
        print(f"  Worst-case score:  {r['scores']['worst_case_score']} ({r['scores']['worst_case_rating']})")
        print(f"  Risk flags: {len(r['risk_flags'])}, recommendations: {len(r['recommendations'])}")


if __name__ == "__main__":
    main()
