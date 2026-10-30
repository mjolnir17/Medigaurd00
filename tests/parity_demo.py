"""Temporary acceptance check: the new engine must reproduce the saved results of the old scripts."""
import json

from mediguard.config import DEMO_DIR, ROOT, get_device
from mediguard.evaluation.engine import run_evaluation, save_result
from mediguard.models.loader import load_model

MODELS = [("smallcnn", "SmallCNN (trained from scratch)", "smallcnn_pneumonia.pt"),
          ("resnet18", "ResNet18 (pretrained + fine-tuned)", "resnet18_pneumonia.pt")]


def compare(new, old):
    diffs = []
    if new["baseline"]["accuracy"] != old["baseline"]["accuracy"]:
        diffs.append("baseline accuracy")
    for cat, old_cat in old["categories"].items():
        for nc, oc in zip(new["categories"][cat]["conditions"], old_cat["conditions"]):
            for k in ("accuracy", "sensitivity", "specificity"):
                if nc["metrics"][k] != oc["metrics"][k]:
                    diffs.append(f"{cat}/{oc['label']}/{k}")
    for k in ("robustness_score", "worst_case_score"):
        if new["scores"][k] != old["scores"][k]:
            diffs.append(k)
    if len(new["risk_flags"]) != len(old["risk_flags"]):
        diffs.append("risk flag count")
    if len(new["recommendations"]) != len(old["recommendations"]):
        diffs.append("recommendation count")
    return diffs


if __name__ == "__main__":
    device = get_device()
    for key, display, file in MODELS:
        model = load_model(DEMO_DIR / file, "weights", key, 2, device)
        result = run_evaluation(model, ROOT / "data" / "chest_xray", key, display,
                                progress=lambda d, t, k=key: print(f"  {k}: pass {d}/{t}", end="\r"))
        path = save_result(result)
        old = json.loads((ROOT / "results" / f"{key}.json").read_text())
        s, o = result["scores"], old["scores"]
        print(f"\n{key}: robustness {s['robustness_score']} (old {o['robustness_score']}), "
              f"worst-case {s['worst_case_score']} (old {o['worst_case_score']}), saved {path.name}")
        diffs = compare(result, old)
        print("  PARITY OK" if not diffs else f"  MISMATCH in: {diffs[:10]}")
