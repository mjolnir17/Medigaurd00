"""Rule-based risk flags and recommendations. Thresholds are heuristic, not clinically validated."""

RECOMMENDATIONS = {
    "noise": ["Add denoising (e.g. median or non-local means filtering) before inference",
              "Retrain with Gaussian-noise augmentation"],
    "blur": ["Add a sharpness check that rejects or flags blurry scans", "Retrain with blur augmentation"],
    "brightness": ["Normalize exposure before inference (e.g. histogram equalization)",
                   "Retrain with brightness augmentation, especially darker images"],
    "contrast": ["Apply contrast enhancement (e.g. CLAHE) before inference",
                 "Retrain with contrast augmentation, especially low-contrast images"],
}


def make_risk_flags(baseline, categories, class_names):
    binary = len(class_names) == 2
    flags = []

    def add(level, condition, message):
        flags.append({"level": level, "condition": condition, "message": message})

    if binary:
        if baseline["specificity"] < 0.6:
            add("medium", "Clean images", f"Even on clean images the model flags "
                f"{round((1 - baseline['specificity']) * 100)}% of healthy patients as diseased (false alarms).")
        if baseline["sensitivity"] < 0.9:
            add("high", "Clean images", f"Even on clean images the model misses "
                f"{round((1 - baseline['sensitivity']) * 100)}% of diseased patients.")
    elif baseline["sensitivity"] < 0.8:
        add("high", "Clean images", f"Even on clean images the average per-class sensitivity is only "
            f"{round(baseline['sensitivity'] * 100)}%.")

    base_top = max(baseline["predicted_share"])
    for cat in categories.values():
        for c in cat["conditions"]:
            m, name = c["metrics"], f"{cat['display']} ({c['label']})"
            sens_drop = baseline["sensitivity"] - m["sensitivity"]
            spec_drop = baseline["specificity"] - m["specificity"]
            if binary and m["specificity"] < 0.1 and m["sensitivity"] > 0.95:
                add("high", name, "Model collapses to predicting disease for almost every patient "
                    "and can no longer identify healthy cases.")
            elif binary and m["sensitivity"] < 0.1 and m["specificity"] > 0.95:
                add("high", name, "Model collapses to predicting healthy for almost every patient "
                    "and misses nearly all disease.")
            elif not binary and max(m["predicted_share"]) >= 0.9 and base_top < 0.9:
                top = class_names[m["predicted_share"].index(max(m["predicted_share"]))]
                add("high", name, f"Model collapses to predicting '{top}' for almost every image.")
            elif sens_drop > 0.10:
                add("high", name, f"Missed diagnoses increase: sensitivity falls from "
                    f"{round(baseline['sensitivity'] * 100)}% to {round(m['sensitivity'] * 100)}%.")
            elif spec_drop > 0.20:
                add("medium", name, f"False alarms rise: specificity falls from "
                    f"{round(baseline['specificity'] * 100)}% to {round(m['specificity'] * 100)}%.")
    return flags


def make_recommendations(categories, flags):
    recs = []
    for key, cat in categories.items():
        if cat["retention"] < 0.90 or cat["worst_case_retention"] < 0.80:
            recs.append({"category": cat["display"],
                         "priority": "high" if cat["retention"] < 0.85 else "medium",
                         "actions": RECOMMENDATIONS[key]})
    if any("collapses" in f["message"] for f in flags):
        recs.append({"category": "Input quality control", "priority": "high",
                     "actions": ["Add an image-quality gate that rejects very dark, flat or blurry "
                                 "scans instead of letting the model guess"]})
    return recs
