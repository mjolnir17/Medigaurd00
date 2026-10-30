"""Robustness scoring.

Retention = a category's average accuracy / clean accuracy (capped at 1).
Robustness score = mean retention over categories x 100.
Worst-case score uses each category's single worst severity instead of the average.
"""


def rating_for_score(score):
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Moderate"
    if score >= 40:
        return "Weak"
    return "Poor"


def category_summary(display, conditions, baseline_accuracy):
    base = max(baseline_accuracy, 1e-9)
    accs = [c["metrics"]["accuracy"] for c in conditions]
    avg, worst = sum(accs) / len(accs), min(accs)
    return {
        "display": display,
        "conditions": conditions,
        "avg_accuracy": round(avg, 4),
        "worst_accuracy": round(worst, 4),
        "retention": round(min(avg / base, 1.0), 4),
        "worst_case_retention": round(min(worst / base, 1.0), 4),
    }


def overall_scores(categories):
    n = len(categories)
    score = sum(c["retention"] for c in categories.values()) / n * 100
    worst = sum(c["worst_case_retention"] for c in categories.values()) / n * 100
    return {"robustness_score": round(score, 2), "rating": rating_for_score(score),
            "worst_case_score": round(worst, 2), "worst_case_rating": rating_for_score(worst)}
