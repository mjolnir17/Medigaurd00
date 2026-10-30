"""Classification metrics for any number of classes.

Binary datasets report the positive class (default: the second class).
Multi-class datasets report macro averages over per-class one-vs-rest values.
"""
import numpy as np


def _div(a, b):
    return a / b if b else 0.0


def compute_metrics(y_true, y_pred, confidence, class_names, positive=1):
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    total = int(cm.sum())

    per_class, counts = [], []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i].sum()) - tp
        fp = int(cm[:, i].sum()) - tp
        tn = total - tp - fn - fp
        sens, spec, prec = _div(tp, tp + fn), _div(tn, tn + fp), _div(tp, tp + fp)
        per_class.append({"class": name, "support": tp + fn, "sensitivity": sens,
                          "specificity": spec, "precision": prec, "f1": _div(2 * prec * sens, prec + sens)})
        counts.append((tp, tn, fp, fn))

    def summary(k):
        return per_class[positive][k] if n == 2 else float(np.mean([c[k] for c in per_class]))

    out = {"accuracy": float(np.trace(cm)) / total,
           **{k: summary(k) for k in ("precision", "sensitivity", "specificity", "f1")},
           "mean_confidence": float(confidence.mean())}
    out = {k: round(v, 4) for k, v in out.items()}
    if n == 2:
        out.update(zip(("tp", "tn", "fp", "fn"), counts[positive]))
    out["predicted_share"] = [round(float(x), 4) for x in cm.sum(axis=0) / total]
    out["confusion_matrix"] = cm.tolist()
    out["per_class"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in per_class]
    return out
