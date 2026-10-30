# api.py
# Mediguard backend API (Day 2).
#
# Serves the results saved by engine.py and the sample images made by
# make_samples.py. It does not run any model itself, so every response is
# instant. To refresh the data, re-run `python engine.py`.
#
# Start with:
#   python -m uvicorn api:app --port 8000
# Interactive docs:
#   http://127.0.0.1:8000/docs

import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

RESULTS_DIR = "results"
SAMPLES_DIR = os.path.join(RESULTS_DIR, "samples")

app = FastAPI(title="Mediguard API", version="0.1")

# Allow the dashboard (a different local port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(SAMPLES_DIR, exist_ok=True)
app.mount("/files/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


def available_models():
    """Reads every results/<model>.json produced by engine.py."""
    models = {}
    if not os.path.isdir(RESULTS_DIR):
        return models
    for name in sorted(os.listdir(RESULTS_DIR)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(RESULTS_DIR, name)) as f:
                data = json.load(f)
            if "meta" in data and "baseline" in data:
                models[data["meta"]["model_key"]] = data
        except (OSError, ValueError):
            continue
    return models


def summarize(data):
    """Compact view of one model, used by /models and /compare."""
    return {
        "key": data["meta"]["model_key"],
        "display_name": data["meta"]["display_name"],
        "baseline": data["baseline"],
        "scores": data["scores"],
        "categories": {
            key: {
                "display": cat["display"],
                "avg_accuracy": cat["avg_accuracy"],
                "worst_accuracy": cat["worst_accuracy"],
                "retention": cat["retention"],
                "worst_case_retention": cat["worst_case_retention"],
            }
            for key, cat in data["categories"].items()
        },
    }


@app.get("/")
def root():
    return {"name": "Mediguard API", "docs": "/docs",
            "endpoints": ["/health", "/models", "/results/{model}", "/compare", "/samples"]}


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(available_models())}


@app.get("/models")
def list_models():
    """Short list of every evaluated model, for the selection screen."""
    return [
        {
            "key": s["key"],
            "display_name": s["display_name"],
            "clean_accuracy": s["baseline"]["accuracy"],
            "robustness_score": s["scores"]["robustness_score"],
            "rating": s["scores"]["rating"],
            "worst_case_score": s["scores"]["worst_case_score"],
            "worst_case_rating": s["scores"]["worst_case_rating"],
        }
        for s in map(summarize, available_models().values())
    ]


@app.get("/results/{model_key}")
def get_results(model_key: str):
    """Full report for one model: baseline, every disturbance, scores, flags, recommendations."""
    models = available_models()
    if model_key not in models:
        raise HTTPException(404, f"No results for '{model_key}'. Available: {list(models)}")
    return models[model_key]


@app.get("/compare")
def compare(models: Optional[str] = None):
    """Side-by-side summary. Use ?models=smallcnn,resnet18 or omit for all."""
    all_models = available_models()
    keys = [k.strip() for k in models.split(",")] if models else list(all_models)
    missing = [k for k in keys if k not in all_models]
    if missing:
        raise HTTPException(404, f"Unknown model(s): {missing}. Available: {list(all_models)}")
    return {"models": [summarize(all_models[k]) for k in keys]}


@app.get("/samples")
def get_samples():
    """Original and disturbed example X-rays, with image URLs."""
    path = os.path.join(SAMPLES_DIR, "manifest.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Samples not generated yet. Run: python make_samples.py")
    with open(path) as f:
        manifest = json.load(f)

    for entry in manifest["classes"].values():
        original = entry["original"]
        entry["original"] = {"file": original, "url": f"/files/samples/{original}"}
        for key, items in entry.items():
            if key == "original":
                continue
            for item in items:
                item["url"] = f"/files/samples/{item['file']}"
    return manifest
