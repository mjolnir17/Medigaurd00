"""File-based storage for uploaded models, datasets and results (everything lives under storage/)."""
import json
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path

import torch

from .config import DATASETS_DIR, DEMO_DIR, RESULTS_DIR, ROOT, UPLOADED_MODELS_DIR
from .data.dataset import IMAGE_EXTENSIONS, DatasetError, collect_samples
from .models.loader import ModelLoadError, load_model, validate_model

MAX_UNZIPPED_BYTES = 5 * 1024 ** 3
MAX_FILES = 50_000
_ID = re.compile(r"^[A-Za-z0-9-]{1,40}$")


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _new_id():
    return uuid.uuid4().hex[:8]


def _meta_path(parent, item_id):
    if not _ID.match(item_id):
        raise KeyError(item_id)
    return parent / item_id / "meta.json"


def _write_meta(folder, meta):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))


def _list(parent):
    metas = [json.loads(p.read_text()) for p in parent.glob("*/meta.json")]
    return sorted(metas, key=lambda m: (m.get("created", ""), m["id"]))


def _get(parent, item_id):
    path = _meta_path(parent, item_id)
    if not path.exists():
        raise KeyError(item_id)
    return json.loads(path.read_text())


def _delete(parent, item_id):
    meta = _get(parent, item_id)
    if meta.get("demo"):
        raise PermissionError("The built-in demo items cannot be deleted.")
    shutil.rmtree(parent / item_id, ignore_errors=True)


# ---------- models ----------

def list_models():
    return _list(UPLOADED_MODELS_DIR)


def get_model(model_id):
    return _get(UPLOADED_MODELS_DIR, model_id)


def delete_model(model_id):
    _delete(UPLOADED_MODELS_DIR, model_id)


def save_model(src, name, kind, architecture=None, num_classes=None):
    """Copies an uploaded model into storage after checking that it loads."""
    model_id = _new_id()
    folder = UPLOADED_MODELS_DIR / model_id
    folder.mkdir(parents=True)
    dest = folder / "model.pt"
    shutil.copy(src, dest)
    try:
        model = load_model(dest, kind, architecture, num_classes)
        if kind == "weights":
            validate_model(model, num_classes, 224, torch.device("cpu"))
    except ModelLoadError:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    meta = {"id": model_id, "name": name, "kind": kind, "architecture": architecture,
            "num_classes": num_classes, "path": str(dest), "created": _now()}
    _write_meta(folder, meta)
    return meta


# ---------- datasets ----------

def list_datasets():
    return _list(DATASETS_DIR)


def get_dataset(dataset_id):
    return _get(DATASETS_DIR, dataset_id)


def delete_dataset(dataset_id):
    _delete(DATASETS_DIR, dataset_id)


def _safe_extract(zip_path, dest):
    """Extracts only image files, refusing paths that escape the destination folder."""
    dest.mkdir(parents=True, exist_ok=True)
    dest = dest.resolve()
    total = 0
    with zipfile.ZipFile(zip_path) as z:
        members = [i for i in z.infolist()
                   if not i.is_dir() and Path(i.filename).suffix.lower() in IMAGE_EXTENSIONS
                   and "__MACOSX" not in i.filename and not Path(i.filename).name.startswith("._")]
        if len(members) > MAX_FILES:
            raise DatasetError(f"Too many images (limit {MAX_FILES}).")
        for info in members:
            total += info.file_size
            if total > MAX_UNZIPPED_BYTES:
                raise DatasetError("The dataset is too large (limit 5 GB uncompressed).")
            target = (dest / info.filename).resolve()
            if dest not in target.parents:
                raise DatasetError("The zip contains an unsafe file path and was rejected.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def save_dataset(zip_path, name):
    dataset_id = _new_id()
    folder = DATASETS_DIR / dataset_id
    try:
        _safe_extract(zip_path, folder / "files")
        data = collect_samples(folder / "files")
    except zipfile.BadZipFile:
        shutil.rmtree(folder, ignore_errors=True)
        raise DatasetError("That file is not a valid zip archive.")
    except DatasetError:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    meta = {"id": dataset_id, "name": name, "root": str(data.root), "class_names": data.class_names,
            "class_counts": data.class_counts, "n_images": len(data.samples), "created": _now()}
    _write_meta(folder, meta)
    return meta


# ---------- results ----------

def list_results():
    out = []
    for p in sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []:
        try:
            out.append(json.loads(p.read_text()))
        except ValueError:
            continue
    return sorted(out, key=lambda r: r["meta"].get("generated_at", ""))


def load_result(run_id):
    if not _ID.match(run_id) or not (RESULTS_DIR / f"{run_id}.json").exists():
        raise KeyError(run_id)
    return json.loads((RESULTS_DIR / f"{run_id}.json").read_text())


def delete_result(run_id):
    load_result(run_id)
    (RESULTS_DIR / f"{run_id}.json").unlink()


# ---------- built-in demo items ----------

def ensure_demo():
    """Registers the two demo models and the demo dataset, if their files are present."""
    demo_models = [("demo-smallcnn", "SmallCNN (trained from scratch)", "smallcnn", "smallcnn_pneumonia.pt"),
                   ("demo-resnet18", "ResNet18 (pretrained, fine-tuned)", "resnet18", "resnet18_pneumonia.pt")]
    for model_id, name, arch, file in demo_models:
        path, folder = DEMO_DIR / file, UPLOADED_MODELS_DIR / model_id
        if path.exists() and not (folder / "meta.json").exists():
            _write_meta(folder, {"id": model_id, "name": name, "kind": "weights", "architecture": arch,
                                 "num_classes": 2, "path": str(path), "created": "0000", "demo": True})
    data_dir, folder = ROOT / "data" / "chest_xray", DATASETS_DIR / "demo-pneumonia"
    if data_dir.exists() and not (folder / "meta.json").exists():
        data = collect_samples(data_dir)
        _write_meta(folder, {"id": "demo-pneumonia", "name": "Chest X-ray pneumonia (test split)",
                             "root": str(data.root), "class_names": data.class_names,
                             "class_counts": data.class_counts, "n_images": len(data.samples),
                             "created": "0000", "demo": True})
