from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ... import store
from ...models.architectures import ARCHITECTURES
from ...models.loader import ModelLoadError
from ..uploads import save_upload

router = APIRouter(prefix="/api", tags=["models"])
MAX_MODEL_BYTES = 1024 ** 3


def _public(m):
    return {**{k: m.get(k) for k in ("id", "name", "kind", "architecture", "num_classes", "created")},
            "demo": m.get("demo", False)}


@router.get("/architectures")
def architectures():
    """Architectures a weights file can be loaded into."""
    return [{"key": k, "label": v["label"]} for k, v in ARCHITECTURES.items()]


@router.get("/models")
def list_models():
    return [_public(m) for m in store.list_models()]


@router.post("/models", status_code=201)
def upload_model(file: UploadFile = File(...), name: str = Form(...), kind: str = Form(...),
                 architecture: Optional[str] = Form(None), num_classes: Optional[int] = Form(None)):
    if kind not in ("weights", "torchscript"):
        raise HTTPException(400, "kind must be 'weights' or 'torchscript'.")
    if kind == "weights":
        if architecture not in ARCHITECTURES:
            raise HTTPException(400, f"Choose an architecture: {list(ARCHITECTURES)}")
        if not num_classes or num_classes < 2:
            raise HTTPException(400, "Enter the number of classes the model was trained on (at least 2).")
    else:
        architecture = num_classes = None
    tmp = save_upload(file, MAX_MODEL_BYTES, ".pt")
    try:
        meta = store.save_model(tmp, name.strip() or file.filename, kind, architecture, num_classes)
    except ModelLoadError as e:
        raise HTTPException(400, str(e))
    finally:
        tmp.unlink(missing_ok=True)
    return _public(meta)


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: str):
    try:
        store.delete_model(model_id)
    except KeyError:
        raise HTTPException(404, "Model not found.")
    except PermissionError as e:
        raise HTTPException(403, str(e))
