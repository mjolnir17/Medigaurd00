from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ... import store
from ...data.dataset import DatasetError
from ..uploads import save_upload

router = APIRouter(prefix="/api", tags=["datasets"])
MAX_DATASET_BYTES = 2 * 1024 ** 3


def _public(d):
    return {**{k: d.get(k) for k in ("id", "name", "class_names", "class_counts", "n_images", "created")},
            "demo": d.get("demo", False)}


@router.get("/datasets")
def list_datasets():
    return [_public(d) for d in store.list_datasets()]


@router.post("/datasets", status_code=201)
def upload_dataset(file: UploadFile = File(...), name: str = Form(...)):
    """A zip with one folder per class, each holding that class's images."""
    tmp = save_upload(file, MAX_DATASET_BYTES, ".zip")
    try:
        meta = store.save_dataset(tmp, name.strip() or file.filename)
    except DatasetError as e:
        raise HTTPException(400, str(e))
    finally:
        tmp.unlink(missing_ok=True)
    return _public(meta)


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str):
    try:
        store.delete_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, "Dataset not found.")
    except PermissionError as e:
        raise HTTPException(403, str(e))
