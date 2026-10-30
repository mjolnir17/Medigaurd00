"""Temporary manual check for Phase 2a. Replaced by real tests in the testing phase."""
import tempfile
import time
import zipfile
from pathlib import Path

from mediguard import store
from mediguard.config import DEMO_DIR, ROOT
from mediguard.data.dataset import DatasetError
from mediguard.jobs.manager import JobManager
from mediguard.models.loader import ModelLoadError


def make_zip(path, per_class=10):
    src = ROOT / "data" / "chest_xray" / "chest_xray" / "test"
    with zipfile.ZipFile(path, "w") as z:
        for cls in ("NORMAL", "PNEUMONIA"):
            for f in sorted((src / cls).iterdir())[:per_class]:
                z.write(f, f"mini/{cls}/{f.name}")


if __name__ == "__main__":
    store.ensure_demo()
    print("models  :", [m["id"] for m in store.list_models()])
    print("datasets:", [d["id"] for d in store.list_datasets()])

    tmp = Path(tempfile.mkdtemp())
    make_zip(tmp / "mini.zip")
    ds = store.save_dataset(tmp / "mini.zip", "Mini pneumonia (20 images)")
    print("dataset uploaded:", ds["id"], ds["class_names"], ds["class_counts"])

    with zipfile.ZipFile(tmp / "evil.zip", "w") as z:
        z.writestr("../../evil.png", b"x")
    try:
        store.save_dataset(tmp / "evil.zip", "evil")
    except DatasetError as e:
        print("unsafe zip rejected:", e)
    try:
        store.save_dataset(tmp / "mini.zip".replace("mini", "missing") if False else DEMO_DIR / "smallcnn_pneumonia.pt", "not a zip")
    except DatasetError as e:
        print("non-zip rejected:", e)

    model = store.save_model(DEMO_DIR / "smallcnn_pneumonia.pt", "Uploaded SmallCNN", "weights", "smallcnn", 2)
    print("model uploaded:", model["id"])
    try:
        store.save_model(DEMO_DIR / "resnet18_pneumonia.pt", "wrong architecture", "weights", "smallcnn", 2)
    except ModelLoadError:
        print("wrong architecture rejected")

    jobs = JobManager()
    job = jobs.submit({"model_id": model["id"], "dataset_id": ds["id"]})
    while jobs.get(job["id"])["status"] in ("queued", "running"):
        j = jobs.get(job["id"])
        print(f"  job {j['status']} {j['done']}/{j['total']}", end="\r")
        time.sleep(1)
    j = jobs.get(job["id"])
    print(f"\njob {j['status']}", j["error"] or "")
    if j["status"] == "done":
        r = store.load_result(job["id"])
        print("images:", r["meta"]["n_images"], "| scores:", r["scores"])
        store.delete_result(job["id"])

    store.delete_model(model["id"])
    store.delete_dataset(ds["id"])
    print("cleaned up test uploads")
