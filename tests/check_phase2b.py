"""Temporary manual check for Phase 2b. Replaced by real tests in the testing phase."""
import io
import time
import zipfile

from fastapi.testclient import TestClient

from mediguard.api.main import app
from mediguard.config import DEMO_DIR, ROOT


def mini_zip(per_class=10):
    buf = io.BytesIO()
    src = ROOT / "data" / "chest_xray" / "chest_xray" / "test"
    with zipfile.ZipFile(buf, "w") as z:
        for cls in ("NORMAL", "PNEUMONIA"):
            for f in sorted((src / cls).iterdir())[:per_class]:
                z.write(f, f"mini/{cls}/{f.name}")
    return buf.getvalue()


def show(label, r, keys=()):
    detail = {k: r.json().get(k) for k in keys} if keys else ""
    print(f"{label:<34} {r.status_code} {detail}")


if __name__ == "__main__":
    with TestClient(app) as c:
        show("health", c.get("/api/health"))
        print("architectures:", [a["key"] for a in c.get("/api/architectures").json()])
        print("models       :", [m["id"] for m in c.get("/api/models").json()])
        print("datasets     :", [d["id"] for d in c.get("/api/datasets").json()])
        print("existing runs:", [(r["id"], r["rating"]) for r in c.get("/api/runs").json()])

        r = c.post("/api/datasets", files={"file": ("mini.zip", mini_zip(), "application/zip")}, data={"name": "Mini (20 images)"})
        show("upload dataset", r, ("id", "class_counts"))
        ds_id = r.json()["id"]
        show("upload non-zip as dataset", c.post("/api/datasets", files={"file": ("x.zip", b"hello")}, data={"name": "bad"}), ("detail",))

        weights = (DEMO_DIR / "smallcnn_pneumonia.pt").read_bytes()
        r = c.post("/api/models", files={"file": ("m.pt", weights)},
                   data={"name": "Uploaded SmallCNN", "kind": "weights", "architecture": "smallcnn", "num_classes": "2"})
        show("upload model", r, ("id", "architecture"))
        model_id = r.json()["id"]
        show("upload model, wrong architecture", c.post("/api/models", files={"file": ("m.pt", weights)},
             data={"name": "x", "kind": "weights", "architecture": "resnet18", "num_classes": "2"}), ("detail",))
        show("upload model, bad kind", c.post("/api/models", files={"file": ("m.pt", weights)},
             data={"name": "x", "kind": "onnx"}), ("detail",))

        show("run with unknown model", c.post("/api/runs", json={"model_id": "nope", "dataset_id": ds_id}))
        show("run with bad positive class", c.post("/api/runs", json={"model_id": model_id, "dataset_id": ds_id, "positive_class": "X"}), ("detail",))
        r = c.post("/api/runs", json={"model_id": model_id, "dataset_id": ds_id})
        show("start run", r, ("id", "status"))
        run_id = r.json()["id"]

        for _ in range(300):
            run = c.get(f"/api/runs/{run_id}").json()
            print(f"  run {run['status']} {run['done']}/{run['total']}", end="\r")
            if run["status"] in ("done", "failed"):
                break
            time.sleep(1)
        print(f"\nrun finished: {run['status']} {run['error'] or ''}")
        if run["status"] == "done":
            print("scores:", run["result"]["scores"])
            cmp = c.get(f"/api/compare?runs=smallcnn,resnet18").json()
            print("compare demo runs   : same_dataset =", cmp["same_dataset"], [x["id"] for x in cmp["runs"]])
            cmp = c.get(f"/api/compare?runs=smallcnn,{run_id}").json()
            print("compare with new run: same_dataset =", cmp["same_dataset"])
            show("compare with one run", c.get("/api/compare?runs=smallcnn"), ("detail",))

        show("delete run", c.delete(f"/api/runs/{run_id}"))
        show("delete model", c.delete(f"/api/models/{model_id}"))
        show("delete dataset", c.delete(f"/api/datasets/{ds_id}"))
        show("delete demo model (blocked)", c.delete("/api/models/demo-smallcnn"), ("detail",))
