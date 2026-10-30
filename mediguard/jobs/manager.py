"""Runs evaluations in the background, one at a time, and tracks their progress."""
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from .. import store
from ..config import DEFAULT_IMAGE_SIZE, NORMALIZATIONS, NUM_WORKERS, get_device
from ..data.dataset import DatasetError
from ..evaluation.engine import run_evaluation, save_result
from ..models.loader import ModelLoadError, load_model

log = logging.getLogger("mediguard.jobs")


def execute_run(run_id, spec, progress):
    model_meta = store.get_model(spec["model_id"])
    data_meta = store.get_dataset(spec["dataset_id"])
    mean, std = NORMALIZATIONS[spec.get("normalization", "imagenet")]
    model = load_model(model_meta["path"], model_meta["kind"], model_meta.get("architecture"),
                       model_meta.get("num_classes"), get_device())
    result = run_evaluation(
        model, data_meta["root"], run_id, model_meta["name"], dataset_name=data_meta["name"],
        image_size=int(spec.get("image_size", DEFAULT_IMAGE_SIZE)), mean=mean, std=std,
        max_images=spec.get("max_images"), positive_class=spec.get("positive_class"),
        num_workers=NUM_WORKERS, progress=progress)
    save_result(result)


class JobManager:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1)  # evaluations are GPU-bound: one at a time

    def submit(self, spec):
        """spec: model_id, dataset_id and optional image_size, normalization, max_images, positive_class."""
        store.get_model(spec["model_id"])      # KeyError if unknown
        store.get_dataset(spec["dataset_id"])
        if spec.get("normalization", "imagenet") not in NORMALIZATIONS:
            raise ValueError(f"Unknown normalization. Choose from {list(NORMALIZATIONS)}.")
        job = {"id": uuid.uuid4().hex[:8], "status": "queued", "done": 0, "total": 0,
               "error": None, "spec": spec, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        with self._lock:
            self._jobs[job["id"]] = job
        self._pool.submit(self._run, job)
        return dict(job)

    def _run(self, job):
        job["status"] = "running"

        def progress(done, total):
            job["done"], job["total"] = done, total

        try:
            execute_run(job["id"], job["spec"], progress)
            job["status"] = "done"
        except (ModelLoadError, DatasetError) as e:
            job["status"], job["error"] = "failed", str(e)
        except Exception as e:  # keep the server alive whatever goes wrong
            log.exception("evaluation failed")
            job["status"], job["error"] = "failed", f"Unexpected error: {e}"

    def get(self, job_id):
        return dict(self._jobs[job_id])  # KeyError if unknown

    def list(self):
        return [dict(j) for j in self._jobs.values()]
