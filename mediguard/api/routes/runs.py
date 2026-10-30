from fastapi import APIRouter, HTTPException, Request

from ... import store
from ..schemas import RunRequest

router = APIRouter(prefix="/api", tags=["runs"])


def _summary(r):
    m = r["meta"]
    return {"id": m["model_key"], "status": "done", "model_name": m["display_name"], "dataset": m["dataset"],
            "n_images": m["n_images"], "class_names": m["class_names"],
            "clean_accuracy": r["baseline"]["accuracy"], **r["scores"], "created": m["generated_at"]}


def _job_summary(j):
    try:
        model_name = store.get_model(j["spec"]["model_id"])["name"]
        dataset = store.get_dataset(j["spec"]["dataset_id"])["name"]
    except KeyError:
        model_name = dataset = "(deleted)"
    return {"id": j["id"], "status": j["status"], "model_name": model_name, "dataset": dataset,
            "done": j["done"], "total": j["total"], "error": j["error"], "created": j["created"]}


@router.post("/runs", status_code=202)
def start_run(body: RunRequest, request: Request):
    """Starts an evaluation in the background. Poll GET /api/runs/{id} for progress."""
    try:
        dataset = store.get_dataset(body.dataset_id)
        store.get_model(body.model_id)
    except KeyError:
        raise HTTPException(404, "Unknown model or dataset.")
    if body.positive_class and body.positive_class not in dataset["class_names"]:
        raise HTTPException(400, f"positive_class must be one of {dataset['class_names']}.")
    try:
        job = request.app.state.jobs.submit(body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": job["id"], "status": job["status"]}


@router.get("/runs")
def list_runs(request: Request):
    finished = {r["meta"]["model_key"]: r for r in store.list_results()}
    runs = [_summary(r) for r in finished.values()]
    runs += [_job_summary(j) for j in request.app.state.jobs.list() if j["id"] not in finished]
    return sorted(runs, key=lambda r: r["created"], reverse=True)


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request):
    try:
        job = request.app.state.jobs.get(run_id)
    except KeyError:
        job = None
    try:
        result = store.load_result(run_id)
    except KeyError:
        result = None
    if job is None and result is None:
        raise HTTPException(404, "Run not found.")
    status = job["status"] if job else "done"
    return {"id": run_id, "status": status, "done": job["done"] if job else None,
            "total": job["total"] if job else None, "error": job["error"] if job else None,
            "result": result if status == "done" else None}


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: str):
    try:
        store.delete_result(run_id)
    except KeyError:
        raise HTTPException(404, "Run not found.")


@router.get("/compare")
def compare(runs: str):
    """Side-by-side summary of finished runs: /api/compare?runs=id1,id2"""
    ids = [x.strip() for x in runs.split(",") if x.strip()]
    if len(ids) < 2:
        raise HTTPException(400, "Choose at least two runs to compare.")
    results = []
    for run_id in ids:
        try:
            results.append(store.load_result(run_id))
        except KeyError:
            raise HTTPException(404, f"Run '{run_id}' not found or not finished.")
    return {
        "same_dataset": len({r["meta"]["dataset"] for r in results}) == 1,
        "runs": [{
            "id": r["meta"]["model_key"], "display_name": r["meta"]["display_name"],
            "dataset": r["meta"]["dataset"], "baseline": r["baseline"], "scores": r["scores"],
            "categories": {k: {f: c[f] for f in ("display", "avg_accuracy", "worst_accuracy",
                                                 "retention", "worst_case_retention")}
                           for k, c in r["categories"].items()},
        } for r in results],
    }
