"""Experimental HTTP API. Job state lives in memory; at merge time landcover_detection's
durable store, outbox and callbacks replace this file.

    uv run uvicorn s2lc.api:app --port 8000
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from s2lc import pipeline

OUT = Path("out")
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="sentinel2-lc", version="0.1.0")
# ponytail: one worker, in-memory jobs; durable state comes from landcover_detection on merge.
pool = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}


class ClassifyRequest(BaseModel):
    aoi: dict = Field(description="GeoJSON Polygon/MultiPolygon, Feature or FeatureCollection")
    year: int
    quarter: Literal[1, 2, 3, 4]
    output_formats: list[Literal["geotiff", "geojson", "image"]] = ["geotiff", "geojson"]
    gap_fill: bool = True


def _work(job_id: str, req: ClassifyRequest) -> None:
    job = jobs[job_id]
    job["status"] = "processing"
    try:
        files, summary = pipeline.run(
            req.aoi, req.year, req.quarter, OUT / job_id, tuple(req.output_formats),
            req.gap_fill, progress=lambda p: job.update(progress_percent=p))
        job.update(status="complete", progress_percent=100, result_summary=summary,
                   files=[f.name for f in files])
    except pipeline.AoiError as e:
        job.update(status="failed", error_code="INVALID_INPUT", error_message=str(e))
    except Exception as e:  # noqa: BLE001 - report, never crash the worker
        job.update(status="failed", error_code="INTERNAL_ERROR", error_message=type(e).__name__)


@app.post("/v1/classify", status_code=202)
async def classify(req: ClassifyRequest) -> dict:
    try:  # reject bad input before queueing
        pipeline.check_period(req.year, req.quarter)
        pipeline.parse_aoi(req.aoi)
    except (pipeline.AoiError, ValueError, KeyError, TypeError) as e:
        raise HTTPException(422, str(e)) from e
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"job_id": job_id, "status": "queued", "progress_percent": 0}
    asyncio.get_running_loop().run_in_executor(pool, _work, job_id, req)
    return {"accepted": True, "job_id": job_id}


@app.get("/v1/jobs/{job_id}")
def status(job_id: str) -> dict:
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    return jobs[job_id]


@app.get("/v1/jobs/{job_id}/files/{name}")
def file(job_id: str, name: str) -> FileResponse:
    if name not in jobs.get(job_id, {}).get("files", []):
        raise HTTPException(404, "unknown file")
    return FileResponse(OUT / job_id / name)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def demo() -> FileResponse:
    return FileResponse(STATIC / "index.html")
