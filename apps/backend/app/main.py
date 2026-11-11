from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .imaging import (
    clean_binary_mask,
    decode_rgb,
    render_outputs,
    split_instances,
    split_instances_component_aware,
)
from .model import InferenceService
from .metastasis import MetastasisService


def _integer_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


MAX_UPLOAD_BYTES = _integer_env("NUCLEI_MAX_UPLOAD_MB", 20) * 1024 * 1024
MAX_IMAGE_PIXELS = _integer_env("NUCLEI_MAX_IMAGE_PIXELS", 40_000_000)
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/webp"}
service = InferenceService()
metastasis_service = MetastasisService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load()
    metastasis_service.load()
    yield


app = FastAPI(
    title="Nuclei Lens API",
    version="2.0.0",
    description="Research-only nuclei segmentation and PCam metastatic-pattern mapping for H&E images.",
    lifespan=lifespan,
)
origins = [item.strip() for item in os.getenv("NUCLEI_CORS_ORIGINS", "http://localhost:5173").split(",") if item.strip()]
origin_regex = os.getenv("NUCLEI_CORS_ORIGIN_REGEX", "").strip() or None
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    any_ready = service.state.ready or metastasis_service.state.ready
    nuclei_model = {
        "ready": service.state.ready,
        "status": service.state.status,
        "message": service.state.message,
        "threshold": service.state.config.get("mask_threshold"),
        "patch_size": service.state.config.get("patch_size", 256),
        "instance_pipeline": service.state.config.get("instance_pipeline_version", "v1_watershed"),
        "analysis_endpoint": "/api/segment",
        "capabilities": {"nuclei_segmentation": True, "instance_counting": True},
    }
    pcam_model = metastasis_service.health_model()
    return {
        "service": "nuclei-lens-api",
        "ready": any_ready,
        "status": "ready" if any_ready else "setup_required",
        "message": "At least one analysis model is ready." if any_ready else "No model artifacts are available.",
        "model": nuclei_model,
        "models": {"nuclei": nuclei_model, "metastasis": pcam_model},
        "endpoints": {
            "nuclei": "/api/segment",
            "metastasis": "/api/metastasis/analyze",
        },
    }


@app.post("/api/segment")
async def segment(image: UploadFile = File(...)) -> dict:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, TIFF, or WebP image.")
    if not service.state.ready:
        raise HTTPException(status_code=503, detail=service.state.message)

    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image is larger than the configured limit.")
    try:
        started = time.perf_counter()
        rgb = decode_rgb(payload, MAX_IMAGE_PIXELS)
        probability = service.predict_probability(rgb)
        config = service.state.config
        threshold = float(config.get("mask_threshold", config.get("threshold", 0.5)))
        min_size = int(config.get("min_object_size", 12))
        min_distance = int(config.get("watershed_min_distance", 5))
        mask = clean_binary_mask(probability, threshold, min_size)
        instance_config = config.get("instance_pipeline", {})
        instance_version = str(config.get("instance_pipeline_version", "v1_watershed"))
        if instance_version == "v2_component_aware":
            instances = split_instances_component_aware(probability, threshold, instance_config)
        else:
            instances = split_instances(mask, min_distance, min_size)
        visuals = render_outputs(rgb, mask, instances)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "filename": image.filename,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "predicted_count": int(instances.max()),
        "nuclei_area_percent": round(float(mask.mean() * 100), 2),
        "threshold": threshold,
        "instance_pipeline": instance_version,
        "inference_ms": elapsed_ms,
        "visuals": visuals,
        "notice": "Research demonstration only. Not for clinical diagnosis.",
    }


@app.post("/api/metastasis/analyze")
async def analyze_metastasis(
    image: UploadFile = File(...),
    operating_point: str = "balanced",
) -> dict:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, TIFF, or WebP image.")
    if not metastasis_service.state.ready:
        raise HTTPException(status_code=503, detail=metastasis_service.state.message)
    if operating_point not in {"balanced", "sensitivity"}:
        raise HTTPException(status_code=422, detail="operating_point must be balanced or sensitivity")

    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image is larger than the configured limit.")
    try:
        started = time.perf_counter()
        rgb = decode_rgb(payload, MAX_IMAGE_PIXELS)
        analysis = metastasis_service.analyze(rgb, operating_point)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PCam inference failed: {exc}") from exc

    return {
        "analysis_type": "metastasis_patch_map",
        "filename": image.filename,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        **analysis,
        "inference_ms": elapsed_ms,
        "benchmark": metastasis_service.state.config.get("evaluation", {}),
        "notice": (
            "Research demonstration only. Patch-level metastatic-pattern probability is not a "
            "patient diagnosis, tumour stage, grade, prognosis, or treatment recommendation."
        ),
    }

