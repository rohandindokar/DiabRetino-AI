"""FastAPI routes for the Diabetic Retinopathy severity grading API."""
import os
import base64
import logging
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

from app.core.config import CLASSES, CLASS_DISPLAY_NAMES, SEVERITY_METADATA
from app.models.inference import DRInferenceEngine
from app.models.preprocessing import image_to_base64

logger = logging.getLogger("diabretino.api")
router = APIRouter(prefix="/api")
engine = DRInferenceEngine()

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@router.get("/health")
def health_check():
    """System status, active device, and checkpoint-load state."""
    return {
        "status": "healthy",
        "service": "Diabetic Retinopathy Severity Grading API",
        "version": "2.0.0",
        "device": str(engine.device),
        "available_models": list(engine.models.keys()),
        "active_model": engine.active_model_name,
        "checkpoints_loaded": engine.model_loaded_from_checkpoint,
    }


@router.get("/metadata")
def get_metadata():
    """ICDR clinical scale and staging criteria."""
    return {
        "classes": CLASSES,
        "class_display_names": CLASS_DISPLAY_NAMES,
        "severity_metadata": SEVERITY_METADATA,
    }


@router.get("/models")
def get_models():
    from app.models.architectures import MODEL_REGISTRY
    return {
        "models": [
            {**MODEL_REGISTRY[k], "id": k,
             "checkpoint_loaded": engine.model_loaded_from_checkpoint.get(k, False)}
            for k in MODEL_REGISTRY
        ]
    }


@router.get("/samples")
def get_samples():
    """Built-in reference fundus images for one-click frontend testing."""
    sample_files = [
        ("stage_0_no_dr.png", 0, "Normal retina"),
        ("stage_1_mild.png", 1, "Mild microaneurysms"),
        ("stage_2_moderate.png", 2, "Moderate hemorrhages/exudates"),
        ("stage_3_severe.png", 3, "Severe blot hemorrhages"),
        ("stage_4_proliferative.png", 4, "Proliferative neovascularization"),
    ]
    results = []
    for fname, stage_id, label in sample_files:
        fpath = os.path.join(SAMPLES_DIR, fname)
        if os.path.isfile(fpath):
            try:
                with Image.open(fpath) as img:
                    b64 = image_to_base64(img.resize((140, 140)))
                results.append({
                    "filename": fname,
                    "stage_id": stage_id,
                    "label": label,
                    "metadata": SEVERITY_METADATA[stage_id],
                    "thumbnail_base64": b64,
                })
            except Exception as e:
                logger.warning("Sample %s failed: %s", fname, e)
    return {"samples": results}


@router.get("/samples/{filename}")
def get_sample_file(filename: str):
    safe = os.path.basename(filename)
    fpath = os.path.realpath(os.path.join(SAMPLES_DIR, safe))
    if not fpath.startswith(os.path.realpath(SAMPLES_DIR)) or not os.path.isfile(fpath):
        raise HTTPException(404, "Sample not found")
    return FileResponse(fpath, media_type="image/png", filename=safe)


@router.get("/metrics")
def get_metrics():
    """Trained-model evaluation metrics (accuracy, F1, confusion matrix)."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "metrics_report.json")
    if not os.path.isfile(path):
        raise HTTPException(404, "No training report found yet.")
    with open(path) as f:
        import json
        return json.load(f)


@router.post("/predict")
async def predict_single(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None),
    model_name: str = Form("hybrid_vit_cnn"),
    apply_enhancement: bool = Form(False),
):
    """Classify a retinal fundus image; returns severity, confidence,
    probability distribution, Grad-CAM map, and ICDR guidelines."""
    image_bytes = None

    if file is not None and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(400, f"Unsupported format '{ext}'. Use JPEG/PNG/WebP.")
        image_bytes = await file.read()
        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(400, "File too large (max 20 MB).")
    elif sample_filename:
        safe = os.path.basename(sample_filename)
        fpath = os.path.realpath(os.path.join(SAMPLES_DIR, safe))
        if not fpath.startswith(os.path.realpath(SAMPLES_DIR)) or not os.path.isfile(fpath):
            raise HTTPException(404, "Sample not found")
        with open(fpath, "rb") as fh:
            image_bytes = fh.read()
    else:
        raise HTTPException(400, "Provide an image file or sample_filename")

    try:
        return JSONResponse(content=engine.predict(
            image_bytes, model_name=model_name, apply_enhancement=apply_enhancement))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Inference error: %s", e, exc_info=True)
        raise HTTPException(500, f"Inference error: {e}")


@router.post("/batch-predict")
async def predict_batch(
    files: List[UploadFile] = File(...),
    model_name: str = Form("hybrid_vit_cnn"),
    apply_enhancement: bool = Form(True),
):
    """Batch-classify up to 20 images with a stage-distribution summary."""
    if len(files) > 20:
        raise HTTPException(400, "Batch limit is 20 images.")

    # Batch uses plain inference (matches training) for consistent results.
    items, stage_counts = [], {i: 0 for i in range(5)}
    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            items.append({"filename": f.filename, "error": "Not an image"})
            continue
        data = await f.read()
        if not data:
            items.append({"filename": f.filename, "error": "Empty file"})
            continue
        try:
            res = engine.predict(data, model_name=model_name,
                                 apply_enhancement=apply_enhancement)
            pred = res["prediction"]
            stage_counts[pred["class_id"]] += 1
            items.append({
                "filename": f.filename,
                "prediction": pred,
                "thumbnail": res["visualizations"]["original_image"],
                "heatmap": res["visualizations"]["heatmap"],
            })
        except Exception as err:
            items.append({"filename": f.filename, "error": str(err)})

    ok = [i for i in items if "error" not in i]
    return {
        "total_processed": len(items),
        "successful": len(ok),
        "failed": len(items) - len(ok),
        "stage_distribution": {CLASSES[i]: stage_counts[i] for i in range(5)},
        "results": items,
    }
