"""
FastAPI REST API Routes for Diabetic Retinopathy Severity Grading.
"""
import os
import io
import traceback
import logging
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

from app.core.config import CLASSES, CLASS_DISPLAY_NAMES, SEVERITY_METADATA
from app.models.inference import DRInferenceEngine
from app.models.preprocessing import image_to_base64

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
engine = DRInferenceEngine()


@router.get("/health")
def health_check():
    """Returns system status, active PyTorch device, and loaded models."""
    try:
        return {
            "status": "healthy",
            "service": "Diabetic Retinopathy Classification API",
            "version": "1.0.0",
            "device": str(engine.device),
            "available_models": list(engine.models.keys()),
            "active_model": engine.active_model_name
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "detail": str(e)}
        )


@router.get("/metadata")
def get_metadata():
    """Returns ICDR clinical scale and staging criteria."""
    try:
        return {
            "classes": CLASSES,
            "class_display_names": CLASS_DISPLAY_NAMES,
            "severity_metadata": SEVERITY_METADATA
        }
    except Exception as e:
        logger.error(f"Metadata error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metadata")


@router.get("/models")
def get_models():
    """Returns information on available deep learning architectures."""
    try:
        return {
            "models": [
                {
                    "id": "hybrid_vit_cnn",
                    "name": "Hybrid ViT + Attention-CNN (Flagship)",
                    "description": "Combines SE-ResNet18 local attention convolutions with Vision Transformer (ViT-Patch16) global self-attention and gated feature fusion.",
                    "recommended": True
                },
                {
                    "id": "se_resnet18",
                    "name": "SE-ResNet18 (Attention-Based CNN)",
                    "description": "Convolutional neural network integrated with Squeeze-and-Excitation (SE) channel recalibration blocks.",
                    "recommended": False
                },
                {
                    "id": "vit_dr",
                    "name": "Vision Transformer (ViT-Patch16)",
                    "description": "Multi-head self-attention transformer operating over 16x16 retinal image patches.",
                    "recommended": False
                }
            ]
        }
    except Exception as e:
        logger.error(f"Models list error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model list")


@router.get("/samples")
def get_samples():
    """Returns built-in sample images for one-click testing in frontend."""
    samples_dir = os.path.join(os.path.dirname(__file__), "..", "samples")
    sample_files = [
        ("stage_0_no_dr.jpg", 0, "Stage 0 - Normal Retina"),
        ("stage_1_mild_dr.jpg", 1, "Stage 1 - Mild Microaneurysms"),
        ("stage_2_moderate_dr.jpg", 2, "Stage 2 - Moderate Hemorrhages/Exudates"),
        ("stage_3_severe_dr.jpg", 3, "Stage 3 - Severe Blot Hemorrhages"),
        ("stage_4_proliferative_dr.jpg", 4, "Stage 4 - Proliferative Neovascularization")
    ]

    results = []
    for fname, stage_id, label in sample_files:
        fpath = os.path.join(samples_dir, fname)
        if os.path.isfile(fpath):
            try:
                with Image.open(fpath) as img:
                    b64 = image_to_base64(img.resize((140, 140)))
                results.append({
                    "filename": fname,
                    "stage_id": stage_id,
                    "label": label,
                    "metadata": SEVERITY_METADATA[stage_id],
                    "thumbnail_base64": b64
                })
            except Exception as e:
                logger.warning(f"Could not load sample {fname}: {e}")

    return {"samples": results}


@router.get("/samples/{filename}")
def get_sample_file(filename: str):
    """Serves raw sample image file."""
    # Security: prevent path traversal
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    samples_dir = os.path.join(os.path.dirname(__file__), "..", "samples")
    fpath = os.path.join(samples_dir, safe_name)

    # Ensure the resolved path is within the samples directory
    real_path = os.path.realpath(fpath)
    real_samples = os.path.realpath(samples_dir)
    if not real_path.startswith(real_samples):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.isfile(real_path):
        raise HTTPException(status_code=404, detail="Sample image not found")

    return FileResponse(real_path, media_type="image/jpeg", filename=safe_name)


@router.post("/predict")
async def predict_single(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None),
    model_name: str = Form("hybrid_vit_cnn"),
    apply_enhancement: bool = Form(True)
):
    """
    Classifies a retinal fundus image:
    Returns severity level (0-4), confidence, probability distribution, Grad-CAM attention map,
    and ICDR clinical guidelines.
    """
    image_bytes = None

    if file is not None and file.filename:
        # Validate MIME type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must be a valid image (JPEG/PNG)"
            )
        # Validate file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {ext}. Use JPEG, PNG, or WebP."
            )
        # Limit file size to 20MB
        image_bytes = await file.read()
        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File too large. Maximum size is 20MB."
            )
    elif sample_filename:
        # Security: prevent path traversal
        safe_name = os.path.basename(sample_filename)
        samples_dir = os.path.join(os.path.dirname(__file__), "..", "samples")
        fpath = os.path.join(samples_dir, safe_name)
        real_path = os.path.realpath(fpath)
        real_samples = os.path.realpath(samples_dir)
        if not real_path.startswith(real_samples):
            raise HTTPException(status_code=403, detail="Access denied")
        if not os.path.isfile(real_path):
            raise HTTPException(status_code=404, detail="Specified sample file not found")
        with open(real_path, "rb") as f:
            image_bytes = f.read()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either an uploaded image or a sample_filename"
        )

    try:
        result = engine.predict(
            image_bytes=image_bytes,
            model_name=model_name,
            apply_enhancement=apply_enhancement
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@router.post("/batch-predict")
async def predict_batch(
    files: List[UploadFile] = File(...),
    model_name: str = Form("hybrid_vit_cnn"),
    apply_enhancement: bool = Form(True)
):
    """
    Batch classification for multiple retinal images.
    Returns per-image results and summary stage distribution.
    """
    if len(files) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch limit is 20 images at a time."
        )

    items = []
    stage_counts = {i: 0 for i in range(5)}

    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            items.append({
                "filename": f.filename or "unknown",
                "error": "Skipped: not an image file"
            })
            continue

        b = await f.read()
        if len(b) == 0:
            items.append({
                "filename": f.filename or "unknown",
                "error": "Empty file"
            })
            continue

        try:
            res = engine.predict(b, model_name=model_name, apply_enhancement=apply_enhancement)
            pred = res["prediction"]
            stage_counts[pred["class_id"]] += 1
            items.append({
                "filename": f.filename,
                "prediction": pred,
                "thumbnail": res["visualizations"]["original_image"],
                "heatmap": res["visualizations"]["heatmap"]
            })
        except Exception as err:
            logger.warning(f"Batch item failed for {f.filename}: {err}")
            items.append({
                "filename": f.filename or "unknown",
                "error": str(err)
            })

    successful = [i for i in items if "error" not in i]

    return {
        "total_processed": len(items),
        "successful": len(successful),
        "failed": len(items) - len(successful),
        "stage_distribution": {
            CLASSES[i]: stage_counts[i] for i in range(5)
        },
        "results": items
    }


@router.post("/upload-weights")
async def upload_weights(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    """Uploads custom trained PyTorch weights (.pth) for a given architecture."""
    if not (file.filename.endswith(".pth") or file.filename.endswith(".pt") or file.filename.endswith(".bin")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Weights file must have .pth, .pt, or .bin extension"
        )

    # Limit file size to 500MB for model weights
    contents = await file.read()
    if len(contents) > 500 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Weights file too large. Maximum size is 500MB."
        )

    if model_name not in engine.models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model '{model_name}'. Available: {list(engine.models.keys())}"
        )

    success = engine.load_custom_weights(contents, model_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to load and apply model state dictionary."
        )

    return {
        "success": True,
        "message": f"Successfully loaded weights for {model_name}"
    }
