"""
Diabetic Retinopathy Severity Grading API.
Combines Attention-Based CNN (SE-ResNet18) and Vision Transformers (ViT)
for clinical classification, Grad-CAM visual attention, and medical reporting.
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os

from app.api.routes import router, engine

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("diabretino")

# ---------------------------------------------------------------------------
# Lifespan — initialize models once at startup, clean up on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_begin = time.time()
    logger.info("=" * 60)
    logger.info("Diabetic Retinopathy AI Severity Classifier — Starting Up")
    logger.info("=" * 60)

    # Trigger model initialization (loads weights if available)
    logger.info(f"Device: {engine.device}")
    logger.info(f"Architectures: {list(engine.models.keys())}")

    load_time = round((time.time() - startup_begin) * 1000, 1)
    logger.info(f"Startup complete in {load_time} ms")
    logger.info(f"API docs available at /docs")

    yield  # Server is running

    logger.info("Shutting down DiabRetino AI API...")


app = FastAPI(
    title="Diabetic Retinopathy Severity Grading API",
    description="Optimal Detection of Diabetic Retinopathy Severity using Attention-Based CNN and Vision Transformers (ViT). "
                "Provides Grad-CAM explainability, ICDR clinical guidelines, and automated ophthalmic report generation.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for decoupled frontend deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)
    response.headers["X-Process-Time-MS"] = str(elapsed)
    return response

# Global exception handler for unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."}
    )

# Register API Router
app.include_router(router)

# Mount samples directory for direct static asset access
samples_dir = os.path.join(os.path.dirname(__file__), "samples")
if os.path.isdir(samples_dir):
    app.mount("/static/samples", StaticFiles(directory=samples_dir), name="samples")


@app.get("/")
def root():
    return {
        "message": "Diabetic Retinopathy AI Severity Classifier API is online",
        "docs": "/docs",
        "version": "2.0.0",
        "api_endpoints": {
            "health": "/api/health",
            "models": "/api/models",
            "metadata": "/api/metadata",
            "samples": "/api/samples",
            "predict": "/api/predict (POST)",
            "batch_predict": "/api/batch-predict (POST)",
            "upload_weights": "/api/upload-weights (POST)"
        }
    }
