"""
Diabetic Retinopathy Severity Grading API.
Flagship hybrid (SE-ResNet18 + Vision Transformer) attention CNN with
Grad-CAM explainability and ICDR clinical reporting.
"""
import logging
import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.api.routes import router, engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("diabretino")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Diabetic Retinopathy AI Severity Classifier - Starting")
    logger.info("Device: %s", engine.device)
    logger.info("Checkpoints loaded: %s", engine.model_loaded_from_checkpoint)
    logger.info("Startup in %.0f ms", (time.time() - t0) * 1000)
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Diabetic Retinopathy Severity Grading API",
    description=("Hybrid Attention-CNN + Vision Transformer classifier for 5-stage "
                 "ICDR diabetic retinopathy grading, with Grad-CAM explainability "
                 "and clinical reporting."),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    resp.headers["X-Process-Time-MS"] = f"{(time.time() - t0) * 1000:.1f}"
    return resp


@app.exception_handler(Exception)
async def global_error(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url, exc)
    return JSONResponse(500, {"detail": "Internal server error. Check server logs."})


app.include_router(router)

samples_dir = os.path.join(os.path.dirname(__file__), "samples")
if os.path.isdir(samples_dir):
    app.mount("/static/samples", StaticFiles(directory=samples_dir), name="samples")


@app.get("/")
def root():
    return {
        "message": "Diabetic Retinopathy AI Severity Classifier is online",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": [
            "/api/health", "/api/metadata", "/api/models",
            "/api/samples", "/api/metrics",
            "/api/predict (POST)", "/api/batch-predict (POST)",
        ],
    }
