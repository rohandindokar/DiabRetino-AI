# Diabetic Retinopathy AI - FastAPI backend (PyTorch CPU build)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# System deps for imaging + healthcheck curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU wheels (compact; GPU build is local-only)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + real reference samples + trained weights (committed to git)
COPY app/ ./app/
COPY run.py .
COPY train.py .
COPY regen_report.py .
COPY weights/ ./weights/
COPY reports/ ./reports/

# The real APTOS reference samples and trained checkpoints ship in the image.
# If a weight file is ever absent, the inference engine falls back to a
# freshly-initialised architecture so the API still returns well-formed output.

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

EXPOSE 8000
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
