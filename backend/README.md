# Diabetic Retinopathy Severity Grading Backend API

Production-ready REST API implementing Attention-Based CNN (SE-ResNet18) and Vision Transformers (ViT) for 5-stage Diabetic Retinopathy classification, Grad-CAM visual attention mapping, and clinical reporting according to the ICDR scale.

## 🚀 Key Features

- **Hybrid Deep Learning Core**:
  - `HybridViTAttentionCNN`: Combines local attention convolutions (SE-ResNet18) with global multi-head self-attention (ViT Patch16).
  - `SEResNet18`: Squeeze-and-Excitation channel attention recalibration.
  - `VisionTransformerDR`: Pure Vision Transformer sequence model.
- **Explainable AI (XAI)**:
  - Dynamic Grad-CAM heatmap generation highlighting retinal lesion hotspots (microaneurysms, hemorrhages, cotton-wool spots, neovascularization).
- **Ben Graham Retinal Preprocessing**:
  - Auto-circular crop to eliminate black ocular aperture borders.
  - Local color subtraction / Gaussian contrast enhancement for illumination normalization.
- **Clinical ICDR Reporting**:
  - Severity stage, risk tier, urgency level, laser photocoagulation requirements, and actionable doctor recommendations.
- **RESTful Endpoints**:
  - Single prediction (`/api/predict`) with multipart image upload or sample selection.
  - Batch prediction (`/api/batch-predict`) for clinical batches.
  - Sample library (`/api/samples`) with instant pre-generated fundus images.
  - Weight upload (`/api/upload-weights`) for applying fine-tuned `.pth` checkpoints.

---

## 🛠️ Local Development & Quick Start

### 1. Set Up Virtual Environment
```bash
# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Sample Retinal Images
```bash
python app/samples/generate_samples.py
```

### 4. Run the Server
```bash
python run.py --host 0.0.0.0 --port 8000 --reload
```
Open your browser to:
- Interactive API Docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 🐳 Docker Deployment

### Build and Run with Docker
```bash
docker build -t diabretino-backend .
docker run -p 8000:8000 diabretino-backend
```

---

## 📡 API Reference Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Healthcheck, active device (CPU/CUDA), loaded models |
| `GET` | `/api/metadata` | ICDR clinical grading scale definitions & findings |
| `GET` | `/api/models` | Available model architectures & descriptions |
| `GET` | `/api/samples` | List of built-in sample images for instant testing |
| `POST` | `/api/predict` | Run DR inference on an uploaded image |
| `POST` | `/api/batch-predict` | Batch classification for multiple images |
| `POST` | `/api/upload-weights` | Upload custom PyTorch `.pth` checkpoint |
