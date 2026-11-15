# Optimal Detection of Diabetic Retinopathy Severity Using Attention-Based CNN and Vision Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![API: FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker: Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

A production-ready, clinical-grade medical AI system combining **Attention-Based Convolutional Neural Networks (SE-ResNet18)** and **Vision Transformers (ViT)** to classify the severity levels of **Diabetic Retinopathy (DR)** into 5 clinical stages with **Grad-CAM visual attention mapping**, **Ben Graham retinal preprocessing**, and **automated ophthalmic report generation**.

---

## 📸 Architecture & Clinical Severity Pipeline

```
  Input Retinal Fundus Scan (224×224 RGB)
                  │
     ┌────────────┴────────────┐
     ▼                         ▼
Ben Graham Local          Auto-Circular
Color Subtraction         Aperture Crop
     │                         │
     └────────────┬────────────┘
                  ▼
   ┌──────────────────────────────┐
   │     HYBRID DUAL BACKBONE     │
   ├──────────────────────────────┤
   │  Branch 1: SE-ResNet18       │ ──► Local lesion textures (microaneurysms, blot hemorrhages)
   │  Branch 2: Vision Transformer│ ──► Global retinal context (optic disc & vascular arcade)
   └──────────────┬───────────────┘
                  ▼
     Gated Multi-Modal Fusion (768-d)
                  │
         ┌────────┴────────┐
         ▼                 ▼
   Softmax (5 Classes)  Grad-CAM Visual Attention
   [Stage 0 to Stage 4] (Heatmap Lesion Hotspots)
         │                 │
         └────────┬────────┘
                  ▼
  Clinical ICDR Report & Diagnostic Guidelines
```

### The 5 Clinical Severity Stages (ICDR Standard)
1. **Stage 0 — No DR**: Normal retina; no microaneurysms or vascular abnormalities. (Routine 12-month follow-up)
2. **Stage 1 — Mild NPDR**: Microaneurysms only. Earliest clinically visible phase. (6–12 month review)
3. **Stage 2 — Moderate NPDR**: Microaneurysms, hard exudates (lipid deposits), and blot hemorrhages. (3–6 month ophthalmology referral)
4. **Stage 3 — Severe NPDR**: 4-2-1 rule (>20 intraretinal hemorrhages in 4 quadrants, venous beading in 2+ quadrants, prominent IRMA). (2–4 week urgent specialist referral)
5. **Stage 4 — Proliferative DR (PDR)**: Neovascularization (NVD/NVE), preretinal/vitreous hemorrhages. (Immediate 24–48h referral for laser photocoagulation / anti-VEGF)

---

## 📁 Repository Structure

```
.
├── backend/                         # Deployable FastAPI Deep Learning Backend
│   ├── app/
│   │   ├── api/routes.py            # REST endpoints (predict, batch, samples, weights)
│   │   ├── core/config.py           # ICDR clinical criteria, findings & metadata
│   │   ├── models/
│   │   │   ├── architectures.py     # SEBlock, SEResNet18, ViT, HybridViTAttentionCNN
│   │   │   ├── gradcam.py           # Grad-CAM heatmap generator & colormap blender
│   │   │   ├── inference.py         # Multi-model inference manager & clinical report
│   │   │   └── preprocessing.py     # Ben Graham contrast filter & circle cropper
│   │   ├── samples/                 # Pre-generated clinical fundus samples (Stage 0-4)
│   │   └── main.py                  # FastAPI app with CORS & static mounting
│   ├── run.py                       # CLI server runner
│   ├── requirements.txt             # Production dependencies
│   ├── Dockerfile                   # Containerized backend
│   └── README.md
│
├── frontend/                        # Deployable Ophthalmic GUI Frontend
│   ├── css/style.css                # Glassmorphic clinical dark/light design system
│   ├── js/app.js                    # UI controller, Grad-CAM slider, batch engine
│   ├── samples/                     # 1-click test fundus samples (Stage 0-4)
│   ├── index.html                   # Responsive clinical dashboard
│   ├── nginx.conf                   # Nginx web server configuration
│   ├── Dockerfile                   # Containerized frontend
│   └── README.md
│
├── repo/                            # Original cloned research repository & notebook
│   ├── DiabRetino.ipynb
│   └── README.md
│
├── docker-compose.yml               # One-command full-stack container deployment
├── start_all.bat                    # Windows 1-click launcher
├── start_all.ps1                    # PowerShell 1-click launcher
└── README.md                        # Documentation
```

---

## ⚡ Quick Start

### Option 1: 1-Click Launch on Windows
Double-click `start_all.bat` (or execute `start_all.ps1`).  
This automatically:
1. Starts the FastAPI backend at `http://localhost:8000`
2. Starts the frontend GUI at `http://localhost:3000`
3. Opens the application in your default web browser!

---

### Option 2: Docker Compose (Production Full-Stack)
Run both backend and frontend isolated in Docker containers:
```bash
docker compose up --build
```
- Frontend GUI: [http://localhost:3000](http://localhost:3000)
- Backend API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option 3: Manual Step-by-Step

#### 1. Start the Backend
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python app/samples/generate_samples.py
python run.py --host 0.0.0.0 --port 8000 --reload
```

#### 2. Start the Frontend
In a new terminal:
```bash
cd frontend
python -m http.server 3000
```
Open your browser at `http://localhost:3000`.

---

## 🖥️ Frontend GUI Features

- **1-Click Clinical Sample Gallery**: Test all 5 ICDR disease stages immediately with built-in retinal fundus scans.
- **Drag & Drop / Clipboard Paste**: Drop any fundus image or paste directly from clipboard (`Ctrl+V`).
- **Interactive Grad-CAM Heatmap Viewer**: Real-time slider to adjust heatmap blending opacity from 0% to 100%.
- **Side-by-Side Comparison**: Original retina vs. Ben Graham Enhanced vs. Grad-CAM attention hotspots.
- **Batch Screening Mode**: Upload up to 20 images at once for automated screening with summary statistics.
- **Medical Examination Report**: Formatted clinical report with Patient ID, scan timestamp, model confidence, and ophthalmologist signature block, ready to print or export to PDF.
- **Configurable Backend API URL**: Built-in settings modal to connect the frontend to any deployed cloud backend.

---

## 🌐 Deploying to the Cloud

### Backend Deployment (Render / Railway / Fly.io / AWS ECS)
1. Point your cloud platform to the `backend/` directory.
2. Select Docker build or Python 3.12 environment with `pip install -r requirements.txt`.
3. Set start command: `python run.py --host 0.0.0.0 --port $PORT`

### Frontend Deployment (Vercel / Netlify / Cloudflare Pages)
1. Point your static host to the `frontend/` directory.
2. Build command: None (pure static HTML/CSS/JS).
3. Publish directory: `frontend`.
4. Open the deployed website, click the **Settings** icon, and paste your cloud backend API URL.

---

## 📜 Citation & Credits

Original research project: [Optimal Detection of Diabetic Retinopathy Severity Using Attention-Based CNN and Vision Transformers](https://github.com/ChaitanyaK77/Optimal-Detection-of-Diabetic-Retinopathy-Severity-Using-Attention-Based-CNN-and-Vision-Transformers).

## 🧪 Testing & Verification

### Reference Images (Test Samples)
The project ships with **5 generated reference fundus images** representing each ICDR severity stage:

| File | Stage | Description |
|------|-------|-------------|
| `backend/app/samples/stage_0_no_dr.jpg` | 0 | Clean retina, no lesions |
| `backend/app/samples/stage_1_mild_dr.jpg` | 1 | Isolated microaneurysms (tiny dark dots) |
| `backend/app/samples/stage_2_moderate_dr.jpg` | 2 | Hard exudates (yellow spots) + blot hemorrhages |
| `backend/app/samples/stage_3_severe_dr.jpg` | 3 | Dense 4-quadrant hemorrhages + cotton-wool spots |
| `backend/app/samples/stage_4_proliferative_dr.jpg` | 4 | Neovascularization fronds + vitreous hemorrhage |

**To test**: Click any sample card in the gallery at the top of the dashboard — the app will send it to the backend for analysis and display severity classification, confidence scores, Grad-CAM heatmap, and clinical guidelines.

### Health Check Endpoint
```bash
curl http://localhost:8000/api/health
```
Expected response:
```json
{
    "status": "healthy",
    "service": "Diabetic Retinopathy Classification API",
    "version": "2.0.0",
    "device": "cpu",
    "available_models": ["se_resnet18", "vit_dr", "hybrid_vit_cnn"],
    "active_model": "hybrid_vit_cnn"
}
```

### API Documentation
Open `http://localhost:8000/docs` in your browser to access the interactive Swagger UI where you can test all endpoints directly.

### Frontend Tests
1. Open `http://localhost:3000` — confirm the page loads with 5 sample fundus cards
2. Click **Stage 0** sample — should show "No Diabetic Retinopathy" with high confidence
3. Click **Stage 4** sample — should show "Proliferative DR" with laser warning badge
4. Drag and drop any JPEG/PNG — should display the image and enable analysis
5. Switch tabs to **Batch Screening** — upload multiple images
6. Toggle **Dark/Light mode** — should persist across refresh
7. Open **Settings** and test a custom API URL connection

---

## 📊 Clinical Reference Materials

### ICDR Severity Scale (International Clinical Diabetic Retinopathy Scale)

| Stage | Name | Key Findings | Recommended Action |
|-------|------|-------------|-------------------|
| 0 | No DR | Normal fundus, no microaneurysms | Annual routine screening |
| 1 | Mild NPDR | Microaneurysms only | 6–12 month review, optimize HbA1c |
| 2 | Moderate NPDR | Hard exudates, cotton-wool spots, hemorrhages | OCT, ophthalmology referral in 3-6 months |
| 3 | Severe NPDR | 4-2-1 rule: 4-quadrant hemorrhages, venous beading, IRMA | Urgent specialist referral 2-4 weeks |
| 4 | Proliferative DR | NVD/NVE, vitreous hemorrhage, fibrovascular proliferation | PRP laser / anti-VEGF within 24-48 hours |

### Reference Literature
1. Wilkinson, C.P., et al. (2003). "Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales." *Ophthalmology*, 110(9), 1677-1682.
2. Graham, B. (2015). "Kaggle Diabetic Retinopathy Detection — Ben Graham's Preprocessing." *Kaggle Competition*.
3. Hu, J., Shen, L., & Sun, G. (2018). "Squeeze-and-Excitation Networks." *CVPR 2018*.
4. Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*.
5. Selvaraju, R.R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV 2017*.

---

## 🔧 Troubleshooting

### Backend won't start
- Ensure Python 3.10+ is installed
- Run `pip install --upgrade pip` then `pip install -r requirements.txt`
- If PyTorch fails, install separately: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

### Frontend can't connect to backend
- Check the API URL in Settings modal (default: `http://localhost:8000`)
- If using Docker, the API URL should be `http://localhost:8000`
- Ensure CORS is configured correctly in `backend/app/main.py`

### Docker build is slow
- First build downloads PyTorch (~200MB). Subsequent builds use Docker layer cache.
- Use `--build` flag only on first build or after dependency changes.

### Images look wrong
- Ensure sample images are generated: `python backend/app/samples/generate_samples.py`
- Check that `frontend/samples/` contains 5 `.jpg` files
