# DiabRetino AI — Complete Deployment Guide

This guide covers every step to get the project live — from GitHub upload to a fully deployed, publicly accessible application.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Prepare Your Local Repo](#2-prepare-your-local-repo)
3. [Push to GitHub](#3-push-to-github)
4. [Verify Local Docker Deployment](#4-verify-local-docker-deployment)
5. [Deploy Backend to Render](#5-deploy-backend-to-render)
6. [Deploy Frontend to Vercel](#6-deploy-frontend-to-vercel)
7. [Connect Frontend to Backend](#7-connect-frontend-to-backend)
8. [Alternative: Full-Stack Docker on Render/Railway](#8-alternative-full-stack-docker-on-renderrailway)
9. [Post-Deployment Verification Checklist](#9-post-deployment-verification-checklist)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

Install these before you start:

| Tool | Purpose | Install |
|------|---------|---------|
| **Git** | Version control | https://git-scm.com/download/win |
| **GitHub Account** | Code hosting | https://github.com/signup |
| **Docker Desktop** | Local container testing | https://www.docker.com/products/docker-desktop/ |
| **Node.js** (optional) | Frontend dev server | https://nodejs.org/ |

Verify installations:
```bash
git --version
docker --version
docker compose version
node --version   # optional
```

---

## 2. Prepare Your Local Repo

### Step 2.1 — Navigate to the project folder

```bash
cd "D:\MIT\SEM 7\ANNA\ESE Project"
```

### Step 2.2 — Initialize git (if not already initialized)

```bash
git init
git add .
git commit -m "Initial commit: DiabRetino AI - DR Severity Grading with Hybrid ViT-CNN"
```

### Step 2.3 — Create a `.gitignore` (if not present)

The project already has a `.gitignore` that excludes:
- `venv/` — Python virtual environment
- `__pycache__/` — Python bytecode
- `data/` — the raw APTOS 2019 dataset (8+ GB, never committed; train locally)
- `.vscode/`, `.idea/` — IDE files

**The trained flagship checkpoint (`backend/weights/hybrid_vit_cnn.pth`, ~53 MB)
IS committed** so the Render build ships a working model. Only `data/` is kept
out of git.

### Step 2.4 — Verify what will be committed

```bash
git ls-files
```

You should see these key files listed:
```
docker-compose.yml
start_all.bat
start_all.ps1
.env.example
README.md
backend/Dockerfile
backend/requirements.txt
backend/run.py
backend/app/main.py
backend/app/api/routes.py
backend/app/core/config.py
backend/app/models/architectures.py
backend/app/models/gradcam.py
backend/app/models/inference.py
backend/app/models/preprocessing.py
backend/app/samples/stage_0_no_dr.jpg  ... stage_4_proliferative_dr.jpg
frontend/Dockerfile
frontend/nginx.conf
frontend/index.html
frontend/css/style.css
frontend/js/app.js
frontend/samples/stage_0_no_dr.jpg  ... stage_4_proliferative_dr.jpg
```

**Do NOT see**: `venv/`, `__pycache__/`, `*.pth`, `*.pyc` — these are gitignored.

---

## 3. Push to GitHub

### Step 3.1 — Create a new repository on GitHub

1. Go to https://github.com/new
2. Repository name: `DiabRetino-AI` (or your preferred name)
3. Set it to **Private** or **Public** (your choice)
4. **Do NOT** check "Add a README file" — you already have one
5. Click **Create repository**

### Step 3.2 — Link local repo to GitHub and push

```bash
cd "D:\MIT\SEM 7\ANNA\ESE Project"

# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/DiabRetino-AI.git

# Rename default branch to main
git branch -M main

# Push all code and history to GitHub
git push -u origin main
```

### Step 3.3 — Verify on GitHub

Open your repository URL in a browser and confirm all files are visible.

### Step 3.4 — Model weights ship in the repo

The trained flagship checkpoint is committed at `backend/weights/hybrid_vit_cnn.pth`
(~53 MB), so Render's Docker build picks it up automatically — **no separate upload
or Hugging Face step needed**. Only the raw `data/` dataset is kept out of git
because it is too large.

---

## 4. Verify Local Docker Deployment

Before deploying to the cloud, confirm everything works locally with Docker.

### Step 4.1 — Make sure Docker Desktop is running

Open Docker Desktop and confirm the engine is active.

### Step 4.2 — Build and start everything

```bash
cd "D:\MIT\SEM 7\ANNA\ESE Project"

# Build images and start containers (first build takes 5-10 min — downloads PyTorch)
docker compose up --build
```

### Step 4.3 — Verify the services

Open your browser:

| Service | URL | Expected Result |
|---------|-----|-----------------|
| Frontend | http://localhost:3000 | Clinical dashboard with 5 sample cards loads |
| Backend API Docs | http://localhost:8000/docs | Swagger UI with all endpoints |
| Health Check | http://localhost:8000/api/health | `{"status": "healthy", ...}` |

### Step 4.4 — Test a prediction

In the Swagger UI at `/docs`:
1. Open `POST /api/predict`
2. Click **Try it out**
3. In `sample_filename`, enter: `stage_4_proliferative_dr.jpg`
4. In `model_name`, enter: `hybrid_vit_cnn`
5. Click **Execute**
6. You should get a full JSON response with prediction, heatmap, overlay, and clinical guidelines

### Step 4.5 — Stop containers

```bash
# In the terminal where compose is running, press Ctrl+C
# Or from a new terminal:
cd "D:\MIT\SEM 7\ANNA\ESE Project"
docker compose down
```

---

## 5. Deploy Backend to Render

We'll deploy the FastAPI backend as a Docker container on Render.

### Step 5.1 — Push to GitHub first

Make sure your latest code is pushed (Step 3 above).

### Step 5.2 — Create a Render account

1. Go to https://render.com/register
2. Sign up with your GitHub account

### Step 5.3 — Create a new Web Service

1. Click **New +** → **Web Service**
2. Click **Build and deploy from a Git repository** → **Next**
3. Connect your GitHub account and authorize Render
4. Select your repository (`DiabRetino-AI`)
5. Configure:
   ```
   Name: diabretino-backend
   Region: Choose closest to you (e.g., Ohio / Frankfurt)
   Branch: main
   Root Directory: backend
   Runtime: Docker
   ```

### Step 5.4 — Configure the service

Under **Instance Type**:
- **Starter** ($7/month) — always-on, recommended for a live demo.
- (Render's free tier was discontinued, so budget the Starter plan for the backend.)

Under **Health Check Path**, set:
```
/api/health
```

### Step 5.5 — Add environment variables

Scroll to **Environment** and add:

| Key | Value |
|-----|-------|
| `PYTHONDONTWRITEBYTECODE` | `1` |
| `PYTHONUNBUFFERED` | `1` |

### Step 5.6 — Deploy

Click **Create Web Service**. Render will:
1. Build the Docker image (5-10 minutes first time)
2. Run `python app/samples/generate_samples.py` during build
3. Start the server on port 8000

You'll see build logs in real-time. When complete, Render gives you a URL like:
```
https://diabretino-backend.onrender.com
```

### Step 5.7 — Test the deployed backend

```bash
# Replace with your actual Render URL
curl https://diabretino-backend.onrender.com/api/health
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

### Step 5.8 — (Optional) Upload trained model weights

If you have trained `.pth` files:

1. In Render, go to your service → **Shell** tab
2. Download your weights file to the shell
3. Or use the `/api/upload-weights` endpoint from the Swagger docs

---

## 6. Deploy Frontend to Vercel

The frontend is a static site (pure HTML/CSS/JS) — Vercel deploys it in seconds.

### Step 6.1 — Create a Vercel account

1. Go to https://vercel.com/signup
2. Sign up with your GitHub account

### Step 6.2 — Import your project

1. Click **Add New...** → **Project**
2. Select your `DiabRetino-AI` repository
3. Vercel auto-detects settings. Override:

| Setting | Value |
|---------|-------|
| **Framework Preset** | Other |
| **Root Directory** | `frontend` |
| **Build Command** | *(leave empty)* |
| **Output Directory** | `.` |

### Step 6.3 — Deploy

Click **Deploy**. Vercel will deploy in ~30 seconds and give you a URL like:
```
https://diabretino-ai.vercel.app
```

### Step 6.4 — Test the frontend

Open the Vercel URL. You should see:
- The DiabRetino AI clinical dashboard
- 5 sample fundus cards in the gallery
- Dark theme applied
- API status showing "Backend Offline" (expected — you haven't connected the backend yet)

### Step 6.5 — Configure API URL in the frontend

1. On the frontend page, click the **Settings (gear) icon** in the top-right
2. In **Backend API Base URL**, paste your Render backend URL:
   ```
   https://diabretino-backend.onrender.com
   ```
3. Click **Test Server Connection** — should show "Connected successfully!"
4. Click **Save Configuration**
5. Refresh the page

---

## 7. Connect Frontend to Backend

### Step 7.1 — Verify the API URL is saved

The frontend saves the API URL to `localStorage` in the browser. Once saved, it persists across page reloads.

To change it later:
- Open the app → Click gear icon → Update URL → Save

### Step 7.2 — Test end-to-end

1. Open your Vercel frontend URL
2. Confirm the API status badge shows **"Online (CPU)"** in green
3. Click **Stage 4 - Proliferative Neovascularization** sample card
4. After ~2 seconds, you should see:
   - Diagnosis: "Proliferative Diabetic Retinopathy"
   - Confidence bar and probability breakdown
   - Grad-CAM heatmap overlay (adjustable opacity)
   - ICDR clinical criteria and findings
   - Recommended clinical action
   - Laser warning badge (red)

### Step 7.3 — (Optional) Set a custom domain on Vercel

1. In Vercel dashboard → your project → **Settings** → **Domains**
2. Add your custom domain (e.g., `diabretino.yourdomain.com`)
3. Follow Vercel's DNS configuration instructions

---

## 8. Alternative: Full-Stack Docker on Render/Railway

If you want to deploy both backend and frontend together on a single server:

### Option A: Render with Docker Compose (via Blueprint)

1. Create a file `render.yaml` in your repo root:

```yaml
services:
  - type: web
    name: diabretino-backend
    env: docker
    region: ohio
    plan: free
    branch: main
    rootDir: backend
    healthCheckPath: /api/health
    envVars:
      - key: PORT
        value: 8000

  - type: web
    name: diabretino-frontend
    env: docker
    region: ohio
    plan: free
    branch: main
    rootDir: frontend
    envVars:
      - key: VITE_API_URL
        value: https://diabretino-backend.onrender.com
```

2. In Render, click **New +** → **Blueprint**
3. Connect your repo and deploy

### Option B: Railway

1. Go to https://railway.app
2. Sign up with GitHub
3. Click **New Project** → **Deploy from GitHub repo**
4. Select your repository
5. Railway auto-detects `docker-compose.yml` and deploys both services
6. Set environment variables in Railway dashboard

### Option C: Single VPS (DigitalOcean / AWS / GCP)

```bash
# On your VPS (Ubuntu 22.04+)
# 1. Install Docker
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker

# 2. Clone your repo
git clone https://github.com/YOUR_USERNAME/DiabRetino-AI.git
cd DiabRetino-AI

# 3. Start with docker compose
docker compose up -d

# 4. Check logs
docker compose logs -f
```

With nginx as reverse proxy on port 80:
```bash
# Install nginx
sudo apt install nginx -y

# Configure /etc/nginx/sites-available/diabretino
# Proxy / to port 3000, /api/ to port 8000
# (See the enhanced nginx.conf already in the project)

sudo systemctl reload nginx
```

---

## 9. Post-Deployment Verification Checklist

Go through every item to confirm everything works:

### Backend Checks
- [ ] `GET https://your-backend.com/api/health` returns 200 with `"status": "healthy"`
- [ ] `GET https://your-backend.com/api/models` returns 3 architectures
- [ ] `GET https://your-backend.com/api/metadata` returns 5 ICDR stages
- [ ] `GET https://your-backend.com/api/samples` returns 5 sample images
- [ ] `POST /api/predict` with `sample_filename=stage_0_no_dr.jpg` returns prediction with heatmap
- [ ] `POST /api/predict` with `sample_filename=stage_4_proliferative_dr.jpg` returns prediction + laser warning
- [ ] `POST /api/batch-predict` with multiple images returns batch results
- [ ] Swagger docs at `/docs` load and are interactive

### Frontend Checks
- [ ] Page loads at your Vercel/Netlify URL
- [ ] 5 sample cards appear in the gallery at the top
- [ ] API status badge shows "Online" in green
- [ ] Dark/light theme toggle works and persists on refresh
- [ ] Clicking a sample card triggers analysis and shows results
- [ ] Drag-and-drop image upload works
- [ ] Clipboard paste (Ctrl+V) works
- [ ] Grad-CAM overlay renders with adjustable opacity slider
- [ ] Side-by-side comparison view works
- [ ] Probability distribution bars animate
- [ ] Clinical report (ICDR criteria, findings, recommended action) displays
- [ ] Batch screening mode works (upload multiple images)
- [ ] Print/Save PDF report works
- [ ] Settings modal saves and persists API URL
- [ ] Responsive on mobile viewport (375px width)

### Integration Checks
- [ ] Frontend successfully calls deployed backend (not localhost)
- [ ] CORS headers allow frontend origin
- [ ] Sample images load from backend `/api/samples/` endpoint
- [ ] Prediction latency is acceptable (<5s per image on CPU)

---

## 10. Troubleshooting

### Docker build fails with "permission denied"
```bash
# On Windows, run PowerShell as Administrator
# Or enable WSL 2 integration in Docker Desktop settings
```

### Backend health check fails on Render
- Check the build logs for errors
- Ensure `curl` is installed (it is, in the Dockerfile)
- Make sure `start-period` is 15s (model loading takes time)

### Frontend shows "Backend Offline"
- Open Settings (gear icon) and verify the API URL
- Ensure backend URL uses `https://` (not `http://`)
- Check that CORS is set to `allow_origins=["*"]` in `main.py`
- Test the backend URL directly in browser: `https://your-backend.com/api/health`

### Predictions return "Moderate NPDR" for everything
- This is expected when no trained weights are loaded (random init)
- Upload your trained `.pth` weights via `/api/upload-weights` in Swagger docs
- Or place `.pth` files in `backend/weights/` before Docker build

### Docker Compose "port already in use"
```bash
# Change ports in docker-compose.yml:
#   - "3001:80"   for frontend
#   - "8001:8000" for backend
```

### Vercel deployment fails
- Ensure **Root Directory** is set to `frontend`
- Ensure **Output Directory** is `.` (dot, not empty)
- Do NOT set a build command for a static site

### Backend cold start on Render
- First request after the service idles may take 30-60 seconds (model loading)
- The Starter plan keeps the service warm more reliably than a free tier would

### Weight file is missing at deploy time
- If `backend/weights/hybrid_vit_cnn.pth` is absent, the inference engine falls
  back to a freshly-initialised model so the API still returns well-formed
  (but untrained) predictions. To ship real predictions, commit the `.pth`.

---

## Quick Reference

| What | Where | Cost |
|------|-------|------|
| GitHub repo + committed model weights | github.com | Free |
| Backend (Render) | render.com | Starter $7/month (free tier discontinued) |
| Frontend (Vercel) | vercel.com | Free (hobby plan, 100GB bandwidth) |
| Custom domain (optional) | Any registrar | ~$10/year |

**Total to go live: ~$7/month** (Render Starter backend + free Vercel frontend).
The trained model ships inside the repo, so there is no separate weights-hosting
cost.
