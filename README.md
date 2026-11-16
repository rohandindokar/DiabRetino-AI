# DiabRetino AI — Diabetic Retinopathy Severity Grading

A clinical-grade web app that classifies retinal fundus photographs into the five
ICDR diabetic-retinopathy (DR) severity stages, using a **hybrid deep-learning
model** that combines an attention-based CNN with a Vision Transformer, and shows
*where the model was looking* with Grad-CAM attention maps.

```
Retinal fundus scan (224×224 RGB)
   │
   ├── Ben Graham local color-subtraction (contrast enhancement)
   │
   ▼
┌────────────────────────────┐
│   HYBRID DUAL BACKBONE     │
│  Branch 1: SE-ResNet18     │  ← local lesion texture (microaneurysms,
│  Branch 2: Vision Transform│     exudates, hemorrhages)
└──────────────┬─────────────┘  ← global vascular context
   Gated feature fusion
   │
   ▼
5-class softmax  ──►  ICDR stage  +  confidence  +  Grad-CAM heatmap
                       +  clinical guidelines  +  printable report
```

---

## 🎯 What this project actually does (the short version)

Diabetic retinopathy is the leading cause of vision loss among working-age adults
with diabetes. Doctors grade it manually across five ICDR stages (0 = no DR …
4 = proliferative). This project automates that grading:

1. **Input** — a single retinal photograph (or a batch of up to 20).
2. **Preprocessing** — circular crop + Ben Graham contrast enhancement to make
   microaneurysms and exudates stand out.
3. **Classification** — a trained hybrid model outputs a 5-class probability
   distribution and the most likely ICDR stage.
4. **Explainability** — a Grad-CAM heatmap overlays the exact retinal regions
   (bleeds, exudates, neovascular fronds) that drove the call, so a clinician
   can see *why* the model decided.
5. **Clinical reporting** — ICDR criteria, identified findings, recommended
   action, and urgency level are returned as structured data and rendered as a
   printable medical report.

> ⚠️ **This is a research / educational screening tool, not a medical device.**
> It supports, and does not replace, an ophthalmologist.

---

## 🧠 The model & the work that was done

The flagship architecture is **`hybrid_vit_cnn`**:

- **SE-ResNet18 branch** — Squeeze-and-Excitation ResNet18 (ImageNet-pretrained).
  The SE blocks recalibrate channels so the network emphasises the few
  retinal-feature channels that matter. Catches fine local texture:
  microaneurysms, hard exudates, cotton-wool spots.
- **Vision Transformer branch** — a compact ViT (16×16 patches, 4 encoder
  layers) that models global spatial relationships: optic-disc position,
  vascular arcade layout, and whether neovascularization is spreading.
- **Gated fusion head** — the two feature vectors (512 + 192) are concatenated,
  passed through a learned sigmoid gate, then through a small MLP to the 5
  ICDR classes. Gating lets the model learn *when* to trust the CNN vs the
  Transformer for a given image.

**Training was done on the APTOS 2019 dataset** (3,662 labelled fundus images,
5 classes, with class weighting to handle the imbalance — stage 0 is 49% of the
data, stage 3 only 5%). Because the dataset is class-imbalanced, an inverse-
frequency `CrossEntropyLoss` weighting is used so the rare severe classes are not
swamped.

Two more models are served for comparison (`se_resnet18`, `vit_dr`) so a viewer
can switch architectures live in the UI.

### What the sample images are
The reference gallery at the top of the app is populated with **one real APTOS
2019 fundus photo per stage** (extracted from the training set, not synthetic):

| Stage | File | What you can see |
|-------|------|------------------|
| 0 | `stage_0_no_dr.png` | Clean retina, healthy optic disc, no lesions |
| 1 | `stage_1_mild.png` | Scattered microaneurysms (tiny red dots) |
| 2 | `stage_2_moderate.png` | Hard exudates + intraretinal hemorrhages |
| 3 | `stage_3_severe.png` | Dense exudates, 4-quadrant hemorrhages |
| 4 | `stage_4_proliferative.png` | Neovascularization / vitreous bleed |

If the real dataset is not present (e.g. inside a Docker build), the gallery
falls back to clearly-labelled **synthetic** fundus images so the UI still
demonstrates the full pipeline.

### Where to get more test images
- **APTOS 2019** (used here): https://www.kaggle.com/competitions/aptos2019-blindness-detection
- **EyePACS** (DR with macular edema): https://www.eyepacs.org/resources/data-downloads/eyepacs-database
- **OCT-DR / IDRD** (multi-modal OCT + DR): https://ieee-fog2018.drive.gr/ / https://www.eye.ku.nl/research/ophthalmic-imaging-ai-ophthalmology-ai-research
- **ESAR / MESSIDOR-2**: classic NPDR datasets for the older 5-level ICDR coding.

---

## 📊 Measurable impact / metrics

After training, per-architecture metrics are written to
[`backend/reports/metrics_report.json`](backend/reports/metrics_report.json).
The live values are also served by `GET /api/metrics` and shown in the UI's
"Model performance" panel. From the latest training run on an NVIDIA RTX 3050:

### Flagship — `hybrid_vit_cnn` (gated SE-ResNet18 + ViT fusion)

| Metric | Value |
|--------|-------|
| **Held-out validation accuracy** | **74.6%** |
| Macro F1 / Weighted F1 | 53.7% / 73.2% |
| Macro Precision / Recall | 56.7% / 58.5% |
| Training set size | 2,929 train / 733 held-out val (3,662 APTOS) |
| Device / time | NVIDIA RTX 3050, ~96 min (15 epochs) |

**Per-class performance on the held-out validation set** ("how many times it
was right" for each stage):

| Stage | Precision | Recall | F1 | # samples (val) |
|-------|-----------|--------|-----|-----------|
| No DR | 94.7% | **98.9%** | 96.8% | 361 |
| Mild | 50.6% | 60.8% | 55.2% | 74 |
| Moderate | 65.9% | 57.0% | 61.1% | 200 |
| Severe | 32.1% | 69.2% | 43.9% | 39 |
| Proliferative | 40.0% | 6.8% | 11.6% | 59 |

**Confusion matrix** (rows = actual, columns = predicted, held-out val):

```
              NoDR  Mild  Mod   Sev   Pro
No_DR           357    2   2   0     0
Mild             7   45  22   0     0
Moderate        11   31 114  38    6
Severe            0   5   7  27    0
Proliferative    2   6  28  19     4
```

> **Reading these numbers honestly:** The model is strong where the data is dense
> (No DR: 98% recall) and degrades on the rarer, visually-overlapping severe and
> proliferative stages — neovascularization (stage 4) is frequently confused with
> severe NPDR (stage 3), which is a well-known hard case in DR grading. Recall on
> the critical *proliferative* class is the weakest point and the main avenue for
> future improvement (more stage-4 examples / class-balanced augmentation).

A short, human-readable summary of these numbers is generated at
[`reports/REPORT.md`](reports/REPORT.md) (run `python make_report.py` to regenerate).
The full machine-readable report is [`backend/reports/metrics_report.json`](backend/reports/metrics_report.json).

---

## 📁 Repository structure

```
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app, CORS, lifespan, timing
│   │   ├── api/routes.py          # /predict, /batch, /samples, /metrics, /models
│   │   ├── core/config.py         # ICDR classes + clinical metadata
│   │   ├── models/
│   │   │   ├── architectures.py   # SE-ResNet18, ViT, Hybrid fusion
│   │   │   ├── gradcam.py         # Grad-CAM explainability
│   │   │   ├── inference.py       # checkpoint loading + clinical report
│   │   │   └── preprocessing.py   # Ben Graham + circular crop
│   │   └── samples/               # real APTOS reference images (one per stage)
│   ├── train.py                   # full training + evaluation pipeline
│   ├── run.py                     # server entry point
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── weights/                   # trained .pth checkpoints (committed)
│   └── reports/                   # metrics_report.json + confusion matrices
│
├── frontend/
│   ├── index.html / css/ / js/    # clinical dashboard (vanilla, no build step)
│   ├── samples/                   # reference fundus images
│   ├── nginx.conf / Dockerfile    # docker-compose static serving
│   └── vercel.json                # Vercel static deployment
│
├── data/                          # APTOS 2019 (NOT in git - too large)
│   ├── train_images/  test_images/  train.csv  test.csv
│
├── reports/REPORT.md              # human-readable metrics summary
├── docker-compose.yml
├── DEPLOY.md                      # step-by-step cloud deployment
└── README.md
```

---

## 🚀 Run locally

### 1. Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# (one-time) extract the reference sample images from the dataset
python app/samples/generate_samples.py

# train the model  (needs the data/ folder + a GPU for ~20-40 min)
python train.py

# start the API
python run.py                    # -> http://localhost:8000/docs
```

### 2. Frontend
```bash
cd frontend
python -m http.server 3000       # -> http://localhost:3000
```
Then open http://localhost:3000 — the app auto-detects the local backend.

### Docker (full stack)
```bash
docker compose up --build
# Frontend  http://localhost:3000
# API docs  http://localhost:8000/docs
```

---

## ☁️ Deploying

- **Backend → Render** (Web Service, Docker, root dir `backend`, health check `/api/health`).
- **Frontend → Vercel** (static site, root dir `frontend`).
- Open the app, click **⚙ API settings**, and set the backend base URL to your
  Render service. See [`DEPLOY.md`](DEPLOY.md) for the full walkthrough.

CORS is open (`allow_origins=["*"]`) so the browser can call your Render backend
directly from the Vercel frontend.

---

## 📚 References

1. Wilkinson et al., *Proposed International Clinical Diabetic Retinopathy
   and Diabetic Macular Edema Disease Severity Scales* (ICDR), Ophthalmology, 2003.
2. Graham, B., *APTOS 2019 Blindness Detection* — Kaggle competition (dataset).
3. Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018 (SE blocks).
4. Dosovitskiy et al., *An Image is Worth 16×16 Words: ViT*, ICLR 2021.
5. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017.
6. He et al., *Deep Residual Learning (ResNet)*, CVPR 2016.

### Original research base
Adapted from: [Optimal Detection of Diabetic Retinopathy Severity Using
Attention-Based CNN and Vision Transformers](https://github.com/ChaitanyaK77/Optimal-Detection-of-Diabetic-Retinopathy-Severity-Using-Attention-Based-CNN-and-Vision-Transformers).
