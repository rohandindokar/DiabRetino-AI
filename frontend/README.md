# Diabetic Retinopathy Diagnostic GUI Frontend

Modern, clinical-grade web application for real-time diabetic retinopathy severity grading and Grad-CAM visual attention mapping.

## ✨ Highlights

- **Aesthetic Glassmorphic UI**: High-contrast ophthalmic clinical palette with dark/light themes.
- **1-Click Clinical Sample Gallery**: Test all 5 ICDR disease stages (Stage 0 to Stage 4) instantly.
- **Interactive Grad-CAM Heatmap Viewer**: Dynamic opacity slider to blend lesion heatmaps over fundus images.
- **Side-by-Side Comparison**: Original retina vs Ben Graham local color subtraction filter vs Attention map.
- **Batch Screening Mode**: Upload up to 20 retinal scans with automated distribution summary.
- **Printable Medical Examination Report**: Doctor sign-off, patient ID, and clinical action checklist.
- **Configurable Backend API URL**: Deploy independently on Vercel, Netlify, Cloudflare Pages, or Docker.

---

## 🚀 Running Locally

### Option 1: Python HTTP Server (Zero Install)
```bash
python -m http.server 3000
```
Open [http://localhost:3000](http://localhost:3000)

### Option 2: Node.js (via npm)
```bash
npm start
```

### Option 3: VS Code Live Server
Right click `index.html` -> "Open with Live Server".

---

## 🐳 Docker Deployment

```bash
docker build -t diabretino-frontend .
docker run -p 3000:80 diabretino-frontend
```
Open [http://localhost:3000](http://localhost:3000)

---

## ☁️ Cloud Static Hosting (Vercel / Netlify / Cloudflare Pages)

1. Deploy the `frontend/` folder to Vercel or Netlify.
2. In the web interface, click the **Settings (gear)** icon in the top right.
3. Enter your deployed cloud backend URL (e.g. `https://your-backend-api.onrender.com`).
4. Click **Save Configuration**.
