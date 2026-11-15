@echo off
title Diabetic Retinopathy AI Launcher
echo =====================================================================
echo  Starting Diabetic Retinopathy AI (Hybrid ViT-CNN)
echo  Backend:  http://localhost:8000
echo  Frontend: http://localhost:3000
echo  Docs:     http://localhost:8000/docs
echo =====================================================================

start "DiabRetino Backend" cmd /k "cd /d "%~dp0backend" && venv\Scripts\python.exe run.py --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 >nul

start "DiabRetino Frontend" cmd /k "cd /d "%~dp0frontend" && ..\backend\venv\Scripts\python.exe -m http.server 3000"
timeout /t 2 >nul

start http://localhost:3000
echo [✓] Both servers running! Press any key in the server windows to close.
