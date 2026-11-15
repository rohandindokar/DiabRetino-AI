Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Starting Diabetic Retinopathy AI (Hybrid ViT-CNN)" -ForegroundColor Green
Write-Host " Backend:  http://localhost:8000" -ForegroundColor Yellow
Write-Host " Frontend: http://localhost:3000" -ForegroundColor Yellow
Write-Host " Docs:     http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Start Backend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir/backend'; & '$scriptDir/backend/venv/Scripts/python.exe' run.py --host 0.0.0.0 --port 8000 --reload"
Start-Sleep -Seconds 3

# Start Frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir/frontend'; & '$scriptDir/backend/venv/Scripts/python.exe' -m http.server 3000"
Start-Sleep -Seconds 2

# Open browser
Start-Process "http://localhost:3000"
Write-Host "[✓] Services started successfully in new windows!" -ForegroundColor Green
